import numpy as np
import torch
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_dense_2d import EnvDense2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class EnvDense2DExtraObjects(EnvBase):
    """
    Dynamic dense 2D environment with obstacle dropout and obstacle injection.
    """
    def __init__(self, 
                 name='EnvDense2DExtraObjects', 
                 tensor_args=None, 
                 drop_old_num=6,        
                 num_extra_spheres=5,   
                 num_extra_boxes=8,     
                 seed=42, 
                 **kwargs):
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS

        rng = np.random.default_rng(seed)
        
        # Build a temporary Dense2D env and read its fixed obstacles.
        dummy_kwargs = kwargs.copy()
        dummy_kwargs['precompute_sdf_obj_fixed'] = False
        dummy_env = EnvDense2D(tensor_args=tensor_args, **dummy_kwargs)
        
        all_base_centers = []
        all_base_radii = []
        
        if dummy_env.obj_fixed_list is not None:
            for obj_field in dummy_env.obj_fixed_list:
                fields = obj_field.fields if hasattr(obj_field, 'fields') else [obj_field]
                for field in fields:
                    if isinstance(field, MultiSphereField):
                        centers = field.centers.cpu().numpy()
                        radii = field.radii.cpu().numpy()
                        all_base_centers.extend(centers)
                        all_base_radii.extend(radii)
                        
        all_base_centers = np.array(all_base_centers)
        all_base_radii = np.array(all_base_radii)
        
        # Randomly drop a subset of base obstacles.
        total_base = len(all_base_centers)
        keep_num = max(0, total_base - drop_old_num)
        
        indices = np.arange(total_base)
        rng.shuffle(indices)
        keep_indices = indices[:keep_num]
        drop_indices = indices[keep_num:]

        kept_centers = all_base_centers[keep_indices] if total_base > 0 else np.empty((0, 2))
        kept_radii = all_base_radii[keep_indices] if total_base > 0 else np.empty((0,))
        
        # Removed obstacles are kept for optional visualization only.
        self.dropped_centers = all_base_centers[drop_indices] if total_base > 0 else np.empty((0, 2))
        self.dropped_radii = all_base_radii[drop_indices] if total_base > 0 else np.empty((0,))
        
        # Rejection-sample extra obstacles in free space.
        bounds = [-0.9, 0.9]
        extra_sphere_centers, extra_sphere_radii = [], []
        extra_box_centers, extra_box_sizes = [], []
        
        occupied_centers = kept_centers.copy() if keep_num > 0 else np.empty((0, 2))

        def is_safe_location(pt, existing_pts, min_dist=0.3):
            if len(existing_pts) == 0: return True
            dists = np.linalg.norm(existing_pts - pt, axis=1)
            return np.all(dists > min_dist)

        for _ in range(num_extra_spheres):
            for _ in range(50):
                pt = rng.uniform(bounds[0], bounds[1], size=2)
                if is_safe_location(pt, occupied_centers, min_dist=0.3):
                    extra_sphere_centers.append(pt)
                    extra_sphere_radii.append(rng.uniform(0.08, 0.11))
                    occupied_centers = np.vstack([occupied_centers, pt])
                    break

        for _ in range(num_extra_boxes):
            for _ in range(50):
                pt = rng.uniform(bounds[0], bounds[1], size=2)
                if is_safe_location(pt, occupied_centers, min_dist=0.3):
                    extra_box_centers.append(pt)
                    extra_box_sizes.append(rng.uniform(0.15, 0.18, size=2))
                    occupied_centers = np.vstack([occupied_centers, pt])
                    break

        # Build fixed and extra object fields.
        obj_fixed_list = []
        if len(kept_centers) > 0:
            obj_fixed_list.append(
                MultiSphereField(kept_centers, kept_radii, tensor_args=tensor_args)
            )
            
        obj_extra_list = []
        if len(extra_sphere_centers) > 0:
            obj_extra_list.append(
                MultiSphereField(np.array(extra_sphere_centers), np.array(extra_sphere_radii), tensor_args=tensor_args)
            )
        if len(extra_box_centers) > 0:
            obj_extra_list.append(
                MultiBoxField(np.array(extra_box_centers), np.array(extra_box_sizes), tensor_args=tensor_args)
            )
            
        # Aggregate active obstacle metadata for external planners.
        active_centers = []
        active_types = []
        active_dims = []
        
        sphere_centers = []
        sphere_radii = []
        if len(kept_centers) > 0:
            sphere_centers.extend(kept_centers)
            sphere_radii.extend(kept_radii)
        if len(extra_sphere_centers) > 0:
            sphere_centers.extend(extra_sphere_centers)
            sphere_radii.extend(extra_sphere_radii)
            
        if len(sphere_centers) > 0:
            active_centers.extend(sphere_centers)
            active_types.extend(['sphere'] * len(sphere_centers))
            active_dims.extend([[r] for r in sphere_radii])

        if len(extra_box_centers) > 0:
            active_centers.extend(extra_box_centers)
            active_types.extend(['box'] * len(extra_box_centers))
            half_sizes = np.array(extra_box_sizes) / 2.0
            active_dims.extend(half_sizes.tolist())

        self.active_obs_centers = np.array(active_centers, dtype=np.float32) if active_centers else np.empty((0, 2))
        self.active_obs_types = np.array(active_types) if active_types else np.empty((0,), dtype=object)
        
        active_dims_np = np.empty((len(active_dims),), dtype=object)
        for i, d in enumerate(active_dims):
            active_dims_np[i] = np.array(d, dtype=np.float32)
        self.active_obs_dims = active_dims_np

        # Initialize EnvBase to build collision fields / SDF.
        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_fixed_list, 'dynamic-base')] if obj_fixed_list else [],
            obj_extra_list=[ObjectField(obj_extra_list, 'dynamic-extra')] if obj_extra_list else [],
            tensor_args=tensor_args,
            **kwargs
        )
        self.history_trajectory = None
