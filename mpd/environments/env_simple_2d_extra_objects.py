import numpy as np
import torch
import matplotlib.patches as patches
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class EnvSimple2DExtraObjects(EnvBase):
    """
    Dynamic simple 2D environment with obstacle dropout and obstacle injection.
    """
    def __init__(self, 
                 name='EnvSimple2DExtraObjects', 
                 tensor_args=None, 
                 drop_old_num=10,
                 num_extra_spheres=5,
                 num_extra_boxes=7,
                 seed=36,
                 **kwargs):
        if tensor_args is None:
            tensor_args = DEFAULT_TENSOR_ARGS

        rng = np.random.default_rng(seed)
        
        # Base obstacle pool from the original Simple2D setting.
        all_base_centers = np.array([
            [-0.43378472, 0.33346438], [0.33134747, 0.6288051 ],
            [-0.56569648, -0.4849945 ], [0.42124248, -0.66561657],
            [0.05636655, -0.51496643], [-0.36961785, -0.12315541],
            [-0.87402171, -0.40349367], [-0.63592142, 0.66831249],
            [0.80878216, 0.52878702], [-0.02378611, 0.45900694],
            [0.1455742 , 0.16420497], [0.62841374, -0.43461448],
            [0.17965621, -0.89262766], [0.67759687, 0.8817358 ],
            [-0.36087668, 0.83134586]
        ])
        all_base_radii = np.array([0.125] * 15)
        
        # Randomly drop a subset of base obstacles.
        total_base = len(all_base_centers)
        keep_num = max(0, total_base - drop_old_num)
        
        indices = np.arange(total_base)
        rng.shuffle(indices)
        keep_indices = indices[:keep_num]
        drop_indices = indices[keep_num:]

        kept_centers = all_base_centers[keep_indices]
        kept_radii = all_base_radii[keep_indices]
        
        # Removed obstacles are kept for visualization only.
        self.dropped_centers = all_base_centers[drop_indices]
        self.dropped_radii = all_base_radii[drop_indices]

        # Rejection-sample extra obstacles in free space.
        bounds = [-0.9, 0.9]
        extra_sphere_centers, extra_sphere_radii = [], []
        extra_box_centers, extra_box_sizes = [], []
        
        occupied_centers = kept_centers.copy() if keep_num > 0 else np.empty((0, 2))

        def is_safe_location(pt, existing_pts, min_dist=0.25):
            if len(existing_pts) == 0: return True
            dists = np.linalg.norm(existing_pts - pt, axis=1)
            return np.all(dists > min_dist)

        for _ in range(num_extra_spheres):
            for _ in range(50):
                pt = rng.uniform(bounds[0], bounds[1], size=2)
                if is_safe_location(pt, occupied_centers):
                    extra_sphere_centers.append(pt)
                    extra_sphere_radii.append(rng.uniform(0.1, 0.12))
                    occupied_centers = np.vstack([occupied_centers, pt])
                    break

        for _ in range(num_extra_boxes):
            for _ in range(50):
                pt = rng.uniform(bounds[0], bounds[1], size=2)
                if is_safe_location(pt, occupied_centers, min_dist=0.3):
                    extra_box_centers.append(pt)
                    extra_box_sizes.append(rng.uniform(0.2, 0.25, size=2))
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
        
        # Merge sphere obstacles from kept + extra sets.
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

        # Add box obstacles (store half sizes for consistency).
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

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_fixed_list, 'dynamic-base')] if obj_fixed_list else [],
            obj_extra_list=[ObjectField(obj_extra_list, 'dynamic-extra')] if obj_extra_list else [],
            tensor_args=tensor_args,
            **kwargs
        )
        self.history_trajectory = None

    def render(self, ax, **kwargs):
        # Draw active obstacles first.
        super().render(ax, **kwargs)
        
        # Overlay dropped base obstacles as dashed circles.
        for center, radius in zip(self.dropped_centers, self.dropped_radii):
            circle = patches.Circle(
                (center[0], center[1]), 
                radius, 
                linewidth=1.5,
                edgecolor='gray',
                facecolor='none',
                linestyle='--',
                alpha=0.6,
                zorder=3
            )
            ax.add_patch(circle)
