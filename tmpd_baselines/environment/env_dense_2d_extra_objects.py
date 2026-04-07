import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_dense_2d import EnvDense2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvDense2DExtraObjects(EnvBase):
    """
    终极动态泛化测试环境 (基于 Dense2D 密集丛林)：
    1. 动态提取原 EnvDense2D 的所有基础密集障碍物。
    2. 随机删减旧障碍物 (测试模型是否能发现新敞开的捷径)，并用虚线渲染。
    3. 拒绝采样添加新球体与新方块 (测试模型是否能避开新陷阱)。
    """
    def __init__(self, 
                 name='EnvDense2DExtraObjects', 
                 tensor_args=None, 
                 drop_old_num=6,        # 每次随机删掉旧障碍物的数量
                 num_extra_spheres=5,   # 每次随机新增新圆的数量
                 num_extra_boxes=8,     # 每次随机新增方块的数量
                 seed=42, 
                 **kwargs):
        
        rng = np.random.default_rng(seed)
        
        # =================================================================
        # 1. 偷天换日：创建一个 Dummy 的 EnvDense2D，把它的固定障碍物坐标全偷出来！
        # =================================================================
        dummy_kwargs = kwargs.copy()
        dummy_kwargs['precompute_sdf_obj_fixed'] = False  # 关掉预计算，省时间
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
        
        # =================================================================
        # 2. 随机删除旧障碍物 (Obstacle Dropout)
        # =================================================================
        total_base = len(all_base_centers)
        keep_num = max(0, total_base - drop_old_num)
        
        # 随机抽取保留下来的索引
        indices = np.arange(total_base)
        rng.shuffle(indices)
        keep_indices = indices[:keep_num]
        drop_indices = indices[keep_num:]

        kept_centers = all_base_centers[keep_indices] if total_base > 0 else np.empty((0, 2))
        kept_radii = all_base_radii[keep_indices] if total_base > 0 else np.empty((0,))
        
        # 被删除的旧障碍物，仅作可视化虚线，不参与物理碰撞
        self.dropped_centers = all_base_centers[drop_indices] if total_base > 0 else np.empty((0, 2))
        self.dropped_radii = all_base_radii[drop_indices] if total_base > 0 else np.empty((0,))
        
        # =================================================================
        # 3. 拒绝采样：在空位安全地添加新障碍物
        # =================================================================
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

        # =================================================================
        # 4. 组装 fixed (默认颜色) 和 extra (红色) 物理场
        # =================================================================
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
            
        # =================================================================
        # 5. 汇总所有真实存在的实体障碍物中心，供外部拓扑算子调用
        # =================================================================
        active_centers = []
        active_types = []  # 'sphere' 或 'box'
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
            # BoxField 通常外部算子用的是半尺寸
            half_sizes = np.array(extra_box_sizes) / 2.0
            active_dims.extend(half_sizes.tolist())

        self.active_obs_centers = np.array(active_centers, dtype=np.float32) if active_centers else np.empty((0, 2))
        self.active_obs_types = np.array(active_types) if active_types else np.empty((0,), dtype=object)
        
        active_dims_np = np.empty((len(active_dims),), dtype=object)
        for i, d in enumerate(active_dims):
            active_dims_np[i] = np.array(d, dtype=np.float32)
        self.active_obs_dims = active_dims_np

        # =================================================================
        # 6. 交给底层的 EnvBase 进行标准初始化 (自动算好完美的 SDF)
        # =================================================================
        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_fixed_list, 'dynamic-base')] if obj_fixed_list else [],
            obj_extra_list=[ObjectField(obj_extra_list, 'dynamic-extra')] if obj_extra_list else [],
            tensor_args=tensor_args,
            **kwargs
        )
        self.history_trajectory = None

    # =================================================================
    # 7. 重写渲染函数，画出被删除的障碍物虚线
    # =================================================================
    # def render(self, ax, **kwargs):
    #     # 先让底层的 EnvBase 画出存在的障碍物 (黑色固定 + 红色新增)
    #     super().render(ax, **kwargs)
        
    #     # 手动将那些被 Drop 掉的旧圆画成半透明灰色虚线框
    #     for center, radius in zip(self.dropped_centers, self.dropped_radii):
    #         circle = patches.Circle(
    #             (center[0], center[1]), 
    #             radius, 
    #             linewidth=1.5,           # 线宽
    #             edgecolor='gray',        # 边缘颜色为灰色
    #             facecolor='none',        # 内部透明，不填充
    #             linestyle='--',          # 虚线样式
    #             alpha=0.6,               # 降低透明度，避免抢占视觉焦点
    #             zorder=3                 # 确保绘制层级在上方
    #         )
    #         ax.add_patch(circle)

if __name__ == '__main__':
    env = EnvDense2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS,
        drop_old_num=6,       # 删掉旧的
        num_extra_spheres=8,  # 加新圆
        num_extra_boxes=8     # 加新方块
    )
    
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.title("Dense Dynamic Env: Dropped (Dashed) & Extra (Red)")
    plt.show()