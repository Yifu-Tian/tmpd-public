import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvSimple2DExtraObjects(EnvBase):
    """
    终极动态泛化测试环境：
    1. 随机删减旧障碍物 (测试模型是否能发现新敞开的捷径)
    2. 随机添加新障碍物 (测试模型是否能避开新陷阱)
    """
    def __init__(self, 
                 name='EnvSimple2DExtraObjects', 
                 tensor_args=None, 
                 drop_old_num=10,        # 每次随机删掉 5 个旧障碍物
                 num_extra_spheres=5,   # 每次随机新增 4 个新圆
                 num_extra_boxes=7,     # 每次随机新增 2 个方块
                 seed=36,  # None/固定场景用于测试
                 **kwargs):
        
        rng = np.random.default_rng(seed)
        
        # 1. 基础障碍物池 (直接把原 EnvSimple2D 的 15 个圆抄过来)
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
        
        # 2. 随机删除旧障碍物 (Obstacle Dropout)
        total_base = len(all_base_centers)
        keep_num = max(0, total_base - drop_old_num)
        
        # 随机抽取保留下来的索引
        indices = np.arange(total_base)
        rng.shuffle(indices)
        keep_indices = indices[:keep_num]
        drop_indices = indices[keep_num:]

        kept_centers = all_base_centers[keep_indices]
        kept_radii = all_base_radii[keep_indices]
        
        # 被删除的旧障碍物，仅作可视化，不参与物理碰撞
        self.dropped_centers = all_base_centers[drop_indices]
        self.dropped_radii = all_base_radii[drop_indices]
        # 3. 拒绝采样：添加新障碍物
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

        # ==========================================
        # 4. 组装 fixed (默认颜色) 和 extra (红色) 物理场
        # ==========================================
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
        # ==========================================
        # 汇总所有真实存在的实体障碍物中心，供外部拓扑算子调用
        # ==========================================
        active_centers = []
        active_types = []  # 用字符串标记类型: 'sphere' 或 'box'
        active_dims = []
        
        # 1. 处理圆形障碍物 (kept_centers 和 extra_sphere_centers)
        # 合并这两个列表
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
            # 标记为圆形
            active_types.extend(['sphere'] * len(sphere_centers))
            # 存储半径 (为了格式统一，存为列表)
            active_dims.extend([[r] for r in sphere_radii])

        # 2. 处理方形障碍物 (extra_box_centers)
        if len(extra_box_centers) > 0:
            active_centers.extend(extra_box_centers)
            # 标记为方形
            active_types.extend(['box'] * len(extra_box_centers))
            # 存储半宽和半高 (BoxField 通常用半尺寸，我们将全尺寸 sizes 除以 2)
            # extra_box_sizes 是 (N, 2) 的全尺寸数组
            half_sizes = np.array(extra_box_sizes) / 2.0
            active_dims.extend(half_sizes.tolist())

        # 3. 转换为 numpy 数组并保存为类属性
        self.active_obs_centers = np.array(active_centers, dtype=np.float32) if active_centers else np.empty((0, 2))
        self.active_obs_types = np.array(active_types) if active_types else np.empty((0,), dtype=object)
        # dims 可能长度不一，使用 object 类型存储
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
        # ==========================================
        # 5. 重写渲染函数，画出被删除的障碍物虚线
        # ==========================================
    def render(self, ax, **kwargs):
        # 先让底层的 EnvBase 画出存在的障碍物 (黑色固定 + 红色新增)
        super().render(ax, **kwargs)
        
        # 手动将那些被 Drop 掉的旧圆画成半透明灰色虚线框
        for center, radius in zip(self.dropped_centers, self.dropped_radii):
            circle = patches.Circle(
                (center[0], center[1]), 
                radius, 
                linewidth=1.5,           # 线宽
                edgecolor='gray',        # 边缘颜色为灰色
                facecolor='none',        # 内部透明，不填充
                linestyle='--',          # 虚线样式
                alpha=0.6,               # 降低透明度，避免抢占视觉焦点
                zorder=3                 # 确保绘制层级在上方
            )
            ax.add_patch(circle)

if __name__ == '__main__':
    env = EnvSimple2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS,
        drop_old_num=10,       # 删掉旧的
        num_extra_spheres=5,  # 加新圆
        num_extra_boxes=5     #加新方块
    )
    
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.title("Dynamic Env: Dropped (Dashed) & Extra (Red)")
    plt.show()