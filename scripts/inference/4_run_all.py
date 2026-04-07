import os
import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import heapq
import math
# 导入底层依赖与 MPD 模型
from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length

# 导入经典 Baseline 规划器
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.chomp import CHOMP


from tmpd_baselines.environment.env_dense_2d_extra_objects import EnvDense2DExtraObjects

from mpd.utils.topology_utils import (
    get_trajectory_signature, 
    evaluate_homotopy_topological_energy, 
    prune_self_intersections,
    get_simplest_homotopy_curve,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
SEQ_PLOTS_DIR = 'all_benchmark_plots'
TANGLE_DIR = 'all_tangle_cases'

# ==========================================
# [新增] Topo-A* 依赖类与核心函数
# ==========================================
class TopoNode:
    def __init__(self, x, y, g, h, parent, W_array):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.W_array = W_array

    def __lt__(self, other):
        return self.f < other.f
        
    def get_state_key(self):
        # 对绕数量化，防止状态空间爆炸
        quantized_W = tuple(np.round(self.W_array, 1))
        return (round(self.x, 3), round(self.y, 3), quantized_W)
class RRTNode:
    def __init__(self, x, y, W_array, parent=None):
        self.x = x
        self.y = y
        self.W_array = W_array
        self.parent = parent

def calc_delta_winding_vectorized(p1, p2, obstacles):
    if len(obstacles) == 0: return np.array([])
    p1, p2 = np.array(p1), np.array(p2)
    v1 = p1 - obstacles
    v2 = p2 - obstacles
    theta1 = np.arctan2(v1[:, 1], v1[:, 0])
    theta2 = np.arctan2(v2[:, 1], v2[:, 0])
    delta_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return delta_theta / (2 * np.pi)
# ==========================================
# [新增] Topo-RRT 核心算法
# ==========================================
def lifelong_topo_rrt(start, goal, env, initial_W, step_size=0.05, W_th=0.95, robot_radius=0.03, max_iters=15000):
    obs_centers = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', [])
    obs_dims = getattr(env, 'active_obs_dims', [])
    
    def is_collision(px, py):
        if not (-1.0 <= px <= 1.0 and -1.0 <= py <= 1.0): return True
        for i, center in enumerate(obs_centers):
            ctype = obs_types[i]
            cdim = obs_dims[i]
            if ctype == 'sphere' and math.hypot(px - center[0], py - center[1]) <= cdim[0] + robot_radius:
                return True
            elif ctype == 'box' and abs(px - center[0]) <= cdim[0] + robot_radius and abs(py - center[1]) <= cdim[1] + robot_radius:
                return True
        return False

    start_node = RRTNode(start[0], start[1], initial_W)
    tree_nodes = [start_node]
    
    # 使用预分配的 numpy 数组极速加速 Nearest Neighbor 计算
    tree_coords = np.zeros((max_iters + 2, 2))
    tree_coords[0] = [start[0], start[1]]
    num_nodes = 1
    
    for _ in range(max_iters):
        # 10% Goal Bias
        if np.random.rand() < 0.1:
            rx, ry = goal[0], goal[1]
        else:
            rx, ry = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
            
        # Numpy 向量化寻找最近节点
        dists_sq = (tree_coords[:num_nodes, 0] - rx)**2 + (tree_coords[:num_nodes, 1] - ry)**2
        nearest_idx = np.argmin(dists_sq)
        nearest_node = tree_nodes[nearest_idx]
        
        # Steer 延伸
        theta = math.atan2(ry - nearest_node.y, rx - nearest_node.x)
        nx = nearest_node.x + step_size * math.cos(theta)
        ny = nearest_node.y + step_size * math.sin(theta)
        
        if is_collision(nx, ny):
            continue
            
        # 拓扑绕数增量计算
        delta_W = calc_delta_winding_vectorized((nearest_node.x, nearest_node.y), (nx, ny), obs_centers)
        new_W = nearest_node.W_array + delta_W
        
        # 拓扑硬熔断 (Veto)
        if np.any(np.abs(new_W) >= W_th):
            continue
            
        new_node = RRTNode(nx, ny, new_W, nearest_node)
        tree_nodes.append(new_node)
        tree_coords[num_nodes] = [nx, ny]
        num_nodes += 1
        
        # 终点判定
        if math.hypot(nx - goal[0], ny - goal[1]) <= step_size * 1.5:
            goal_W = new_node.W_array + calc_delta_winding_vectorized((nx, ny), goal, obs_centers)
            if np.any(np.abs(goal_W) >= W_th): continue
            
            goal_node = RRTNode(goal[0], goal[1], goal_W, new_node)
            path = []
            curr = goal_node
            while curr:
                path.append((curr.x, curr.y))
                curr = curr.parent
            return path[::-1]
            
    return None
# ==========================================
# Topo-A* 
# ==========================================
def lifelong_topo_a_star(start, goal, env, initial_W, step_size=0.05, W_th=0.95, robot_radius=0.03):
    obs_centers = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', [])
    obs_dims = getattr(env, 'active_obs_dims', [])
    
    def is_collision(px, py):
        if not (-1.0 <= px <= 1.0 and -1.0 <= py <= 1.0): return True
        for i, center in enumerate(obs_centers):
            ctype = obs_types[i]
            cdim = obs_dims[i]
            if ctype == 'sphere' and math.hypot(px - center[0], py - center[1]) <= cdim[0] + robot_radius:
                return True
            elif ctype == 'box' and abs(px - center[0]) <= cdim[0] + robot_radius and abs(py - center[1]) <= cdim[1] + robot_radius:
                return True
        return False

    start_node = TopoNode(start[0], start[1], 0, math.hypot(start[0]-goal[0], start[1]-goal[1]), None, initial_W)
    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()
    
    motions = [
        (0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0),
        (step_size, step_size), (step_size, -step_size), (-step_size, step_size), (-step_size, -step_size)
    ]
    
    # 增加安全退出机制，防止极端密集环境下 A* 内存 OOM 或卡死
    max_iters = 25000 
    iters = 0

    while open_set and iters < max_iters:
        iters += 1
        curr = heapq.heappop(open_set)
        
        if math.hypot(curr.x - goal[0], curr.y - goal[1]) <= step_size * 1.5:
            path = [(goal[0], goal[1])]
            while curr:
                path.append((curr.x, curr.y))
                curr = curr.parent
            return path[::-1]
            
        state_key = curr.get_state_key()
        if state_key in closed_set: continue
        closed_set.add(state_key)
        
        for dx, dy in motions:
            nx, ny = curr.x + dx, curr.y + dy
            if is_collision(nx, ny): continue
            
            delta_W = calc_delta_winding_vectorized((curr.x, curr.y), (nx, ny), obs_centers)
            new_W = curr.W_array + delta_W
            if np.any(np.abs(new_W) >= W_th): continue # 拓扑熔断
                
            new_g = curr.g + math.hypot(dx, dy)
            new_h = math.hypot(nx - goal[0], ny - goal[1])
            heapq.heappush(open_set, TopoNode(nx, ny, new_g, new_h, curr, new_W))
            
    return None

def plot_tangle_diagnostic(ax, env, waypoints_np, history, taut_traj, sig, title, color):
    """渲染拓扑失效诊断图"""
    env.render(ax)
    if len(history) > 0:
        full_raw = np.concatenate(history)
        ax.plot(full_raw[:, 0], full_raw[:, 1], color='gray', alpha=0.3, label='Raw Path')
    if taut_traj is not None:
        ax.plot(taut_traj[:, 0], taut_traj[:, 1], color=color, linewidth=4, label='Tangled Taut-Curve')
        obs_centers = getattr(env, 'active_obs_centers', [])
        for i, w in enumerate(sig):
            if abs(w) >= 0.95:
                ax.plot(obs_centers[i][0], obs_centers[i][1], 'rx', markersize=15, markeredgewidth=3)
    for k, wp in enumerate(waypoints_np):
        ax.plot(wp[0], wp[1], 'go', markersize=8)
        ax.text(wp[0], wp[1], str(k), weight='bold')
    ax.set_title(title, color='red', weight='bold')
    ax.legend(loc='lower right')

def generate_sequential_waypoints(env, task, q_dim, tensor_args, num_segments=5, max_attempts=1000):
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', ['sphere'] * len(obs_centers_np))
    obs_dims = getattr(env, 'active_obs_dims', [np.array([0.125])] * len(obs_centers_np))
    waypoints_np, waypoints_t = [], []
    while True:
        p0 = np.random.uniform(-0.85, 0.85, size=q_dim)
        t0 = torch.tensor(p0, **tensor_args)
        if task.compute_collision(t0.unsqueeze(0)).item() == 0:
            waypoints_np.append(p0); waypoints_t.append(t0)
            break
    for _ in range(num_segments):
        curr_p = waypoints_np[-1]
        found = False
        for _ in range(max_attempts):
            next_p = np.random.uniform(-0.85, 0.85, size=q_dim)
            next_t = torch.tensor(next_p, **tensor_args)
            if task.compute_collision(next_t.unsqueeze(0)).item() > 0: continue
            if np.linalg.norm(next_p - curr_p) < 0.6: continue
            if not is_trajectory_safe(np.array([curr_p, next_p]), obs_centers_np, obs_types, obs_dims):
                waypoints_np.append(next_p); waypoints_t.append(next_t)
                found = True
                break
        if not found: return None, None
    return waypoints_t, waypoints_np

def resample_trajectory(traj_np, n_points):
    """由于 RRT 返回的点数不固定，需要重采样以对齐 Smoothness 的计算"""
    if len(traj_np) == n_points: return traj_np
    if len(traj_np) < 2: return np.zeros((n_points, traj_np.shape[1]))
    diffs = np.linalg.norm(np.diff(traj_np, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(diffs), 0, 0)
    total_len = cum_dists[-1]
    if total_len == 0: return np.tile(traj_np[0], (n_points, 1))
    new_dists = np.linspace(0, total_len, n_points)
    resampled = np.zeros((n_points, traj_np.shape[1]))
    for i in range(traj_np.shape[1]):
        resampled[:, i] = np.interp(new_dists, cum_dists, traj_np[:, i])
    return resampled

@single_experiment_yaml
def run_all_benchmark(
    model_id: str = 'EnvDense2D-RobotPointMass', # 注意
    n_samples: int = 100,
    num_trials: int = 6,
    num_segments: int = 5,
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(SEQ_PLOTS_DIR, exist_ok=True)
    os.makedirs(TANGLE_DIR, exist_ok=True)
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # ==========================
    # 1. 基础模型与环境加载
    # ==========================
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points; dt = 5.0 / n_support_points 

    model = get_model(model_class=args['diffusion_model_class'], model=TemporalUnet(state_dim=dataset.state_dim, n_support_points=n_support_points, unet_input_dim=args['unet_input_dim'], dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']]), tensor_args=tensor_args, variance_schedule=args['variance_schedule'], n_diffusion_steps=args['n_diffusion_steps'], predict_epsilon=args['predict_epsilon'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'), map_location=device))
    model.eval(); freeze_torch_model_params(model); model.warmup(horizon=n_support_points, device=device)

    all_results = []
    methods_list = ["Topo-RRT", "Topo-A*", "MPD", "TMPD (Ours)"]

    for trial in range(num_trials):
        # 【修改点 2】：实例化 EnvDense2DExtraObjects，拔掉 6 个旧的，塞入 8 个球和 8 个方块
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=trial*100)
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', []); obs_types = getattr(dynamic_env,'active_obs_types',[]); obs_dims = getattr(dynamic_env,'active_obs_dims',[])
        
        waypoints_t, waypoints_np = generate_sequential_waypoints(dynamic_env, dynamic_task, 2, tensor_args, num_segments)
        if waypoints_t is None: continue

        print(f"\nTrial {trial+1}/{num_trials}: Navigating 4 Planners over {num_segments} segments...")

        # ==========================
        # 2. 初始化所有 Planner
        # ==========================
        
        cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) for f in dynamic_task.get_collision_fields()]
        cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)], weights_cost_l=[1e-2]*len(cost_collision_l) + [1e-7], tensor_args=tensor_args)
        guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * 1.5), tensor_args=tensor_args)

        tracker = {m: {"history": [], "curr_t": waypoints_t[0].clone(), "sr": 0, "time": 0.0, 
                       "is_tangled": False, "collision": False, "pl_list": [], "sm_list": []} for m in methods_list}

        # ==========================
        # 3. 序列推理循环
        # ==========================
        for seg in range(num_segments):
            goal_seg = waypoints_t[seg+1]

            for method in methods_list:
                state = tracker[method]
                if state["collision"]: continue 

                fix_random_seed(seed + trial * 1000 + seg) 
                t0 = time.time()
                best_traj_np = None
                
                # --- [1] Topo-RRT 推理 ---
                if method == "Topo-RRT":
                    start_np = state["curr_t"].cpu().numpy()
                    goal_np = goal_seg.cpu().numpy()
                    
                    hist_mem = np.concatenate(state["history"]) if state["history"] else np.array([])
                    
                    # 极其关键：计算历史累计绕数，作为 RRT 根节点的初始状态
                    if len(hist_mem) > 0:
                        taut_hist = get_simplest_homotopy_curve(hist_mem, obs_centers_np, obs_types, obs_dims)
                        check_hist = taut_hist if taut_hist is not None else hist_mem
                        initial_W = np.array(get_trajectory_signature(check_hist, obs_centers_np))
                    else:
                        initial_W = np.zeros(len(obs_centers_np))
                        
                    try:
                        path_list = lifelong_topo_rrt(
                            start=start_np, 
                            goal=goal_np, 
                            env=dynamic_env, 
                            initial_W=initial_W,
                            step_size=0.05, 
                            W_th=0.95, 
                            robot_radius=0.03
                        )
                        if path_list is not None and len(path_list) > 1:
                            best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
                    except Exception as e:
                        print(f"⚠️ Topo-RRT 失败: {e}")
                # --- [2] Topo-A* (Lifelong) 推理 ---
                elif method == "Topo-A*":
                    start_np = state["curr_t"].cpu().numpy()
                    goal_np = goal_seg.cpu().numpy()
                    
                    hist_mem = np.concatenate(state["history"]) if state["history"] else np.array([])
                    
                    # 极其关键：计算历史累计绕数，作为 A* 的初始状态
                    if len(hist_mem) > 0:
                        taut_hist = get_simplest_homotopy_curve(hist_mem, obs_centers_np, obs_types, obs_dims)
                        check_hist = taut_hist if taut_hist is not None else hist_mem
                        initial_W = np.array(get_trajectory_signature(check_hist, obs_centers_np))
                    else:
                        initial_W = np.zeros(len(obs_centers_np))
                        
                    try:
                        path_list = lifelong_topo_a_star(
                            start=start_np, 
                            goal=goal_np, 
                            env=dynamic_env, 
                            initial_W=initial_W,
                            step_size=0.05, 
                            W_th=0.95, 
                            robot_radius=0.03
                        )
                        if path_list is not None and len(path_list) > 1:
                            # 将 A* 离散点重采样以对齐 TMPD 的 Tensor 维度规范
                            best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
                    except Exception as e:
                        print(f"⚠️ Topo-A* 失败: {e}")
                # --- [3] MPD 推理 ---
                elif method == "MPD":
                    h_v = dataset.get_hard_conditions(torch.vstack((state["curr_t"], goal_seg)), normalize=True)
                    t_v = model.run_inference(None, h_v, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=10, t_start_guide=ceil(0.25*model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.5, n_diffusion_steps_without_noise=5)
                    _, _, free_v, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(t_v)[-1], return_indices=True)
                    if free_v is not None: best_traj_np = free_v[0, ..., :2].cpu().numpy()

                # --- [4] TMPD (Ours) 推理 ---
                elif method == "TMPD (Ours)":
                    h_t = dataset.get_hard_conditions(torch.vstack((state["curr_t"], goal_seg)), normalize=True)
                    hist_mem = np.concatenate(state["history"]) if state["history"] else np.array([])
                    if len(hist_mem) > 0:
                        refined = get_simplest_homotopy_curve(hist_mem, obs_centers_np, obs_types, obs_dims)
                        if refined is not None: hist_mem = refined
                    t_t = model.run_inference(None, h_t, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=10, t_start_guide=ceil(0.25*model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 1.8, n_diffusion_steps_without_noise=20)
                    _, _, free_t, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(t_t)[-1], return_indices=True)
                    if free_t is not None:
                        t_np = free_t[..., :2].cpu().numpy(); u_classes = [prune_self_intersections(traj) for traj in t_np]
                        energies, _ = evaluate_homotopy_topological_energy(hist_mem, u_classes, obs_centers_np)
                        scores = []
                        for traj, e in zip(u_classes, energies):
                            # ========================================================
                            # 全局缠绕一票否决 (Hard Constraint for Tangling)
                            # ========================================================
                            if len(hist_mem) > 0:
                                combined_raw = np.vstack((hist_mem, traj[1:]))
                            else:
                                combined_raw = traj
                                
                            refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                            check_traj = refined_combined if refined_combined is not None else combined_raw
                            
                            global_sig = get_trajectory_signature(check_traj, obs_centers_np)
                            
                            # 一旦发现全局打结，赋予 10000 分的极高惩罚
                            if np.any(np.abs(global_sig) >= 1):
                                tangle_penalty = 10000.0
                            else:
                                tangle_penalty = 0.0
                            t_tensor = torch.tensor(traj, **tensor_args).unsqueeze(0)
                            pl = compute_path_length(t_tensor, robot).item()
                            sm = compute_smoothness(t_tensor, robot).item()
                            
                            # 综合打分：拓扑势能 + 长度 + 平滑度 + 缠绕熔断惩罚
                            scores.append(e + 0.8 * pl + 1.0 * sm + tangle_penalty)
                            
                        best_traj_np = u_classes[np.argmin(scores)]

                state["time"] += (time.time() - t0)

                # --- 状态更新与拓扑安全判定 ---
                if best_traj_np is not None:
                    state["history"].append(best_traj_np)
                    state["curr_t"] = torch.tensor(best_traj_np[-1], **tensor_args)
                    state["sr"] += 1
                    state["pl_list"].append(np.sum(np.linalg.norm(np.diff(best_traj_np, axis=0), axis=1)))
                    state["sm_list"].append(np.mean(np.square(np.diff(best_traj_np, n=2, axis=0))))
                    
                    full_hist = np.concatenate(state["history"])
                    taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)
                    if taut_traj is not None:
                        sig = get_trajectory_signature(taut_traj, obs_centers_np)
                        if np.any(np.abs(sig) >= 0.95):
                            if not state["is_tangled"]: 
                                fig, ax = plt.subplots()
                                plot_tangle_diagnostic(ax, dynamic_env, waypoints_np, state["history"], taut_traj, sig, f"{method} Tangle (Seg {seg})", 'red' if 'TMPD' in method else 'blue')
                                plt.savefig(os.path.join(TANGLE_DIR, f"{method.replace(' ','_')}_tangle_trial{trial+1}_seg{seg}.png")); plt.close()
                            state["is_tangled"] = True
                else:
                    state["collision"] = True

        # ==========================
        # 4. 绘制 2x2 全局对比图
        # ==========================
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Continuous Lifelong Navigation Benchmark - Trial {trial+1}", fontsize=22, weight='bold')
        axes = axes.flatten()

        for i, method in enumerate(methods_list):
            ax = axes[i]
            state = tracker[method]
            dynamic_env.render(ax)
            ax.set_title(method, fontsize=16, weight='bold', color='darkred' if 'TMPD' in method else 'black')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)

            for k, wp in enumerate(waypoints_np):
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

            valid_segs = len(state["history"])
            line_color = 'darkred' if 'TMPD' in method else 'blue'
            for seg_idx, traj in enumerate(state["history"]):
                ax.plot(traj[:, 0], traj[:, 1], color=line_color, linewidth=2.5, alpha=0.8)
                mid = len(traj) // 2
                ax.annotate('', xy=(traj[mid+1, 0], traj[mid+1, 1]), xytext=(traj[mid, 0], traj[mid, 1]), arrowprops=dict(arrowstyle="->", color=line_color, lw=2))
            # ========================================================
            # 2. 【核心新增】计算并绘制全局拓扑同伦轨迹 (Taut Curve)
            # ========================================================
            if valid_segs > 0:
                full_hist = np.concatenate(state["history"])
                taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)
                
                if taut_traj is not None:
                    # 使用粗体、亮橙色、虚线 来表示拓扑等价的收紧曲线
                    ax.plot(taut_traj[:, 0], taut_traj[:, 1], color='darkorange', linewidth=3.5, linestyle='--', alpha=0.9, zorder=15, label='Topological Taut-Curve')
                    
                    # 如果打结了，在图上额外标出勒住的那个障碍物
                    if state["is_tangled"]:
                        sig = get_trajectory_signature(taut_traj, obs_centers_np)
                        for obs_idx, w in enumerate(sig):
                            if abs(w) >= 0.95:
                                ax.plot(obs_centers_np[obs_idx][0], obs_centers_np[obs_idx][1], 'rx', markersize=18, markeredgewidth=3, zorder=25)
            # 绘制图例 (Legend)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='lower right', fontsize=11, framealpha=0.9)                    
            if state["collision"]: status_text, box_color = "Collision (Failed)", 'lightgray'
            elif state["is_tangled"]: status_text, box_color = "Tangled (Failed)", 'lightcoral'
            else: status_text, box_color = "Success", 'lightgreen'
            
            props = dict(boxstyle='round', facecolor=box_color, alpha=0.9)
            ax.text(0.05, 0.95, f"Status: {status_text}\nSegments: {valid_segs}/{num_segments}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, weight='bold', zorder=30)

        plt.tight_layout()
        plt.savefig(os.path.join(SEQ_PLOTS_DIR, f"benchmark_trial_{trial:03d}.png"), dpi=200); plt.close(fig)

        # ==========================
        # 5. 录入指标
        # ==========================
        for method in methods_list:
            st = tracker[method]
            all_results.append({
                "Method": method, "Trial": trial,
                "Success_Rate": st["sr"] / num_segments, 
                "Tangle_Free_Rate": 1.0 if (not st["is_tangled"] and st["sr"] == num_segments) else 0.0,
                "Path_Length": np.mean(st["pl_list"]) if st["pl_list"] else np.nan,
                "Smoothness": np.mean(st["sm_list"]) if st["sm_list"] else np.nan,
                "Time": st["time"]
            })

    # ==========================
    # 6. 统计与打印表格
    # ==========================
    df = pd.DataFrame(all_results)
    if len(all_results) == 0:
        print("没有收集到任何数据，请检查上面的报错信息！")
        return
        
    df = pd.DataFrame(all_results)
    df['Method'] = pd.Categorical(df['Method'], categories=methods_list, ordered=True)
    summary = df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.nanmean(x)*100:.1f}%" if np.sum(~np.isnan(x)) > 0 else "0.0%",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x)*1000:.3f} ± {np.nanstd(x)*1000:.3f}",
        "Time": lambda x: f"{np.nanmean(x):.2f}s"
    })
    
    print("\n" + "="*95)
    print("BENCHMARK REPORT (Dense)")
    print("="*95)
    print(summary.to_string())
    print("="*95)
    print("\n* Note: Smoothness values are multiplied by 10^3 for better precision display.")

if __name__ == '__main__':
    run_experiment(run_all_benchmark)