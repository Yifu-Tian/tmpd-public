import os
import time
import numpy as np
import pandas as pd
import torch
import math
from math import ceil
import heapq
import matplotlib.pyplot as plt

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

# 导入动态密林环境
from tmpd_baselines.environment.env_dense_2d_extra_objects import EnvDense2DExtraObjects

from mpd.utils.topology_utils import (
    get_trajectory_signature,
    prune_self_intersections,
    get_simplest_homotopy_curve,
    evaluate_homotopy_topological_energy,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
RESULTS_DIR = 'benchmark_time_to_success_all'

# ==========================================
# 共享组件：拓扑节点与绕数计算
# ==========================================
class TopoNode:
    def __init__(self, x, y, g, h, parent, W_array):
        self.x = x; self.y = y; self.g = g; self.h = h; self.f = g + h
        self.parent = parent; self.W_array = W_array
    def __lt__(self, other): return self.f < other.f
    def get_state_key(self): return (round(self.x, 3), round(self.y, 3), tuple(np.round(self.W_array, 1)))

class RRTNode:
    def __init__(self, x, y, W_array, parent=None):
        self.x = x; self.y = y; self.W_array = W_array; self.parent = parent

def calc_delta_winding_vectorized(p1, p2, obstacles):
    if len(obstacles) == 0: return np.array([])
    p1, p2 = np.array(p1), np.array(p2)
    v1, v2 = p1 - obstacles, p2 - obstacles
    theta1, theta2 = np.arctan2(v1[:, 1], v1[:, 0]), np.arctan2(v2[:, 1], v2[:, 0])
    delta_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return delta_theta / (2 * np.pi)

# ==========================================
# 算法核心函数
# ==========================================
def lifelong_topo_rrt(start, goal, env, initial_W, step_size=0.05, W_th=0.95, robot_radius=0.01, max_iters=50000):
    obs_centers = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', [])
    obs_dims = getattr(env, 'active_obs_dims', [])
    
    def is_collision(px, py):
        if not (-1.0 <= px <= 1.0 and -1.0 <= py <= 1.0): return True
        for i, center in enumerate(obs_centers):
            ctype, cdim = obs_types[i], obs_dims[i]
            if ctype == 'sphere' and math.hypot(px - center[0], py - center[1]) <= cdim[0] + robot_radius: return True
            elif ctype == 'box' and abs(px - center[0]) <= cdim[0] + robot_radius and abs(py - center[1]) <= cdim[1] + robot_radius: return True
        return False

    start_node = RRTNode(start[0], start[1], initial_W)
    tree_nodes = [start_node]
    tree_coords = np.zeros((max_iters + 2, 2))
    tree_coords[0] = [start[0], start[1]]
    num_nodes = 1
    
    for _ in range(max_iters):
        if np.random.rand() < 0.1: rx, ry = goal[0], goal[1]
        else: rx, ry = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)
            
        dists_sq = (tree_coords[:num_nodes, 0] - rx)**2 + (tree_coords[:num_nodes, 1] - ry)**2
        nearest_node = tree_nodes[np.argmin(dists_sq)]
        
        theta = math.atan2(ry - nearest_node.y, rx - nearest_node.x)
        nx, ny = nearest_node.x + step_size * math.cos(theta), nearest_node.y + step_size * math.sin(theta)
        
        if is_collision(nx, ny): continue
            
        new_W = nearest_node.W_array + calc_delta_winding_vectorized((nearest_node.x, nearest_node.y), (nx, ny), obs_centers)
        if np.any(np.abs(new_W) >= W_th): continue
            
        new_node = RRTNode(nx, ny, new_W, nearest_node)
        tree_nodes.append(new_node); tree_coords[num_nodes] = [nx, ny]; num_nodes += 1
        
        if math.hypot(nx - goal[0], ny - goal[1]) <= step_size * 1.5:
            goal_W = new_node.W_array + calc_delta_winding_vectorized((nx, ny), goal, obs_centers)
            if np.any(np.abs(goal_W) >= W_th): continue
            path, curr = [], RRTNode(goal[0], goal[1], goal_W, new_node)
            while curr: path.append((curr.x, curr.y)); curr = curr.parent
            return path[::-1]
    return None

def lifelong_topo_a_star(start, goal, env, initial_W, step_size=0.05, W_th=0.95, robot_radius=0.03):
    obs_centers = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', [])
    obs_dims = getattr(env, 'active_obs_dims', [])
    
    def is_collision(px, py):
        if not (-1.0 <= px <= 1.0 and -1.0 <= py <= 1.0): return True
        for i, center in enumerate(obs_centers):
            ctype, cdim = obs_types[i], obs_dims[i]
            if ctype == 'sphere' and math.hypot(px - center[0], py - center[1]) <= cdim[0] + robot_radius: return True
            elif ctype == 'box' and abs(px - center[0]) <= cdim[0] + robot_radius and abs(py - center[1]) <= cdim[1] + robot_radius: return True
        return False

    start_node = TopoNode(start[0], start[1], 0, math.hypot(start[0]-goal[0], start[1]-goal[1]), None, initial_W)
    open_set, closed_set = [], set()
    heapq.heappush(open_set, start_node)
    
    motions = [(0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0), (step_size, step_size), (step_size, -step_size), (-step_size, step_size), (-step_size, -step_size)]
    
    # 无限迭代直到找到终点或穷尽状态空间
    while open_set:
        curr = heapq.heappop(open_set)
        if math.hypot(curr.x - goal[0], curr.y - goal[1]) <= step_size * 1.5:
            path = [(goal[0], goal[1])]
            while curr: path.append((curr.x, curr.y)); curr = curr.parent
            return path[::-1]
            
        state_key = curr.get_state_key()
        if state_key in closed_set: continue
        closed_set.add(state_key)
        
        for dx, dy in motions:
            nx, ny = curr.x + dx, curr.y + dy
            if is_collision(nx, ny): continue
            new_W = curr.W_array + calc_delta_winding_vectorized((curr.x, curr.y), (nx, ny), obs_centers)
            if np.any(np.abs(new_W) >= W_th): continue
            heapq.heappush(open_set, TopoNode(nx, ny, curr.g + math.hypot(dx, dy), math.hypot(nx - goal[0], ny - goal[1]), curr, new_W))
    return None

def resample_trajectory(traj_np, n_points):
    if len(traj_np) == n_points: return traj_np
    if len(traj_np) < 2: return np.zeros((n_points, traj_np.shape[1]))
    diffs = np.linalg.norm(np.diff(traj_np, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(diffs), 0, 0)
    if cum_dists[-1] == 0: return np.tile(traj_np[0], (n_points, 1))
    resampled = np.zeros((n_points, traj_np.shape[1]))
    for i in range(traj_np.shape[1]): resampled[:, i] = np.interp(np.linspace(0, cum_dists[-1], n_points), cum_dists, traj_np[:, i])
    return resampled

def generate_sequential_waypoints(env, task, q_dim, tensor_args, num_segments=5, max_attempts=1000):
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', ['sphere'] * len(obs_centers_np))
    obs_dims = getattr(env, 'active_obs_dims', [np.array([0.125])] * len(obs_centers_np))
    waypoints_np, waypoints_t = [], []
    while True:
        p0 = np.random.uniform(-0.85, 0.85, size=q_dim)
        t0 = torch.tensor(p0, **tensor_args)
        if task.compute_collision(t0.unsqueeze(0)).item() == 0:
            waypoints_np.append(p0); waypoints_t.append(t0); break
    for _ in range(num_segments):
        curr_p = waypoints_np[-1]
        found = False
        for _ in range(max_attempts):
            next_p = np.random.uniform(-0.85, 0.85, size=q_dim)
            next_t = torch.tensor(next_p, **tensor_args)
            if task.compute_collision(next_t.unsqueeze(0)).item() > 0 or np.linalg.norm(next_p - curr_p) < 0.6: continue
            if not is_trajectory_safe(np.array([curr_p, next_p]), obs_centers_np, obs_types, obs_dims):
                waypoints_np.append(next_p); waypoints_t.append(next_t); found = True; break
        if not found: return None, None
    return waypoints_t, waypoints_np

# ==========================================
# 评测主流程
# ==========================================
@single_experiment_yaml
def run_benchmark_all(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    num_segments: int = 5,
    n_samples: int = 70,
    n_diffusion_steps_without_noise: int = 10, 
    start_guide_steps_fraction: float = 0.2,
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'🚀 Initializing "Time-to-Success" Benchmark (All 4 Planners)...')
    
    # 借助数据集加载环境和 Task
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points; dt = 5.0 / n_support_points 

    # 加载 MPD 模型
    model = get_model(model_class=args['diffusion_model_class'], model=TemporalUnet(state_dim=dataset.state_dim, n_support_points=n_support_points, unet_input_dim=args['unet_input_dim'], dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']]), tensor_args=tensor_args, variance_schedule=args['variance_schedule'], n_diffusion_steps=args['n_diffusion_steps'], predict_epsilon=args['predict_epsilon'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'), map_location=device))
    model.eval(); freeze_torch_model_params(model); model.warmup(horizon=n_support_points, device=device)

    all_results = []
    methods_list = ["Topo-RRT", "Topo-A*", "MPD", "TMPD (Ours)"]

    for trial in range(num_trials):
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42)
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', [])
        obs_types = getattr(dynamic_env, 'active_obs_types', [])
        obs_dims = getattr(dynamic_env, 'active_obs_dims', [])
        
        waypoints_t, waypoints_np = generate_sequential_waypoints(dynamic_env, dynamic_task, 2, tensor_args, num_segments)
        if waypoints_t is None: continue

        print(f"\n================ [ Trial {trial+1}/{num_trials} ] ================")
        
        cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) for f in dynamic_task.get_collision_fields()]
        cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)], weights_cost_l=[1e-2]*len(cost_collision_l) + [1e-7], tensor_args=tensor_args)
        guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * 1.5), tensor_args=tensor_args)

        tracker = {m: {"history": [], "sr": 0, "time": 0.0, "failed_traj": None, "fatal_error": False, "tangled": False, "pl_list": [], "sm_list": []} for m in methods_list}

        for seg in range(num_segments):
            start_t, goal_t = waypoints_t[seg], waypoints_t[seg+1]
            start_np, goal_np = start_t.cpu().numpy(), goal_t.cpu().numpy()

            for method in methods_list:
                state = tracker[method]
                if state["fatal_error"]: continue
                
                t0 = time.time()
                best_traj_np = None
                
                hist_for_eval = np.concatenate(state["history"]) if state["history"] else np.array([])
                
                initial_W = np.zeros(len(obs_centers_np))
                if len(hist_for_eval) > 0:
                    refined_hist = get_simplest_homotopy_curve(hist_for_eval, obs_centers_np, obs_types, obs_dims)
                    if refined_hist is not None: 
                        hist_for_eval = refined_hist
                        initial_W = np.array(get_trajectory_signature(refined_hist, obs_centers_np))

                # =======================================================
                # 死磕到底逻辑 (Infinite Retry)
                # =======================================================
                if method == "Topo-RRT":
                    attempts = 0
                    max_rrt_attempts = 20  # 设置 RRT 的最大重试次数（防死锁）
                    while best_traj_np is None and attempts < max_rrt_attempts:
                        attempts += 1
                        path_list = lifelong_topo_rrt(start_np, goal_np, dynamic_env, initial_W, step_size=0.05, robot_radius=0.01, max_iters=50000)
                        if path_list: 
                            best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
                        else: 
                            print(f"    [Topo-RRT] Segment {seg+1}: Tree maxed out. Restarting (Attempt {attempts}/{max_rrt_attempts})...")
                    
                    # 如果达到上限依然找不到，直接宣告拓扑死局
                    if best_traj_np is None:
                        print(f"    [Topo-RRT] ⚠️ FATAL: Trapped in topological deadlock after {max_rrt_attempts} attempts.")
                        state["fatal_error"] = True
                elif method == "Topo-A*":
                    # A* 跑遍全图如果没找到，物理上必定是死胡同
                    path_list = lifelong_topo_a_star(start_np, goal_np, dynamic_env, initial_W, step_size=0.05, robot_radius=0.03)
                    if path_list: best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
                    else: 
                        print(f"    [Topo-A*] ⚠️ FATAL: Physically trapped in dead end at Segment {seg+1}!")
                        state["fatal_error"] = True

                elif method == "MPD":
                    attempts = 0
                    while best_traj_np is None:
                        attempts += 1
                        h_cond = dataset.get_hard_conditions(torch.vstack((start_t, goal_t)), normalize=True)
                        t_samples = model.run_inference(None, h_cond, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=5, t_start_guide=ceil(0.25 * model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.5, n_diffusion_steps_without_noise=5)
                        
                        t_unnorm = dataset.unnormalize_trajectories(t_samples)[-1]
                        _, _, free_t, _, _ = dynamic_task.get_trajs_collision_and_free(t_unnorm, return_indices=True)
                        
                        if free_t is not None: 
                            best_traj_np = free_t[0, ..., :2].cpu().numpy()
                        else:
                            state["failed_traj"] = t_unnorm[0, ..., :2].cpu().numpy()
                            print(f"    [Vanilla MPD] Segment {seg+1}: All samples collided. Resampling (Attempt {attempts})...")

                elif method == "TMPD (Ours)":
                    attempts = 0
                    max_tmpd_attempts = 50 # 设置一个合理的防死锁上限
                    current_noise_std = 0.5 # 初始噪声
                    
                    while best_traj_np is None and attempts < max_tmpd_attempts:
                        attempts += 1
                        
                        # 【核心解困机制】：如果卡住了，逐渐增加噪声，逼迫模型探索大范围迂回的“解结”路径
                        if attempts > 5:
                            current_noise_std += 0.3
                            print(f"    [TMPD] Strategy shifted: Increasing noise to {current_noise_std:.1f} for wilder exploration...")

                        h_cond = dataset.get_hard_conditions(torch.vstack((start_t, goal_t)), normalize=True)
                        # 注意这里使用动态的 current_noise_std
                        t_samples = model.run_inference(None, h_cond, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=10, t_start_guide=ceil(0.1 * model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: current_noise_std, n_diffusion_steps_without_noise=10)
                        
                        t_unnorm = dataset.unnormalize_trajectories(t_samples)[-1]
                        _, _, free_t, _, _ = dynamic_task.get_trajs_collision_and_free(t_unnorm, return_indices=True)
                        
                        if free_t is not None:
                            trajs_free_np = free_t[..., :2].cpu().numpy()
                            unique_mpd_classes, unique_sigs = [], []
                            for traj in trajs_free_np:
                                sig = get_trajectory_signature(traj, obs_centers_np)
                                if not any(np.all(np.abs(sig - ext_sig) < 0.3) for ext_sig in unique_sigs):
                                    unique_sigs.append(sig); unique_mpd_classes.append(prune_self_intersections(traj))
                            
                            mpd_energies, _ = evaluate_homotopy_topological_energy(hist_for_eval, unique_mpd_classes, obs_centers_np, w_max=0.8)
                            preliminary_scores = [e + 0.8 * np.sum(np.linalg.norm(np.diff(t, axis=0), axis=1)) for t, e in zip(unique_mpd_classes, mpd_energies)]
                            
                            for rank, idx in enumerate(np.argsort(preliminary_scores)):
                                traj = unique_mpd_classes[idx]
                                combined_raw = np.vstack((hist_for_eval, traj[1:])) if len(hist_for_eval) > 0 else traj
                                refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                                global_sig = get_trajectory_signature(refined_combined if refined_combined is not None else combined_raw, obs_centers_np)
                                
                                if not np.any(np.abs(global_sig) >= 0.95):
                                    best_traj_np = traj
                                    break
                            
                            if best_traj_np is None and len(unique_mpd_classes) > 0:
                                state["failed_traj"] = unique_mpd_classes[np.argsort(preliminary_scores)[0]]
                                print(f"    [TMPD] Segment {seg+1}: All candidates tangled. Resampling (Attempt {attempts}/{max_tmpd_attempts})...")
                        else:
                            state["failed_traj"] = t_unnorm[0, ..., :2].cpu().numpy()
                            print(f"    [TMPD] Segment {seg+1}: All samples collided. Resampling (Attempt {attempts}/{max_tmpd_attempts})...")
                    
                    # 如果达到了最大尝试次数仍然失败，宣告陷入拓扑死局
                    if best_traj_np is None:
                        print(f"    [TMPD] ⚠️ FATAL: Trapped in topological deadlock after {max_tmpd_attempts} attempts.")
                        state["fatal_error"] = True
                # =======================================================
                            
                seg_time = time.time() - t0
                state["time"] += seg_time

                if best_traj_np is not None:
                    state["history"].append(best_traj_np)
                    state["sr"] += 1
                    state["pl_list"].append(np.sum(np.linalg.norm(np.diff(best_traj_np, axis=0), axis=1)))
                    state["sm_list"].append(compute_smoothness(torch.tensor(best_traj_np, **tensor_args).unsqueeze(0), robot).item())
                    full_path = np.concatenate(state["history"])
                    taut = get_simplest_homotopy_curve(full_path, obs_centers_np, obs_types, obs_dims)
                    if taut is not None and np.any(np.abs(get_trajectory_signature(taut, obs_centers_np)) >= 0.95):
                        state["tangled"] = True
        # ==========================
        # 记录数据与 2x2 可视化 
        # ==========================
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"Continuous Lifelong Navigation Benchmark (Time-to-Success) - Trial {trial+1}", fontsize=22, weight='bold')
        axes = axes.flatten()

        for i, method in enumerate(methods_list):
            st = tracker[method]
            ax = axes[i]
            dynamic_env.render(ax)
            ax.set_title(method, fontsize=16, weight='bold', color='darkred' if 'TMPD' in method else 'black')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)

            for k, wp in enumerate(waypoints_np):
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

            valid_segs = len(st["history"])
            line_color = 'darkred' if 'TMPD' in method else 'blue'
            
            for seg_idx, traj in enumerate(st["history"]):
                ax.plot(traj[:, 0], traj[:, 1], color=line_color, linewidth=2.0, alpha=0.6)
                mid = len(traj) // 2
                ax.annotate('', xy=(traj[mid+1, 0], traj[mid+1, 1]), xytext=(traj[mid, 0], traj[mid, 1]), arrowprops=dict(arrowstyle="->", color=line_color, lw=1.5, alpha=0.6))

            if valid_segs > 0:
                full_hist = np.concatenate(st["history"])
                taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)
                if taut_traj is not None:
                    ax.plot(taut_traj[:, 0], taut_traj[:, 1], color='darkorange', linewidth=3.5, linestyle='--', alpha=0.9, zorder=15, label='Topological Taut-Curve')
            
            # 【画出被枪毙的非法轨迹 (罪证)】
            if st["failed_traj"] is not None:
                f_traj = st["failed_traj"]
                ax.plot(f_traj[:, 0], f_traj[:, 1], color='red', linewidth=2.5, linestyle=':', alpha=0.8, zorder=18, label='Resampled Invalid Path')
                ax.plot(f_traj[-1, 0], f_traj[-1, 1], 'rx', markersize=12, markeredgewidth=2, zorder=19)

            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(loc='lower right', fontsize=11, framealpha=0.9)                    
            
            if st["fatal_error"]: status_text, box_color = "Trapped (Failed)", 'lightgray'
            else: status_text, box_color = "Success", 'lightgreen'
            
            props = dict(boxstyle='round', facecolor=box_color, alpha=0.9)
            ax.text(0.05, 0.95, f"Status: {status_text}\nSegments: {valid_segs}/{num_segments}", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, weight='bold', zorder=30)

            all_results.append({
                "Method": method, "Trial": trial + 1,
                "Success_Rate": st["sr"] / num_segments,
                "Tangle_Free_Rate": 1.0 if (not st["tangled"] and not st["fatal_error"] and st["sr"] == num_segments) else 0.0,
                "Avg_Seg_Time": st["time"] / max(1, st["sr"]),
                "Path_Length": np.mean(st["pl_list"]) if st["pl_list"] else np.nan,
                "Smoothness": np.mean(st["sm_list"]) if st["sm_list"] else np.nan,
            })

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"all_trial_{trial+1}.png"), dpi=200); plt.close(fig)

    # ==========================
    # 统计汇总
    # ==========================
    df = pd.DataFrame(all_results)
    df['Method'] = pd.Categorical(df['Method'], categories=methods_list, ordered=True)
    summary = df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Avg_Seg_Time": lambda x: f"{np.nanmean(x):.2f}s ± {np.nanstd(x):.2f}",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x)*1000:.3f} ± {np.nanstd(x)*1000:.3f}"
    })
    
    print("\n" + "="*85)
    print("🏆 FULL BENCHMARK REPORT (Time-to-Success)")
    print("="*85)
    print(summary.to_string())
    print("="*85)

if __name__ == '__main__':
    run_experiment(run_benchmark_all)