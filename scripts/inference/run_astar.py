import os
import time
import numpy as np
import pandas as pd
import torch
import math
import heapq
import matplotlib
matplotlib.use('Agg') # 开启 Headless 极速画图模式
import matplotlib.pyplot as plt
import pickle
# 导入底层依赖 (无 MPD 模型依赖)
from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.trainer import get_dataset
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.metrics import compute_smoothness

# 导入动态密林环境与拓扑工具
from tmpd_baselines.environment.env_dense_2d_extra_objects import EnvDense2DExtraObjects
from mpd.utils.topology_utils import (
    get_trajectory_signature,
    get_simplest_homotopy_curve,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
RESULTS_DIR = 'benchmark_time_to_success_astar'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots') # 单独的图片保存文件夹
shared_data_dir = "benchmark_shared_data"
os.makedirs(shared_data_dir, exist_ok=True)

# ==========================================
# 共享组件：拓扑 A* 节点与绕数计算
# ==========================================
class TopoNode:
    def __init__(self, x, y, g, h, parent, W_array):
        self.x = x; self.y = y; self.g = g; self.h = h
        self.parent = parent; self.W_array = W_array
        self.f = g +  h 
    def __lt__(self, other): 
        return self.f < other.f
        
    def get_state_key(self): 
        return (round(self.x, 3), round(self.y, 3), tuple(np.round(self.W_array, 1)))

def calc_delta_winding_vectorized(p1, p2, obstacles):
    if len(obstacles) == 0: return np.array([])
    p1, p2 = np.array(p1), np.array(p2)
    v1, v2 = p1 - obstacles, p2 - obstacles
    theta1, theta2 = np.arctan2(v1[:, 1], v1[:, 0]), np.arctan2(v2[:, 1], v2[:, 0])
    delta_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return delta_theta / (2 * np.pi)

# ==========================================
# 算法核心：Topo-A*
# ==========================================
def lifelong_topo_a_star(start, goal, env, initial_W, step_size=0.05, W_th=0.98, robot_radius=0.05, max_time=30.0): 
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
    
    motions = [
        (0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0), 
        (step_size, step_size), (step_size, -step_size), (-step_size, step_size), (-step_size, -step_size)
    ]
    
    search_start_time = time.time()
    
    while open_set:
        if time.time() - search_start_time > max_time:
            print(f"      [Topo-A*] ⚠️ TIMEOUT: State space explosion! Search aborted after {max_time}s.")
            return None 
            
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
            
            new_W = curr.W_array + calc_delta_winding_vectorized((curr.x, curr.y), (nx, ny), obs_centers)
            if np.any(np.abs(new_W) >= W_th): continue  
            
            new_g = curr.g + math.hypot(dx, dy)
            new_h = math.hypot(nx - goal[0], ny - goal[1])
            heapq.heappush(open_set, TopoNode(nx, ny, new_g, new_h, curr, new_W))
            
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
# 评测主流程 (专注 Topo-A*)
# ==========================================
@single_experiment_yaml
def run_benchmark_topo_a_star(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    num_segments: int = 5,
    device: str = 'cpu',  # A* 完全跑在 CPU 上
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'🚀 Initializing "Time-to-Success" Benchmark (Topo-A* ONLY - Visualizer Enabled)...')
    print(f'📁 Plots will be saved to: {PLOTS_DIR}')
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points

    all_results = []

    for trial in range(num_trials):
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42+trial) # 加了 trial 防止每次一样
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', [])
        obs_types = getattr(dynamic_env, 'active_obs_types', [])
        obs_dims = getattr(dynamic_env, 'active_obs_dims', [])
        
        waypoints_t, waypoints_np = generate_sequential_waypoints(dynamic_env, dynamic_task, 2, tensor_args, num_segments)
        if waypoints_t is None: continue

        print(f"\n================ [ Trial {trial+1}/{num_trials} ] ================")
        
        tracker = {"history": [], "sr": 0, "time": 0.0, "fatal_error": False, "tangled": False, "pl_list": [], "sm_list": [], "final_energy": 0.0}

        for seg in range(num_segments):
            if tracker["fatal_error"]: continue
            
            start_np, goal_np = waypoints_np[seg], waypoints_np[seg+1]
            
            t0 = time.time()
            best_traj_np = None
            
            hist_for_eval = np.concatenate(tracker["history"]) if tracker["history"] else np.array([])
            initial_W = np.zeros(len(obs_centers_np))
            
            if len(hist_for_eval) > 0:
                refined_hist = get_simplest_homotopy_curve(hist_for_eval, obs_centers_np, obs_types, obs_dims)
                if refined_hist is not None: 
                    initial_W = np.array(get_trajectory_signature(refined_hist, obs_centers_np))

            path_list = lifelong_topo_a_star(start_np, goal_np, dynamic_env, initial_W, step_size=0.005, robot_radius=0.02) # 保持了 0.05 物理对齐
            
            if path_list: 
                best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
            else: 
                print(f"    [Topo-A*] ⚠️ FATAL: State space exhausted. Physically trapped at Segment {seg+1}!")
                tracker["fatal_error"] = True

            seg_time = time.time() - t0
            tracker["time"] += seg_time

            if best_traj_np is not None:
                tracker["history"].append(best_traj_np)
                tracker["sr"] += 1
                tracker["pl_list"].append(np.sum(np.linalg.norm(np.diff(best_traj_np, axis=0), axis=1)))
                tracker["sm_list"].append(compute_smoothness(torch.tensor(best_traj_np, **tensor_args).unsqueeze(0), robot).item())
                
                full_path = np.concatenate(tracker["history"])
                taut = get_simplest_homotopy_curve(full_path, obs_centers_np, obs_types, obs_dims)
                check_traj = taut if taut is not None else full_path
                
                sig = get_trajectory_signature(check_traj, obs_centers_np)
                tracker["final_energy"] = np.sum(np.abs(sig)) 
                
                if np.any(np.abs(sig) >= 0.95):
                    tracker["tangled"] = True

        # ==========================
        # 【画图逻辑修改】：生成蓝绿渐变色轨迹图
        # ==========================
        fig, ax = plt.subplots(figsize=(10, 10))
        dynamic_env.render(ax)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        
        if tracker["fatal_error"]:
            title_color = 'red'
            status_txt = f"Exhausted at Seg {tracker['sr']+1}"
        elif tracker["tangled"]:
            title_color = 'darkorange'
            status_txt = f"Tangled (Energy: {tracker['final_energy']:.2f})"
        else:
            title_color = 'green'
            status_txt = f"Success (Energy: {tracker['final_energy']:.2f})"
            
        ax.set_title(f"Topo-A* - Trial {trial+1} [{status_txt}]", fontsize=16, weight='bold', color=title_color)

        # 1. 生成 从蓝到绿 的高级渐变色卡 (使用 matplotlib 的 winter 或 viridis 的某一段)
        cmap = plt.cm.winter # winter 刚好是 蓝 -> 绿 的渐变
        segment_colors = cmap(np.linspace(0.0, 1.0, max(1, len(waypoints_np)-1)))

        # 2. 画航点
        for k, wp in enumerate(waypoints_np):
            if k == 0:
                ax.plot(wp[0], wp[1], 's', color='green', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1]-0.08, 'S', color='green', ha='center', va='top', weight='bold', zorder=21)
            else:
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

        # 3. 画出渐变色实线轨迹 + 方向箭头
        for seg_idx, traj in enumerate(tracker["history"]):
            if len(traj) < 2: continue
            
            c = segment_colors[seg_idx]
            # 主轨迹
            ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=3.5, alpha=0.85, label=f'Seg {seg_idx+1}')
            
            # 画箭头指示方向 (加在中间偏后的位置)
            mid = int(len(traj) * 0.6)
            if len(traj) > 2:
                dx, dy = traj[mid+1, 0] - traj[mid, 0], traj[mid+1, 1] - traj[mid, 1]
                norm = math.hypot(dx, dy)
                if norm > 0:
                    # 调整了基础长度、head_width(宽度) 和 head_length(长度)
                    ax.arrow(traj[mid, 0], traj[mid, 1], dx/norm*0.001, dy/norm*0.001, 
                             shape='full', lw=0, length_includes_head=True, 
                             head_width=0.02, head_length=0.03, color=c, zorder=25)

        # 4. 标出导致死锁的目标点 (用红叉高亮)
        if tracker["fatal_error"]:
            failed_wp = waypoints_np[tracker["sr"] + 1]
            ax.plot(failed_wp[0], failed_wp[1], 'rx', markersize=20, markeredgewidth=4, zorder=25, label='Unreachable Goal')

        # 5. 画出全局拉紧后的同伦曲线 (虚线)
        if len(tracker["history"]) > 0:
            full_hist = np.concatenate(tracker["history"])
            taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)
            if taut_traj is not None:
                if tracker["tangled"]:
                    ax.plot(taut_traj[:, 0], taut_traj[:, 1], color='red', linewidth=3.5, linestyle='--', alpha=0.9, zorder=15, label='Tangled Taut Curve')
                else:
                    ax.plot(taut_traj[:, 0], taut_traj[:, 1], color='darkorange', linewidth=2.5, linestyle=':', alpha=0.6, zorder=15, label='Safe Taut Curve')

        # handles, labels = ax.get_legend_handles_labels()
        # if handles: ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"topo_astar_trial_{trial+1:03d}.png"), dpi=150)
        plt.close(fig)

        # 记录指标
        is_fully_successful = (not tracker["tangled"]) and (not tracker["fatal_error"]) and (tracker["sr"] == num_segments)
        attempted_segs = tracker["sr"] + (1 if tracker["fatal_error"] else 0)
        
        all_results.append({
            "Method": "Topo-A*", "Trial": trial + 1,
            "Success_Rate": tracker["sr"] / num_segments,
            "Tangle_Free_Rate": 1.0 if is_fully_successful else 0.0,
            "Avg_Seg_Time": tracker["time"] / max(1, attempted_segs),
            "Path_Length": np.mean(tracker["pl_list"]) if is_fully_successful else np.nan,
            "Smoothness": np.mean(tracker["sm_list"]) if is_fully_successful else np.nan,
            "Final_Topo_Energy": tracker["final_energy"] if is_fully_successful else np.nan
        })

    # ==========================
    # 统计汇总
    # ==========================
    df = pd.DataFrame(all_results)
    summary = df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Avg_Seg_Time": lambda x: f"{np.nanmean(x):.2f}s ± {np.nanstd(x):.2f}",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x)*1000:.3f} ± {np.nanstd(x)*1000:.3f}",
        "Final_Topo_Energy": lambda x: f"{np.nanmean(x):.2f} ± {np.nanstd(x):.2f}"
    })
    
    print("\n" + "="*85)
    print("🏆 TOPO-A* ONLY REPORT")
    print("="*85)
    print(summary.to_string())
    print("="*85)

if __name__ == '__main__':
    run_experiment(run_benchmark_topo_a_star)