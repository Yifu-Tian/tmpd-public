import os
import time
import numpy as np
import pandas as pd
import torch
import math
import heapq
import matplotlib
matplotlib.use('Agg')  # Headless plotting backend.

# Core dependencies (no MPD model).
from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.trainer import get_dataset
from mpd.utils.bench_plotting import render_segmented_trial_plot
from mpd.utils.loading import load_params_from_yaml
from mpd.utils.bench_io import ensure_dir, resolve_output_root
from mpd.utils.bench_metrics import format_time_to_success_summary
from mpd.utils.waypoints import generate_sequential_waypoints, resample_trajectory
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.metrics import compute_smoothness

# Dynamic environment and topology utilities.
from mpd.environments.env_dense_2d_extra_objects import EnvDense2DExtraObjects
from mpd.utils.topology_utils import (
    calc_delta_winding_vectorized,
    get_trajectory_signature,
    get_simplest_homotopy_curve,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "data_trained_models")
RESULTS_DIR = os.path.join(RESULTS_ROOT, "benchmark_time_to_success_astar")

# Topo-A* node and winding helpers.
class TopoNode:
    def __init__(self, x, y, g, h, parent, W_array):
        self.x = x; self.y = y; self.g = g; self.h = h
        self.parent = parent; self.W_array = W_array
        self.f = g +  h 
    def __lt__(self, other): 
        return self.f < other.f
        
    def get_state_key(self): 
        return (round(self.x, 3), round(self.y, 3), tuple(np.round(self.W_array, 1)))

# Topo-A* search.
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
            print(f"      [Topo-A*] TIMEOUT: State space explosion! Search aborted after {max_time}s.")
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

# Benchmark entrypoint.
@single_experiment_yaml
def run_benchmark_topo_a_star(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    num_segments: int = 5,
    device: str = 'cpu',  # A* runs on CPU.
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    benchmark_results_dir = resolve_output_root(RESULTS_DIR, results_dir)
    plots_dir = os.path.join(benchmark_results_dir, 'plots')
    ensure_dir(benchmark_results_dir)
    ensure_dir(plots_dir)
    
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print('Initializing "Time-to-Success" Benchmark (Topo-A* ONLY - Visualizer Enabled)...')
    print(f'Plots will be saved to: {plots_dir}')
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points

    all_results = []

    for trial in range(num_trials):
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42+trial)  # Vary map per trial.
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

            path_list = lifelong_topo_a_star(start_np, goal_np, dynamic_env, initial_W, step_size=0.005, robot_radius=0.02)  # Keep collision model aligned.
            
            if path_list: 
                best_traj_np = resample_trajectory(np.array(path_list), n_support_points)
            else: 
                print(f"    [Topo-A*] FATAL: State space exhausted. Physically trapped at Segment {seg+1}!")
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
                
                if np.any(np.abs(sig) >= 0.98):
                    tracker["tangled"] = True

        taut_traj = None
        if len(tracker["history"]) > 0:
            full_hist = np.concatenate(tracker["history"])
            taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)

        if tracker["fatal_error"]:
            title_color = 'red'
            status_txt = f"Exhausted at Seg {tracker['sr']+1}"
        elif tracker["tangled"]:
            title_color = 'darkorange'
            status_txt = f"Tangled (Energy: {tracker['final_energy']:.2f})"
        else:
            title_color = 'green'
            status_txt = f"Success (Energy: {tracker['final_energy']:.2f})"

        failed_wp = waypoints_np[tracker["sr"] + 1] if tracker["fatal_error"] else None
        render_segmented_trial_plot(
            env=dynamic_env,
            waypoints_np=waypoints_np,
            history_trajs=tracker["history"],
            trial_idx=trial + 1,
            method_label="Topo-A*",
            status_txt=status_txt,
            title_color=title_color,
            output_path=os.path.join(plots_dir, f"topo_astar_trial_{trial+1:03d}.png"),
            taut_traj=taut_traj,
            is_tangled=tracker["tangled"],
            failed_goal=failed_wp,
            failed_goal_label="Unreachable Goal",
            failed_traj=None,
            dpi=150,
        )

        # Log metrics.
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

    # Summarize metrics.
    df = pd.DataFrame(all_results)
    summary = format_time_to_success_summary(df, include_final_topo_energy=True)
    
    print("\n" + "="*85)
    print("TOPO-A* ONLY REPORT")
    print("="*85)
    print(summary.to_string())
    print("="*85)

if __name__ == '__main__':
    run_experiment(run_benchmark_topo_a_star)
