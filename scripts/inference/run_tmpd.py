import os
import time
import pickle
from math import ceil
from pathlib import Path
import numpy as np
import pandas as pd
import einops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import math
from matplotlib.widgets import CheckButtons

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import guide_gradient_steps, ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length, compute_variance_waypoints

from tmpd_baselines.environment.env_dense_2d_extra_objects import EnvDense2DExtraObjects

from mpd.utils.topology_utils import (
    get_trajectory_signature,
    prune_self_intersections,
    get_simplest_homotopy_curve,
    evaluate_homotopy_topological_energy,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
def apply_laplacian_smoothing(traj, iters=5):
    """
    在不改变轨迹拓扑和起终点的情况下，消除高频抖动。
    iters=5 是一个非常安全的保守值，能在大幅降低 Smoothness Cost 的同时防止轨迹“切角”撞墙。
    """
    if len(traj) < 3: return traj
    smoothed = np.copy(traj)
    for _ in range(iters):
        # 核心公式: 当前点 = 自身占50% + 左右邻居各占25% (起终点绝对不动)
        smoothed[1:-1] = 0.5 * smoothed[1:-1] + 0.25 * smoothed[:-2] + 0.25 * smoothed[2:]
    return smoothed
def resample_trajectory(traj_np, n_points):
    if len(traj_np) == n_points: return traj_np
    if len(traj_np) < 2: return np.zeros((n_points, traj_np.shape[1]))
    diffs = np.linalg.norm(np.diff(traj_np, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(diffs), 0, 0)
    if cum_dists[-1] == 0: return np.tile(traj_np[0], (n_points, 1))
    resampled = np.zeros((n_points, traj_np.shape[1]))
    for i in range(traj_np.shape[1]): 
        resampled[:, i] = np.interp(np.linspace(0, cum_dists[-1], n_points), cum_dists, traj_np[:, i])
    return resampled
def render_demo(
    fig, ax, iteration_count, env, history_path_list, 
    latest_all_trajs, latest_best_traj, current_start_pos,
    latest_optimized_traj=None,
    unique_homotopy_classes=None,
    homotopy_energies=None,
    homotopy_indices=None,
    VIS_STATE=None,      
    current_lines=None,   
    latest_best_energy=None,
    target_pos=None,
    trial_idx=1,
    waypoints_np=None
):
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f"Trial {trial_idx} - Segment {iteration_count}: Auto Planning...", fontsize=13, fontweight='bold')
    # ax.grid(False, linestyle='--', alpha=0.5)

    if current_lines is not None:
        for k in current_lines.keys():
            current_lines[k].clear()

    env.render(ax)
    
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    for k, center in enumerate(obs_centers_np):
        ax.text(center[0], center[1], f"O{k+1}", color='white', ha='center', va='center', weight='bold', fontsize=9, zorder=10)
    # ==========================================
    # 【新增逻辑】：渲染全局规划航点 (Waypoints)
    # ==========================================
    if waypoints_np is not None:
        for k, wp in enumerate(waypoints_np):
            if k == 0:
                ax.plot(wp[0], wp[1], 's', color='green', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1]-0.08, 'S', color='green', ha='center', va='top', weight='bold', zorder=21)
            else:
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

    if len(history_path_list) > 0:
        # 使用 winter 色卡 (蓝 -> 绿)
        cmap = plt.cm.winter
        segment_colors = cmap(np.linspace(0.0, 1.0, max(1, len(history_path_list))))
        
        for seg_idx, traj in enumerate(history_path_list):
            if len(traj) < 2: continue
            c = segment_colors[seg_idx]
            
            # 画单段实线
            lines = ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle='-', linewidth=2.5, alpha=0.9, label='Raw History' if seg_idx==0 else "", visible=VIS_STATE['Raw History'])
            current_lines['Raw History'].extend(lines)
            
            # 加上精细箭头 (60%位置)
            if VIS_STATE['Raw History']:
                mid = int(len(traj) * 0.6)
                if len(traj) > 2:
                    dx, dy = traj[mid+1, 0] - traj[mid, 0], traj[mid+1, 1] - traj[mid, 1]
                    norm = math.hypot(dx, dy)
                    if norm > 0:
                        ax.arrow(traj[mid, 0], traj[mid, 1], dx/norm*0.001, dy/norm*0.001, 
                                 shape='full', lw=0, length_includes_head=True, 
                                 head_width=0.02, head_length=0.03, color=c, zorder=25)

        # 起点标识
        if waypoints_np is None:
            # 起点标识
            initial_pt = history_path_list[0][0]
            ax.plot(initial_pt[0], initial_pt[1], 'bs', markersize=8, markeredgecolor='white', zorder=12)
            
            # 途经航点标识
            reached_pts = np.array([t[-1] for t in history_path_list])
            ax.plot(reached_pts[:, 0], reached_pts[:, 1], 'bo', markersize=6, markeredgecolor='white', zorder=12)
    if latest_optimized_traj is not None:
        lines = ax.plot(latest_optimized_traj[:, 0], latest_optimized_traj[:, 1], 
                        color='cyan', linestyle='-', linewidth=4.0, alpha=0.4, 
                        solid_capstyle='round', label='Global Taut Cable', 
                        visible=VIS_STATE['Global H1'])
        current_lines['Global H1'].extend(lines)

    # if latest_best_traj is not None:
        # lines = ax.plot(latest_best_traj[:, 0], latest_best_traj[:, 1], 
        #                 color='darkred', linewidth=2.5, label='MPD Candidate', 
        #                 visible=VIS_STATE['MPD Candidate'])
        # current_lines['MPD Candidate'].extend(lines)
        
        # if latest_best_energy is not None:
        #     mid_idx = len(latest_best_traj) // 2
        #     label_text = f"MPD_Best (E:{latest_best_energy:.1f})"
            # txt = ax.text(latest_best_traj[mid_idx, 0], latest_best_traj[mid_idx, 1], label_text, 
            #               color='fuchsia', weight='bold', fontsize=10, 
            #               bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), 
            #               zorder=30, visible=VIS_STATE['MPD Candidate'])
            # current_lines['MPD Candidate'].append(txt)

    if latest_all_trajs is not None:
        for i in range(len(latest_all_trajs)):
            lines = ax.plot(latest_all_trajs[i, :, 0], latest_all_trajs[i, :, 1], 
                            color='red', linestyle='-', linewidth=1.0, alpha=0.15, 
                            visible=VIS_STATE['All MPD'])
            current_lines['All MPD'].extend(lines)

    if unique_homotopy_classes is not None and homotopy_energies is not None:
        colors = ['magenta', 'orange', 'lime', 'cyan', 'yellow', 'pink']
        for i, (traj, energy) in enumerate(zip(unique_homotopy_classes, homotopy_energies)):
            c = colors[i % len(colors)]
            if i == 0:
                vis_key = 'Ref Topo'
                lw = 3.0       
                ls = '--'
            else:
                vis_key = 'All Topo'
                lw = 1.8       
                ls = ':'       
            lines = ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle=ls, linewidth=lw, alpha=0.9, solid_capstyle='round', visible=VIS_STATE[vis_key])
            current_lines[vis_key].extend(lines)
            h_id = homotopy_indices[i] if homotopy_indices is not None else i + 1
            label_text = f"H{h_id} (E:{energy:.1f})"
            mid_idx = len(traj) // 2
            txt = ax.text(traj[mid_idx, 0], traj[mid_idx, 1], label_text, color=c, weight='bold', fontsize=10, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), zorder=25, visible=VIS_STATE[vis_key])
            current_lines[vis_key].append(txt)

    ax.plot(current_start_pos[0].item(), current_start_pos[1].item(), 'go', markersize=12, label='Current Pos')
    
    if target_pos is not None:
        ax.plot(target_pos[0], target_pos[1], 'rx', markersize=12, markeredgewidth=3, zorder=35, label='Target Pos')
    
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # if handles:
    #     ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

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

@single_experiment_yaml
def experiment(
    model_id: str = 'EnvDense2D-RobotPointMass',
    planner_alg: str = 'tmpd',
    use_guide_on_extra_objects_only: bool = False,
    n_samples: int = 70,  #
    start_guide_steps_fraction: float = 0.1, #0.25
    n_guide_steps: int = 10,
    n_diffusion_steps_without_noise: int = 10, 
    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,
    factor_num_interpolated_points_for_collision: float = 1.5,
    trajectory_duration: float = 5.0,  
    device: str = 'cuda',
    debug: bool = True,
    render: bool = False, 
    seed: int = 42,
    results_dir: str = 'logs',
    # 新增批量参数，但不改变你原有的参数默认值
    num_trials: int = 100,
    num_segments: int = 5,
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'##########################################################################################################')
    print(f'Model -- {model_id} | Algorithm -- {planner_alg}')
    print(f'Running Benchmark: {num_trials} Trials x {num_segments} Segments')
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, 'results_inference', f'benchmark_seed_{seed}')
    os.makedirs(results_dir, exist_ok=True)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05,
        **args, tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    robot = dataset.robot
    base_task = dataset.task 

    env = EnvDense2DExtraObjects(
        tensor_args=tensor_args, 
        drop_old_num=2, 
        num_extra_spheres=12, 
        num_extra_boxes=12, 
        seed=42
    )
    task = type(base_task)(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
    
    dataset.env = env
    dataset.task = task

    dt = trajectory_duration / n_support_points 
    robot.dt = dt

    diffusion_configs = dict(variance_schedule=args['variance_schedule'], n_diffusion_steps=args['n_diffusion_steps'], predict_epsilon=args['predict_epsilon'])
    unet_configs = dict(state_dim=dataset.state_dim, n_support_points=dataset.n_support_points, unet_input_dim=args['unet_input_dim'], dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']])
    
    model = get_model(model_class=args['diffusion_model_class'], model=TemporalUnet(**unet_configs), tensor_args=tensor_args, **diffusion_configs, **unet_configs)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'), map_location=tensor_args['device']))
    model.eval()
    freeze_torch_model_params(model)
    model = torch.compile(model)
    model.warmup(horizon=n_support_points, device=device)

    obs_centers_np = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', ['sphere'] * len(obs_centers_np))
    obs_dims = getattr(env, 'active_obs_dims', [np.array([0.125])] * len(obs_centers_np))

    VIS_STATE = {
        'Raw History': True,
        'Global H1': True,
        'MPD Candidate': True,
        'All MPD': True,
        'Ref Topo': True,
        'All Topo': False
    }

    all_results = []
    
    # 为了防止跑 100 次开 100 个窗口卡死，我们只初始化一次幕布（不使用阻塞）
    plt.ion()  
    fig_topo, ax_topo = plt.subplots(figsize=(10, 10))
    fig_topo.canvas.manager.set_window_title("MPD Auto Benchmark")

    for trial in range(num_trials):
        print(f"\n====================== [ STARTING TRIAL {trial+1}/{num_trials} ] ======================")
        
        # 自动生成 5 段航点
        waypoints_t, waypoints_np = generate_sequential_waypoints(env, task, 2, tensor_args, num_segments=num_segments)
        if waypoints_t is None:
            print(f"Failed to generate safe waypoints for Trial {trial+1}. Skipping.")
            continue

        start_state_pos = torch.zeros(2, **tensor_args)
        start_state_pos[:2] = waypoints_t[0]
        
        history_path_list = []  
        current_start_pos = start_state_pos.clone()
        
        latest_all_trajs = None
        latest_best_traj = None
        latest_optimized_traj = None
        latest_best_energy = None
        current_homotopy_classes = None
        current_homotopy_energies = None
        current_homotopy_indices = None
        current_lines = {k: [] for k in VIS_STATE.keys()}
        
        # 【新增 final_energy】
        metrics_log = {
            "Method": planner_alg, "Trial": trial + 1, "history": [],
            "sr": 0, "time": 0.0, "fatal_error": False, "tangled": False,
            "pl_list": [], "sm_list": [], "final_energy": 0.0
        }

        for seg_idx in range(num_segments):
            iteration_count = seg_idx + 1
            target_x, target_y = waypoints_np[seg_idx + 1]
            print(f"\n  [Seg {iteration_count}] Auto Goal: ({target_x:.3f}, {target_y:.3f})")
            
            render_demo(fig_topo, ax_topo, iteration_count, env, history_path_list, latest_all_trajs, latest_best_traj, current_start_pos,
                        latest_optimized_traj, current_homotopy_classes, current_homotopy_energies, current_homotopy_indices,
                        VIS_STATE, current_lines, latest_best_energy, target_pos=(target_x, target_y), trial_idx=trial+1, waypoints_np=waypoints_np)
            plt.pause(0.1) # 短暂刷新一下画面
            
            t0_seg = time.time()
            
            goal_state_pos = torch.zeros_like(current_start_pos)
            goal_state_pos[0], goal_state_pos[1] = target_x, target_y
            start_state_pos = current_start_pos.clone()
            
            hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
            
            hist_for_eval = np.concatenate(history_path_list, axis=0) if history_path_list else np.array([])
            if len(hist_for_eval) > 0 and len(obs_centers_np) > 0:
                refined_hist = get_simplest_homotopy_curve(hist_for_eval, obs_centers_np, obs_types, obs_dims)
                if refined_hist is not None:
                    hist_for_eval = refined_hist
                    
            cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) 
                                for f in (task.get_collision_fields_extra_objects() if use_guide_on_extra_objects_only else task.get_collision_fields())]
            cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, trajectory_duration/n_support_points, sigma_gp=1.0, tensor_args=tensor_args)], 
                                           weights_cost_l=[weight_grad_cost_collision]*len(cost_collision_l) + [weight_grad_cost_smoothness], tensor_args=tensor_args)

            guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision), tensor_args=tensor_args)
            t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)
            sample_fn_kwargs = dict(guide=guide, n_guide_steps=n_guide_steps, t_start_guide=t_start_guide, noise_std_extra_schedule_fn=lambda x: 0.8)

            with TimerCUDA() as timer_model_sampling:
                trajs_normalized_iters = model.run_inference(None, hard_conds, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, **sample_fn_kwargs, n_diffusion_steps_without_noise=n_diffusion_steps_without_noise)
            
            _, _, trajs_final_free, _, _ = task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(trajs_normalized_iters)[-1], return_indices=True)

            if trajs_final_free is not None:
                trajs_free_np = trajs_final_free[..., :2].cpu().numpy()
                unique_mpd_classes, unique_sigs = [], []
                
                for traj in trajs_free_np:
                    sig = get_trajectory_signature(traj, obs_centers_np)
                    if not any(np.all(np.abs(sig - ext_sig) < 0.3) for ext_sig in unique_sigs):
                        unique_sigs.append(sig)
                        unique_mpd_classes.append(prune_self_intersections(traj))
                        
                mpd_energies, _ = evaluate_homotopy_topological_energy(hist_for_eval, unique_mpd_classes, obs_centers_np, w_max=0.8)
                combined_scores = []
                for traj, energy in zip(unique_mpd_classes, mpd_energies):
                    combined_raw = np.vstack((hist_for_eval, traj[1:])) if len(hist_for_eval) > 0 else traj
                    refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                    check_traj = refined_combined if refined_combined is not None else combined_raw
                    global_sig = get_trajectory_signature(check_traj, obs_centers_np)
                    
                    if np.any(np.abs(global_sig) >= 0.98):
                        tangle_penalty = 10000.0  
                    else:
                        tangle_penalty = 0.0
                        
                    score = energy + 0.8 * np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) + tangle_penalty
                    combined_scores.append(score)
                
                sorted_data = sorted(zip(range(1, len(unique_mpd_classes) + 1), unique_mpd_classes, mpd_energies, combined_scores), key=lambda x: x[3])
                
                current_homotopy_indices = [x[0] for x in sorted_data]
                current_homotopy_classes = [x[1] for x in sorted_data]
                current_homotopy_energies = [x[2] for x in sorted_data]
                
                latest_best_traj = current_homotopy_classes[0]
                latest_best_energy = current_homotopy_energies[0]

                # latest_best_traj = resample_trajectory(latest_best_traj, n_support_points)
                
                # # 消除时间上的加速度毛刺 (重采样分配到均匀点距)

                path_length = np.sum(np.linalg.norm(np.diff(latest_best_traj, axis=0), axis=1))
                smoothness = compute_smoothness(torch.tensor(latest_best_traj, **tensor_args).unsqueeze(0), robot).item()
                if len(hist_for_eval) > 0:
                    combined_raw_traj = np.vstack((hist_for_eval, latest_best_traj[1:]))
                else:
                    combined_raw_traj = latest_best_traj
                    
                latest_optimized_traj = get_simplest_homotopy_curve(combined_raw_traj, obs_centers_np, obs_types, obs_dims)

                history_path_list.append(latest_best_traj)
                current_start_pos[:2] = torch.tensor(latest_best_traj[-1], dtype=torch.float32, device=device)
                latest_all_trajs = trajs_free_np
                
                metrics_log["time"] += (time.time() - t0_seg)
                metrics_log["sr"] += 1
                metrics_log["pl_list"].append(path_length)
                metrics_log["sm_list"].append(smoothness)
                metrics_log["history"].append(latest_best_traj)
                if latest_optimized_traj is not None:
                    curr_sig = get_trajectory_signature(latest_optimized_traj, obs_centers_np)
                    metrics_log["final_energy"] = np.sum(np.abs(curr_sig))
                
                print(f"  [Seg {iteration_count}] Success! Best Topology Energy: {latest_best_energy:.2f}")
            else:
                print(f"  [Seg {iteration_count}] All collision!!! Fatal Error.")
                metrics_log["fatal_error"] = True
                latest_all_trajs, latest_best_traj, latest_optimized_traj, latest_best_energy = None, None, None, None
                break

        # ==========================
        # 当前 Trial 结束：验证纠缠并保存图表
        # ==========================
        if not metrics_log["fatal_error"] and metrics_log["sr"] > 0:
            full_hist = np.concatenate(metrics_log["history"])
            taut_traj = get_simplest_homotopy_curve(full_hist, obs_centers_np, obs_types, obs_dims)
            if taut_traj is not None and np.any(np.abs(get_trajectory_signature(taut_traj, obs_centers_np)) >= 0.9):
                metrics_log["tangled"] = True
            render_demo(fig_topo, ax_topo, "Final", env, history_path_list, None, latest_best_traj, current_start_pos,
                        latest_optimized_traj, None, None, None,
                        VIS_STATE, current_lines, latest_best_energy, target_pos=None, trial_idx=trial+1)
            ax_topo.set_title(f"Trial {trial+1} Completed (SR: {metrics_log['sr']}/{num_segments}, E: {metrics_log['final_energy']:.2f})", color='green', weight='bold')
        else:
            render_demo(fig_topo, ax_topo, "Failed", env, history_path_list, None, None, current_start_pos,
                        None, None, None, None,
                        VIS_STATE, current_lines, None, target_pos=None, trial_idx=trial+1)
            ax_topo.set_title(f"Trial {trial+1} Failed (Collision)", color='red', weight='bold')

        plt.savefig(os.path.join(results_dir, f"auto_trial_{trial+1:03d}.png"), dpi=150)
        
        is_fully_successful = (not metrics_log["tangled"]) and (not metrics_log["fatal_error"]) and (metrics_log["sr"] == num_segments)
        attempted_segs = metrics_log["sr"] + (1 if metrics_log["fatal_error"] else 0)
        all_results.append({
            "Method": metrics_log["Method"], 
            "Success_Rate": metrics_log["sr"] / num_segments,
            "Tangle_Free_Rate": 1.0 if is_fully_successful else 0.0,
            
            # 【时间】：无条件记录！把总耗时除以实际尝试的段数，绝不丢弃任何一次运算开销
            "Avg_Seg_Time": metrics_log["time"] / max(1, attempted_segs),

            # 【轨迹质量】：仅在完美通关（无碰撞且无缠绕）时才计算，否则记为 NaN
            "Path_Length": np.mean(metrics_log["pl_list"]) if is_fully_successful else np.nan,
            "Smoothness": np.mean(metrics_log["sm_list"]) if is_fully_successful else np.nan,
            
            # 拓扑能量可以保留所有记录，用来在论文中展示失败者到底绕得有多惨
            "Final_Topo_Energy": metrics_log["final_energy"]
        })

    plt.close('all') # 跑完 100 次释放资源

    # ==========================
    # 最后打印你指定的总表 Metrics
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
    
    print("\n" + "="*95)
    print(f"🏆 FINAL BENCHMARK REPORT ({num_trials} Trials Auto)")
    print("="*95)
    print(summary.to_string())
    print("="*95)

if __name__ == '__main__':
    run_experiment(experiment)