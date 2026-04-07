import os
import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil

# 导入底层依赖
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
from tmpd_baselines.environment.env_simple_2d_extra_objects import EnvSimple2DExtraObjects
from mpd.utils.topology_utils import (
    get_trajectory_signature, 
    evaluate_homotopy_topological_energy, 
    prune_self_intersections,
    get_simplest_homotopy_curve,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
SEQ_PLOTS_DIR = 'sequential_plots'
TANGLE_DIR = 'tangle_cases'

def plot_tangle_diagnostic(ax, env, waypoints_np, history, taut_traj, sig, title, color):
    """渲染拓扑失效诊断图"""
    env.render(ax)
    if len(history) > 0:
        full_raw = np.concatenate(history)
        ax.plot(full_raw[:, 0], full_raw[:, 1], color='gray', alpha=0.3, label='Raw Path')
    if taut_traj is not None:
        ax.plot(taut_traj[:, 0], taut_traj[:, 1], color=color, linewidth=4, label='Tangled Taut-Curve')
        # 标注故障点
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

@single_experiment_yaml
def run_sequential_benchmark(
    model_id: str = 'EnvSimple2D-RobotPointMass',
    n_samples: int = 70,
    num_trials: int = 100,
    num_segments: int = 5,
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(SEQ_PLOTS_DIR, exist_ok=True)
    os.makedirs(TANGLE_DIR, exist_ok=True)
    # 这里的全局种子只管模型加载和地图生成
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points; dt = 5.0 / n_support_points 

    model = get_model(model_class=args['diffusion_model_class'], model=TemporalUnet(state_dim=dataset.state_dim, n_support_points=n_support_points, unet_input_dim=args['unet_input_dim'], dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']]), tensor_args=tensor_args, variance_schedule=args['variance_schedule'], n_diffusion_steps=args['n_diffusion_steps'], predict_epsilon=args['predict_epsilon'])
    model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'), map_location=device))
    model.eval(); freeze_torch_model_params(model); model.warmup(horizon=n_support_points, device=device)

    all_results = []

    for trial in range(num_trials):
        dynamic_env = EnvSimple2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42)
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', []); obs_types = getattr(dynamic_env,'active_obs_types',[]); obs_dims = getattr(dynamic_env,'active_obs_dims',[])
        
        waypoints_t, waypoints_np = generate_sequential_waypoints(dynamic_env, dynamic_task, 2, tensor_args, num_segments)
        if waypoints_t is None: continue

        cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) for f in dynamic_task.get_collision_fields()]
        cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)], weights_cost_l=[1e-2]*len(cost_collision_l) + [1e-7], tensor_args=tensor_args)
        guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * 1.5), tensor_args=tensor_args)

        print(f"\nTrial {trial+1}/{num_trials}: Processing segments...")

        # =================================================================================
        # 🟢 [A] 独立运行 Vanilla MPD
        # =================================================================================
        fix_random_seed(seed) 
        
        history_v, curr_v = [], waypoints_t[0]
        is_tangled_v, collision_v = False, False
        sr_v, time_v = 0, 0.0
        pl_v_list, sm_v_list = [], []

        for seg in range(num_segments):
            goal_seg = waypoints_t[seg+1]
            if not collision_v:  # 仅仅因为碰撞而终止，不因缠绕而终止
                t0 = time.time()
                h_v = dataset.get_hard_conditions(torch.vstack((curr_v, goal_seg)), normalize=True)
                t_v = model.run_inference(None, h_v, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=10, t_start_guide=ceil(0.25*model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.5, n_diffusion_steps_without_noise=5)
                _, _, free_v, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(t_v)[-1], return_indices=True)
                time_v += (time.time() - t0)
                
                if free_v is not None:
                    best_v = free_v[0, ..., :2].cpu().numpy()
                    history_v.append(best_v); curr_v = torch.tensor(best_v[-1], **tensor_args); sr_v += 1
                    pl_v_list.append(np.sum(np.linalg.norm(np.diff(best_v, axis=0), axis=1)))
                    sm_v_list.append(np.mean(np.square(np.diff(best_v, n=2, axis=0))))
                    
                    # 拓扑判定
                    full_v = np.concatenate(history_v)
                    taut_v = get_simplest_homotopy_curve(full_v, obs_centers_np, obs_types, obs_dims)
                    if taut_v is not None:
                        sig_v = get_trajectory_signature(taut_v, obs_centers_np)
                        if np.any(np.abs(sig_v) >= 0.95):
                            if not is_tangled_v: # 只抓拍并保存第一次发生缠绕的瞬间
                                fig, ax = plt.subplots(); plot_tangle_diagnostic(ax, dynamic_env, waypoints_np, history_v, taut_v, sig_v, f"Vanilla Tangle (Seg {seg})", 'blue')
                                plt.savefig(os.path.join(TANGLE_DIR, f"vanilla_tangle_trial{trial+1}_seg{seg}.png")); plt.close()
                            is_tangled_v = True
                else: 
                    collision_v = True # 撞墙了，物理走不通，终止该段
            else: break

        # =================================================================================
        # 🔴 [B] 独立运行 TMPD (Ours) 
        # =================================================================================
        fix_random_seed(seed) 

        history_t, curr_t = [], waypoints_t[0]
        is_tangled_t, collision_t = False, False
        sr_t, time_t = 0, 0.0
        pl_t_list, sm_t_list = [], []

        for seg in range(num_segments):
            goal_seg = waypoints_t[seg+1]
            if not collision_t:
                t0 = time.time()
                h_t = dataset.get_hard_conditions(torch.vstack((curr_t, goal_seg)), normalize=True)
                hist_mem = np.concatenate(history_t) if history_t else np.array([])
                if len(hist_mem) > 0:
                    refined = get_simplest_homotopy_curve(hist_mem, obs_centers_np, obs_types, obs_dims)
                    if refined is not None: hist_mem = refined
                
                t_t = model.run_inference(None, h_t, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=10, t_start_guide=ceil(0.2*model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.8, n_diffusion_steps_without_noise=10)
                _, _, free_t, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(t_t)[-1], return_indices=True)
                time_t += (time.time() - t0)
                
                if free_t is not None:
                    t_np = free_t[..., :2].cpu().numpy(); u_classes = [prune_self_intersections(traj) for traj in t_np]
                    energies, _ = evaluate_homotopy_topological_energy(hist_mem, u_classes, obs_centers_np)
                    
                    scores = []
                    for traj, e in zip(u_classes, energies):
                        t_tensor = torch.tensor(traj, **tensor_args).unsqueeze(0)
                        pl = compute_path_length(t_tensor, robot).item()
                        sm = compute_smoothness(t_tensor, robot).item()
                        scores.append(e + 0.8 * pl + 1.0 * sm)

                    best_t = u_classes[np.argmin(scores)]
                    history_t.append(best_t); curr_t = torch.tensor(best_t[-1], **tensor_args); sr_t += 1
                    
                    pl_t_list.append(np.sum(np.linalg.norm(np.diff(best_t, axis=0), axis=1)))
                    sm_t_list.append(np.mean(np.square(np.diff(best_t, n=2, axis=0))))
                    
                    full_t = np.concatenate(history_t)
                    taut_t = get_simplest_homotopy_curve(full_t, obs_centers_np, obs_types, obs_dims)
                    if taut_t is not None:
                        sig_t = get_trajectory_signature(taut_t, obs_centers_np)
                        if np.any(np.abs(sig_t) >= 0.95):
                            if not is_tangled_t:
                                fig, ax = plt.subplots(); plot_tangle_diagnostic(ax, dynamic_env, waypoints_np, history_t, taut_t, sig_t, f"TMPD Tangle (Seg {seg})", 'red')
                                plt.savefig(os.path.join(TANGLE_DIR, f"tmpd_tangle_trial{trial+1}_seg{seg}.png")); plt.close()
                            is_tangled_t = True
                else: 
                    collision_t = True
            else: break

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f"Continuous Lifelong Navigation - Trial {trial+1}", fontsize=18, weight='bold')
        
        methods = ["Vanilla MPD (Memoryless)", "TMPD (Global Topology Memory)"]
        histories = [history_v, history_t]
        
        # 定义状态标签：先判断是否碰撞，再判断是否打结，全活下来才是 Success
        def get_status(collision_flag, tangled_flag):
            if collision_flag: return "Collision (Failed)", 'lightgray'
            if tangled_flag: return "Tangled (Failed)", 'lightcoral'
            return "Success", 'lightgreen'

        status_v, color_v = get_status(collision_v, is_tangled_v)
        status_t, color_t = get_status(collision_t, is_tangled_t)
        statuses = [(status_v, color_v), (status_t, color_t)]
        colors_line = ['blue', 'darkred']

        for i, ax in enumerate(axes):
            dynamic_env.render(ax)
            ax.set_title(methods[i], fontsize=15, weight='bold')
            ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            for k, wp in enumerate(waypoints_np):
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=14, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

            valid_segs = len(histories[i])
            for seg_idx, traj in enumerate(histories[i]):
                ax.plot(traj[:, 0], traj[:, 1], color=colors_line[i], linewidth=2.5, alpha=0.8, label='Executed Path' if seg_idx == 0 else "")
                mid = len(traj) // 2
                ax.annotate('', xy=(traj[mid+1, 0], traj[mid+1, 1]), xytext=(traj[mid, 0], traj[mid, 1]), arrowprops=dict(arrowstyle="->", color=colors_line[i], lw=2))
            
            status_text, box_color = statuses[i]
            props = dict(boxstyle='round', facecolor=box_color, alpha=0.9)
            ax.text(0.05, 0.05, f"Status: {status_text}\nValid Segments: {valid_segs}/{num_segments}", transform=ax.transAxes, fontsize=12, verticalalignment='bottom', bbox=props, weight='bold', zorder=30)
            
            if valid_segs > 0: ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(SEQ_PLOTS_DIR, f"sequential_trial_{trial:03d}.png"), dpi=200); plt.close(fig)

        # =================================================================================
        # 📊 [D] 录入指标 (解耦 SR 与 TFR)
        # =================================================================================
        all_results.append({
            "Method": "MPD", "Trial": trial,
            "Success_Rate(Collision-free&Goal Reached)": sr_v / num_segments, 
            "Tangle_Free_Rate": 1.0 if (not is_tangled_v and sr_v == num_segments) else 0.0,
            "Path_Length": np.mean(pl_v_list) if pl_v_list else np.nan,
            "Smoothness": np.mean(sm_v_list) if sm_v_list else np.nan,
            "Time": time_v
        })
        all_results.append({
            "Method": "TMPD (Ours)", "Trial": trial,
            "Success_Rate(Collision-free&Goal Reached)": sr_t / num_segments, 
            "Tangle_Free_Rate": 1.0 if (not is_tangled_t and sr_t == num_segments) else 0.0,
            "Path_Length": np.mean(pl_t_list) if pl_t_list else np.nan,
            "Smoothness": np.mean(sm_t_list) if sm_t_list else np.nan,
            "Time": time_t
        })

    # =================================================================================
    # 📈 [E] 统计与打印表格
    # =================================================================================
    df = pd.DataFrame(all_results)
    
    summary = df.groupby("Method").agg({
        "Success_Rate(Collision-free&Goal Reached)": lambda x: f"{np.mean(x)*100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.nanmean(x)*100:.1f}%" if np.sum(~np.isnan(x)) > 0 else "0.0%",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x)*1000:.3f} ± {np.nanstd(x)*1000:.3f}", # 放大 1000 倍展示精度
        "Time": lambda x: f"{np.nanmean(x):.2f}s"
    })
    
    print("\n" + "="*95)
    print("📊 BENCHMARK REPORT")
    print("="*95)
    print(summary.to_string())
    print("="*95)
    print("\n* Note: Smoothness values are multiplied by 10^3 for better precision display.")

if __name__ == '__main__':
    run_experiment(run_sequential_benchmark)