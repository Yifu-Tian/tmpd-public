import os
import pickle
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from math import ceil

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

from tmpd_baselines.environment.env_dense_2d_extra_objects import EnvDense2DExtraObjects
from mpd.utils.topology_utils import get_trajectory_signature, evaluate_homotopy_topological_energy, prune_self_intersections

TRAINED_MODELS_DIR = '../../data_trained_models/'
CASES_SAVE_PATH = 'hard_cases_100.pkl'
PLOTS_DIR = 'benchmark_plots_dense'
def sample_collision_free_start_goal(task, tensor_args, min_dist=0.8, max_attempts=5000):
    for _ in range(max_attempts):
        start_np = np.random.uniform(-0.85, 0.85, size=2)
        goal_np = np.random.uniform(-0.85, 0.85, size=2)
        
        # 保证起终点之间有足够的距离，避免生成过于简单的任务
        if np.linalg.norm(start_np - goal_np) < min_dist:
            continue
            
        start_t = torch.tensor(start_np, **tensor_args)
        goal_t = torch.tensor(goal_np, **tensor_args)
        
        # 检查起点和终点是否都不在障碍物内部
        if task.compute_collision(start_t.unsqueeze(0)).item() == 0 and \
           task.compute_collision(goal_t.unsqueeze(0)).item() == 0:
            return start_np, goal_np, start_t, goal_t
            
    print("Warning: Failed to sample valid start/goal points within max attempts.")
    return None, None, None, None

@single_experiment_yaml
def run_benchmark_and_plot(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    n_samples: int = 70,
    trajectory_duration: float = 5.0,
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    # 1. 挂载模型
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    dataset, robot, base_task = train_subset.dataset, train_subset.dataset.robot, train_subset.dataset.task
    n_support_points = dataset.n_support_points
    dt = trajectory_duration / n_support_points 

    unet_configs = dict(state_dim=dataset.state_dim, n_support_points=n_support_points, unet_input_dim=args['unet_input_dim'], dim_mults=UNET_DIM_MULTS[args['unet_dim_mults_option']])
    model = get_model(model_class=args['diffusion_model_class'], model=TemporalUnet(**unet_configs), tensor_args=tensor_args, variance_schedule=args['variance_schedule'], n_diffusion_steps=args['n_diffusion_steps'], predict_epsilon=args['predict_epsilon'], **unet_configs)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoints', 'ema_model_current_state_dict.pth' if args['use_ema'] else 'model_current_state_dict.pth'), map_location=device))
    model.eval()
    freeze_torch_model_params(model)
    model.warmup(horizon=n_support_points, device=device)


    results = []
    print(f"\n🚀 Loading 100 cases...Testing...")

    for trial_id in range(num_trials):
        print(f"\n--- 正在处理 Case {trial_id+1:03d} / {num_trials} ---")
        
        # 1. 生成随机的高密度地图
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42)
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', [])
        
        # 2. 随机采样合法的起终点
        start_np, goal_np, start_t, goal_t = sample_collision_free_start_goal(dynamic_task, tensor_args)
        if start_np is None:
            print(f"Skipping Case {trial_id+1} due to sampling failure.")
            continue
            
        hard_conds = dataset.get_hard_conditions(torch.vstack((start_t, goal_t)), normalize=True)
        
        cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) for f in dynamic_task.get_collision_fields()]
        cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, dt, sigma_gp=1.0, tensor_args=tensor_args)], weights_cost_l=[1e-2]*len(cost_collision_l) + [1e-7], tensor_args=tensor_args)
        guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * 1.5), tensor_args=tensor_args)

        # ====================================================================
        # 推理 Vanilla MPD
        # ====================================================================
        t0 = time.time()
        sample_kwargs_v = dict(guide=guide, n_guide_steps=5, t_start_guide=ceil(0.25 * model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.5)
        trajs_norm_v = model.run_inference(None, hard_conds, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, **sample_kwargs_v, n_diffusion_steps_without_noise=5)
        _, _, trajs_free_v, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(trajs_norm_v)[-1], return_indices=True)
        
        best_v_traj = None
        if trajs_free_v is not None:
            costs_v = compute_path_length(trajs_free_v, robot) + compute_smoothness(trajs_free_v, robot)
            best_idx = torch.argmin(costs_v).item()
            best_v_traj = trajs_free_v[best_idx][..., :2].cpu().numpy()
            # 【新增】：计算 Vanilla MPD 最好轨迹的拓扑势能
            pruned_v = prune_self_intersections(best_v_traj)
            eng_v, _ = evaluate_homotopy_topological_energy(np.array([]), [pruned_v], obs_centers_np)
            topo_energy_v = eng_v[0]
            
            results.append({
                "Method": "MPD", "Trial": trial_id+1, "Success_Rate": 1, 
                "Topo_Energy": topo_energy_v, # <--- 记录能量
                "Path_Length": compute_path_length(trajs_free_v[best_idx].unsqueeze(0), robot).item(), 
                "Smoothness": compute_smoothness(trajs_free_v[best_idx].unsqueeze(0), robot).item(), 
                "Time": time.time() - t0
            })
        else:
            results.append({
                "Method": "MPD", "Trial": trial_id+1, "Success_Rate": 0, 
                "Topo_Energy": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0
            })
        # ====================================================================
        # 推理 TMPD (Ours)
        # ====================================================================
        t0 = time.time()
        sample_kwargs_t = dict(guide=guide, n_guide_steps=10, t_start_guide=ceil(0.1 * model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.8)
        trajs_norm_t = model.run_inference(None, hard_conds, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, **sample_kwargs_t, n_diffusion_steps_without_noise=10)
        _, _, trajs_free_t, _, _ = dynamic_task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(trajs_norm_t)[-1], return_indices=True)
        
        best_t_traj = None
        if trajs_free_t is not None:
            trajs_free_t_np = trajs_free_t[..., :2].cpu().numpy()
            unique_classes, unique_sigs = [], []
            for traj in trajs_free_t_np:
                sig = get_trajectory_signature(traj, obs_centers_np)
                if not any(np.all(np.abs(sig - ext_sig) < 0.3) for ext_sig in unique_sigs):
                    unique_sigs.append(sig)
                    unique_classes.append(prune_self_intersections(traj))
            if len(unique_classes) > 0:
                energies, _ = evaluate_homotopy_topological_energy(np.array([]), unique_classes, obs_centers_np)
                
                scores = []
                for t_np, e in zip(unique_classes, energies):
                    t_tensor = torch.tensor(t_np, **tensor_args).unsqueeze(0)
                    pl = compute_path_length(t_tensor, robot).item()
                    scores.append(e + 2.0 * pl)
                best_idx_t = np.argmin(scores)
                best_t_traj = unique_classes[best_idx_t]
                topo_energy_t = energies[best_idx_t]
                best_t_traj_tensor = torch.tensor(best_t_traj, **tensor_args).unsqueeze(0)
                results.append({
                    "Method": "TMPD (Ours)", "Trial": trial_id+1, "Success_Rate": 1, 
                    "Topo_Energy": topo_energy_t, # <--- 记录能量
                    "Path_Length": compute_path_length(best_t_traj_tensor, robot).item(), 
                    "Smoothness": compute_smoothness(best_t_traj_tensor, robot).item(), 
                    "Time": time.time() - t0
                })
            else:
                results.append({"Method": "TMPD (Ours)", "Trial": trial_id+1, "Success_Rate": 0, "Topo_Energy": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0})
        else:
            results.append({"Method": "TMPD (Ours)", "Trial": trial_id+1, "Success_Rate": 0, "Topo_Energy": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0})

        # ====================================================================
        # 画图与保存
        # ====================================================================
        fig, ax = plt.subplots(figsize=(8, 8))
        
        dynamic_env.render(ax) 
        
        ax.plot(start_np[0], start_np[1], 'gs', markersize=14, markeredgecolor='black', label='Start', zorder=20)
        ax.plot(goal_np[0], goal_np[1], 'ro', markersize=14, markeredgecolor='black', label='Goal', zorder=20)
        
        if best_v_traj is not None:
            ax.plot(best_v_traj[:, 0], best_v_traj[:, 1], color='blue', linewidth=3.0, alpha=0.6, label='MPD', zorder=10)
        
        if best_t_traj is not None:
            ax.plot(best_t_traj[:, 0], best_t_traj[:, 1], color='darkred', linewidth=3.5, label='TMPD (Ours)', zorder=15)

        status_text = ""
        if best_v_traj is None: status_text += "MPD: Collision Failed\n"
        if best_t_traj is None: status_text += "TMPD: Failed"
        if status_text:
            ax.text(0.5, 0.5, status_text, transform=ax.transAxes, ha='center', va='center', fontsize=14, color='black', weight='bold', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.xaxis.set_tick_params(which='both', top=False, bottom=False, labeltop=False, labelbottom=False)
        ax.yaxis.set_tick_params(which='both', left=False, right=False, labelleft=False, labelright=False)
        # ax.legend(loc='lower right', fontsize=12, framealpha=0.9); ax.grid(False, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"overlay_{trial_id+1:03d}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig) 
        
    # ====================================================================
    # 统计最终表格
    # ====================================================================
    df = pd.DataFrame(results)
    df.to_csv(f"final_random_overlay_benchmark_metrics.csv", index=False)
    
    print("\n" + "="*90)
    print("SINGLE-TASK ABLATION BENCHMARK (RANDOM POINTS OVERLAY)")
    print("="*90)
    summary = df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Topo_Energy": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}", # <--- 新增展示
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x)*1000:.3f} ± {np.nanstd(x)*1000:.3f}", 
        "Time": lambda x: f"{np.nanmean(x):.2f}s"
    })
    print(summary.to_string())
    print("="*90)
    print(f"✅ {num_trials} overlay figures saved to '{PLOTS_DIR}/'!")

if __name__ == '__main__':
    run_experiment(run_benchmark_and_plot)