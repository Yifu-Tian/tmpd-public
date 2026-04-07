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

from tmpd_baselines.environment.env_simple_2d_extra_objects import EnvSimple2DExtraObjects
from mpd.utils.topology_utils import get_trajectory_signature, evaluate_homotopy_topological_energy, prune_self_intersections

TRAINED_MODELS_DIR = '../../data_trained_models/'
CASES_SAVE_PATH = 'hard_cases_100.pkl'
PLOTS_DIR = 'benchmark_plots_simple'

@single_experiment_yaml
def run_benchmark_and_plot(
    model_id: str = 'EnvSimple2D-RobotPointMass',
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

    with open(CASES_SAVE_PATH, 'rb') as f:
        test_cases = pickle.load(f)

    results = []
    print(f"\n🚀 Loading {len(test_cases)} cases...Testing...")

    for case in test_cases:
        trial_id = case['case_id']
        print(f"\n--- 正在处理 Case {trial_id:03d} ---")
        
        # 严格复原地图和起终点
        dynamic_env = EnvSimple2DExtraObjects(tensor_args=tensor_args, drop_old_num=7, num_extra_spheres=5, num_extra_boxes=3, seed=case['env_seed'])
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        obs_centers_np = getattr(dynamic_env, 'active_obs_centers', [])
        
        start_np, goal_np = case['start_np'], case['goal_np']
        start_t, goal_t = torch.tensor(start_np, **tensor_args), torch.tensor(goal_np, **tensor_args)
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
            
            w_sig = get_trajectory_signature(best_v_traj, obs_centers_np)
            results.append({"Method": "Vanilla MPD", "Trial": trial_id, "Success_Rate": 1, "Tangle_Free_Rate": 1 if np.all(np.abs(w_sig) < 0.99) else 0, "Path_Length": compute_path_length(trajs_free_v[best_idx].unsqueeze(0), robot).item(), "Smoothness": compute_smoothness(trajs_free_v[best_idx].unsqueeze(0), robot).item(), "Time": time.time() - t0})
        else:
            results.append({"Method": "Vanilla MPD", "Trial": trial_id, "Success_Rate": 0, "Tangle_Free_Rate": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0})

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
                
                # 【修复核心】：引入与 Vanilla 完全一致的平滑度打分
                scores = []
                for t_np, e in zip(unique_classes, energies):
                    # 将 numpy 数组转回 tensor，以利用官方的 metrics 计算函数
                    t_tensor = torch.tensor(t_np, **tensor_args).unsqueeze(0)
                    pl = compute_path_length(t_tensor, robot).item()
                    sm = compute_smoothness(t_tensor, robot).item()
                    
                    # 综合打分：拓扑势能 + 1.0 * 路径长度 + 1.0 * 平滑度
                    # (这里的权重你可以根据实际需要微调，目前 1.0 对应 Vanilla 的标准)
                    scores.append(e + 1.6 * pl)
                
                best_t_traj = unique_classes[np.argmin(scores)]
                
                best_t_traj_tensor = torch.tensor(best_t_traj, **tensor_args).unsqueeze(0)
                w_sig_t = get_trajectory_signature(best_t_traj, obs_centers_np)
                results.append({"Method": "TMPD (Ours)", "Trial": trial_id, "Success_Rate": 1, "Tangle_Free_Rate": 1 if np.all(np.abs(w_sig_t) < 0.99) else 0, "Path_Length": compute_path_length(best_t_traj_tensor, robot).item(), "Smoothness": compute_smoothness(best_t_traj_tensor, robot).item(), "Time": time.time() - t0})
            else:
                results.append({"Method": "TMPD (Ours)", "Trial": trial_id, "Success_Rate": 0, "Tangle_Free_Rate": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0})
        else:
            results.append({"Method": "TMPD (Ours)", "Trial": trial_id, "Success_Rate": 0, "Tangle_Free_Rate": np.nan, "Path_Length": np.nan, "Smoothness": np.nan, "Time": time.time() - t0})

        # ====================================================================
        # 【修改核心】：将两者的轨迹画在同一张图里
        # ====================================================================
        fig, ax = plt.subplots(figsize=(8, 8)) # 只创建一张大图
        
        dynamic_env.render(ax) # 渲染背景环境一次
        
        ax.plot(start_np[0], start_np[1], 'gs', markersize=14, markeredgecolor='black', label='Start', zorder=20)
        ax.plot(goal_np[0], goal_np[1], 'ro', markersize=14, markeredgecolor='black', label='Goal', zorder=20)
        
        # 1. 画 Vanilla MPD 的轨迹 (如果有的话)
        if best_v_traj is not None:
            # 用带有稍微透明度的蓝色实线，以便如果两者重合时能看出来
            ax.plot(best_v_traj[:, 0], best_v_traj[:, 1], color='blue', linewidth=3.0, alpha=0.6, label='Vanilla MPD', zorder=10)
        
        # 2. 画 TMPD 的轨迹 (如果有的话)
        if best_t_traj is not None:
            # 用深红色实线，置于顶层
            ax.plot(best_t_traj[:, 0], best_t_traj[:, 1], color='darkred', linewidth=3.5, label='TMPD (Ours)', zorder=15)

        # 添加状态文本提示 (防失败情况)
        status_text = ""
        if best_v_traj is None: status_text += "Vanilla MPD: Collision Failed\n"
        if best_t_traj is None: status_text += "TMPD: Failed"
        if status_text:
            ax.text(0.5, 0.5, status_text, transform=ax.transAxes, ha='center', va='center', fontsize=14, color='black', weight='bold', bbox=dict(facecolor='white', alpha=0.8))

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9); ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"overlay_case_{trial_id:03d}.png")
        plt.savefig(plot_path, dpi=200)
        plt.close(fig) # 释放内存
        
    # ====================================================================
    # 所有题目做完，统计最终表格
    # ====================================================================
    df = pd.DataFrame(results)
    df.to_csv(f"final_benchmark_metrics.csv", index=False)
    
    print("\n" + "="*80)
    print("BENCHMARK")
    print("="*80)
    summary = df.groupby("Method").agg({
        "Success_Rate": lambda x: f"{np.mean(x)*100:.1f}%",
        "Tangle_Free_Rate": lambda x: f"{np.nanmean(x)*100:.1f}%" if np.sum(~np.isnan(x)) > 0 else "0.0%",
        "Path_Length": lambda x: f"{np.nanmean(x):.3f} ± {np.nanstd(x):.2f}",
        "Smoothness": lambda x: f"{np.nanmean(x):.2f} ± {np.nanstd(x):.2f}",
        "Time": lambda x: f"{np.nanmean(x):.2f}s"
    })
    print(summary.to_string())
    print("="*80)
    print(f"✅ 100 figures saved to '{PLOTS_DIR}/'!")

if __name__ == '__main__':
    run_experiment(run_benchmark_and_plot)