import os
import time
import numpy as np
import pandas as pd
import torch
import math
from math import ceil
import matplotlib
matplotlib.use('Agg') # 开启 Headless 极速画图模式
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
    get_simplest_homotopy_curve,
    is_trajectory_safe
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
RESULTS_DIR = 'benchmark_time_to_success_official_mpd'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots') # 【新增】单独的图片保存文件夹

# ==========================================
# 辅助函数：生成合法的连续测试航点
# ==========================================
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
# 评测主流程 (完全复刻原版 MPD 的挑选逻辑)
# ==========================================
@single_experiment_yaml
def run_benchmark_official_mpd(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    num_segments: int = 5,
    n_samples: int = 50,  # 保持与 TMPD 一致的算力公平对比
    n_diffusion_steps_without_noise: int = 5, 
    start_guide_steps_fraction: float = 0.25, # 原代码默认值
    n_guide_steps: int = 5, # 原代码默认值
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True) # 创建画图文件夹
    
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'🚀 Initializing Benchmark (Official Vanilla MPD ONLY - HEADLESS MODE)...')
    print(f'📁 Plots will be saved to: {PLOTS_DIR}')
    
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
        dynamic_env = EnvDense2DExtraObjects(tensor_args=tensor_args, drop_old_num=2, num_extra_spheres=12, num_extra_boxes=12, seed=42+trial)
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

        # 补回了 failed_traj，用于画撞墙的线
        tracker = {"history": [], "sr": 0, "time": 0.0, "tangled": False, "collision": False, "pl_list": [], "sm_list": [], "final_energy": 0.0, "failed_traj": None}

        for seg in range(num_segments):
            if tracker["collision"]: continue
            
            start_t, goal_t = waypoints_t[seg], waypoints_t[seg+1]
            t0 = time.time()
            best_traj_np = None
            
            attempts = 0
            max_mpd_attempts = 20
            while best_traj_np is None and attempts < max_mpd_attempts:
                attempts += 1
                h_cond = dataset.get_hard_conditions(torch.vstack((start_t, goal_t)), normalize=True)
                
                # 官方配置的采样参数
                t_samples = model.run_inference(None, h_cond, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, guide=guide, n_guide_steps=n_guide_steps, t_start_guide=ceil(start_guide_steps_fraction * model.n_diffusion_steps), noise_std_extra_schedule_fn=lambda x: 0.5, n_diffusion_steps_without_noise=n_diffusion_steps_without_noise)
                
                t_unnorm = dataset.unnormalize_trajectories(t_samples)[-1]
                _, _, trajs_final_free, _, _ = dynamic_task.get_trajs_collision_and_free(t_unnorm, return_indices=True)
                
                if trajs_final_free is not None: 
                    cost_smoothness = compute_smoothness(trajs_final_free, robot)
                    cost_path_length = compute_path_length(trajs_final_free, robot)
                    cost_all = cost_path_length + cost_smoothness
                    idx_best_traj = torch.argmin(cost_all).item()
                    best_traj_np = trajs_final_free[idx_best_traj][..., :2].cpu().numpy()
                else:
                    tracker["failed_traj"] = t_unnorm[0, ..., :2].cpu().numpy() # 记录失败轨迹供可视化
                    print(f"    [Official MPD] Segment {seg+1}: Collision detected. Resampling (Attempt {attempts}/{max_mpd_attempts})...")

            if best_traj_np is None:
                print(f"    [Official MPD] ⚠️ FATAL: Stuck in collision after {max_mpd_attempts} attempts.")
                tracker["collision"] = True

            seg_time = time.time() - t0
            tracker["time"] += seg_time

            if best_traj_np is not None:
                tracker["history"].append(best_traj_np)
                tracker["sr"] += 1
                tracker["pl_list"].append(np.sum(np.linalg.norm(np.diff(best_traj_np, axis=0), axis=1)))
                tracker["sm_list"].append(compute_smoothness(torch.tensor(best_traj_np, **tensor_args).unsqueeze(0), robot).item())
                
                # 评估全局缠绕状态和总能量
                full_path = np.concatenate(tracker["history"])
                taut = get_simplest_homotopy_curve(full_path, obs_centers_np, obs_types, obs_dims)
                check_traj = taut if taut is not None else full_path
                
                sig = get_trajectory_signature(check_traj, obs_centers_np)
                tracker["final_energy"] = np.sum(np.abs(sig))  
                
                if np.any(np.abs(sig) >= 0.95):
                    tracker["tangled"] = True

        # ==========================
        # 【画图逻辑】：生成蓝绿渐变色轨迹图 (带精细箭头)
        # ==========================
        fig, ax = plt.subplots(figsize=(10, 10))
        dynamic_env.render(ax)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect('equal')
        
        if tracker["collision"]:
            title_color = 'red'
            status_txt = f"Collision at Seg {tracker['sr']+1}"
        elif tracker["tangled"]:
            title_color = 'darkorange'
            status_txt = f"Tangled (Energy: {tracker['final_energy']:.2f})"
        else:
            title_color = 'green'
            status_txt = f"Success (Energy: {tracker['final_energy']:.2f})"
            
        ax.set_title(f"Vanilla MPD - Trial {trial+1} [{status_txt}]", fontsize=16, weight='bold', color=title_color)

        # 1. 生成 从蓝到绿 的高级渐变色卡
        cmap = plt.cm.winter
        segment_colors = cmap(np.linspace(0.0, 1.0, max(1, len(waypoints_np)-1)))

        # 2. 画航点
        for k, wp in enumerate(waypoints_np):
            if k == 0:
                ax.plot(wp[0], wp[1], 's', color='green', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1]-0.08, 'S', color='green', ha='center', va='top', weight='bold', zorder=21)
            else:
                ax.plot(wp[0], wp[1], 'o', color='gold', markersize=12, markeredgecolor='black', zorder=20)
                ax.text(wp[0], wp[1], str(k), color='black', ha='center', va='center', weight='bold', zorder=21)

        # 3. 画出渐变色实线轨迹 + 瘦长精细方向箭头
        for seg_idx, traj in enumerate(tracker["history"]):
            if len(traj) < 2: continue
            
            c = segment_colors[seg_idx]
            # 主轨迹
            ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=3.5, alpha=0.85, label=f'Seg {seg_idx+1}')
            
            # 画箭头指示方向 (加在中间偏后的位置，使用调整后的参数)
            mid = int(len(traj) * 0.6)
            if len(traj) > 2:
                dx, dy = traj[mid+1, 0] - traj[mid, 0], traj[mid+1, 1] - traj[mid, 1]
                norm = math.hypot(dx, dy)
                if norm > 0:
                    ax.arrow(traj[mid, 0], traj[mid, 1], dx/norm*0.001, dy/norm*0.001, 
                             shape='full', lw=0, length_includes_head=True, 
                             head_width=0.02, head_length=0.03, color=c, zorder=25)

        # 4. 如果发生了撞墙，画出那条撞墙的红色虚线轨迹
        if tracker["collision"] and tracker["failed_traj"] is not None:
            f_traj = tracker["failed_traj"]
            ax.plot(f_traj[:, 0], f_traj[:, 1], color='red', linewidth=2.5, linestyle='--', alpha=0.9, zorder=18, label='Collision Trajectory')
            ax.plot(f_traj[-1, 0], f_traj[-1, 1], 'rx', markersize=12, markeredgewidth=3, zorder=19)
            # 在终点没到之前死锁了，打个大红叉
            failed_wp = waypoints_np[tracker["sr"] + 1]
            ax.plot(failed_wp[0], failed_wp[1], 'rx', markersize=20, markeredgewidth=4, zorder=25)

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
        plt.savefig(os.path.join(PLOTS_DIR, f"vanilla_mpd_trial_{trial+1:03d}.png"), dpi=150)
        plt.close(fig)

        # ==========================
        # 【严格学术门控逻辑】：质量指标只算全通关，耗时算所有尝试
        # ==========================
        is_fully_successful = (not tracker["tangled"]) and (not tracker["collision"]) and (tracker["sr"] == num_segments)
        attempted_segs = tracker["sr"] + (1 if tracker["collision"] else 0)

        all_results.append({
            "Method": "Official Vanilla MPD", "Trial": trial + 1,
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
    
    print("\n" + "="*95)
    print("🏆 OFFICIAL VANILLA MPD REPORT")
    print("="*95)
    print(summary.to_string())
    print("="*95)

if __name__ == '__main__':
    run_experiment(run_benchmark_official_mpd)