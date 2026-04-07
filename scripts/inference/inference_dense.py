import os
import time
import pickle
from math import ceil
from pathlib import Path
import numpy as np
import einops
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
    evaluate_homotopy_topological_energy
)

TRAINED_MODELS_DIR = '../../data_trained_models/'
# ==========================================================
# 【新增】：轨迹后处理工具函数
# ==========================================================
# ==========================================================
# 【新增】：带碰撞保护的轨迹后处理工具函数
# ==========================================================
# ==========================================================
# 【新增】：带碰撞保护的轨迹后处理工具函数 (修复标量转换 Bug)
# ==========================================================
def apply_safe_post_processing(traj_np, task, tensor_args, n_points, max_iters=5):
    """
    碰撞感知的安全平滑器：
    一步一步收紧轨迹，一旦检测到“切角撞墙”，立刻回退到上一个安全状态并终止。
    """
    current_safe_traj = np.copy(traj_np)
    
    # 1. 安全的拉普拉斯平滑迭代
    for i in range(max_iters):
        smoothed = np.copy(current_safe_traj)
        # 核心公式: 当前点 = 自身占50% + 左右邻居各占25% (起终点绝对不动)
        smoothed[1:-1] = 0.5 * smoothed[1:-1] + 0.25 * smoothed[:-2] + 0.25 * smoothed[2:]
        
        # 碰撞检测：如果这一步把轨迹拉进了障碍物，直接熔断！
        smoothed_t = torch.tensor(smoothed, **tensor_args).unsqueeze(0)
        
        # 【修复点 1】：加入 .sum()，只要整条轨迹碰撞代价之和 > 0，即视为碰撞
        if task.compute_collision(smoothed_t).sum().item() > 0:
            # print(f"      [Smooth Guard] Corner cut prevented at iteration {i}. Reverting.")
            break
        else:
            current_safe_traj = smoothed

    # 2. 安全的等距重采样
    def resample(traj, pts):
        if len(traj) < 2: return np.zeros((pts, traj.shape[1]))
        diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(diffs), 0, 0)
        if cum_dists[-1] == 0: return np.tile(traj[0], (pts, 1))
        resampled = np.zeros((pts, traj.shape[1]))
        for dim in range(traj.shape[1]): 
            resampled[:, dim] = np.interp(np.linspace(0, cum_dists[-1], pts), cum_dists, traj[:, dim])
        return resampled

    final_resampled = resample(current_safe_traj, n_points)
    
    # 终极保险：重采样本身也可能因为截弯取直导致撞墙，如果撞了，就用回重采样前的平滑版本
    final_resampled_t = torch.tensor(final_resampled, **tensor_args).unsqueeze(0)
    
    # 【修复点 2】：同样加入 .sum() 
    if task.compute_collision(final_resampled_t).sum().item() > 0:
        # print("      [Resample Guard] Resampling caused collision. Using un-resampled points.")
        return current_safe_traj
        
    return final_resampled
# ==========================================================
# ==========================================================
def render_demo(
    fig, ax, iteration_count, env, history_path_list, 
    latest_all_trajs, latest_best_traj, current_start_pos,
    latest_optimized_traj=None,
    unique_homotopy_classes=None,
    homotopy_energies=None,
    homotopy_indices=None,
    VIS_STATE=None,      
    current_lines=None,   
    latest_best_energy=None
):
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_title(f"Iteration {iteration_count}: Click to set NEXT GOAL.", fontsize=13, fontweight='bold')

    ax.grid(False, linestyle='--', alpha=0.5)
    spine_width = 3.0  # 设置你需要的粗细
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    
    ax.tick_params(axis='both', which='major', labelsize=10, width=spine_width)

    if current_lines is not None:
        for k in current_lines.keys():
            current_lines[k].clear()

    env.render(ax)
    
    # 依然为你保留原有的 O1, O2 文本编号，方便你调试拓扑签名
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    for k, center in enumerate(obs_centers_np):
        ax.text(center[0], center[1], f"O{k+1}", color='white', ha='center', va='center', weight='bold', fontsize=9, zorder=10)

    # 1. 渲染历史轨迹
    if len(history_path_list) > 0:
        full_hist_np = np.concatenate(history_path_list, axis=0)
        lines = ax.plot(full_hist_np[:, 0], full_hist_np[:, 1], color='blue', linestyle='-', linewidth=1.6, alpha=0.9, label='Raw History', visible=VIS_STATE['Raw History'])
        current_lines['Raw History'].extend(lines)
        
        initial_pt = history_path_list[0][0]
        ax.plot(initial_pt[0], initial_pt[1], 'bs', markersize=8, markeredgecolor='white', zorder=12)
        
        reached_pts = np.array([traj[-1] for traj in history_path_list])
        ax.plot(reached_pts[:, 0], reached_pts[:, 1], 'bo', markersize=6, markeredgecolor='white', zorder=12)

    # 2. 渲染全局拓扑参考线
    if latest_optimized_traj is not None:
        lines = ax.plot(latest_optimized_traj[:, 0], latest_optimized_traj[:, 1], 
                        color='cyan', linestyle='-', linewidth=4.0, alpha=0.4, 
                        solid_capstyle='round', label='Global Taut Cable', 
                        visible=VIS_STATE['Global H1'])
        current_lines['Global H1'].extend(lines)

    # 3. 渲染 MPD 轨迹
    if latest_best_traj is not None:
        lines = ax.plot(latest_best_traj[:, 0], latest_best_traj[:, 1], 
                        color='darkred', linewidth=2.5, label='MPD Candidate', 
                        visible=VIS_STATE['MPD Candidate'])
        current_lines['MPD Candidate'].extend(lines)
        
        if latest_best_energy is not None:
            mid_idx = len(latest_best_traj) // 2
            label_text = f"MPD_Best (E:{latest_best_energy:.1f})"
            txt = ax.text(latest_best_traj[mid_idx, 0], latest_best_traj[mid_idx, 1], label_text, 
                          color='fuchsia', weight='bold', fontsize=10, 
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), 
                          zorder=30, visible=VIS_STATE['MPD Candidate'])
            current_lines['MPD Candidate'].append(txt)

    if latest_all_trajs is not None:
        for i in range(len(latest_all_trajs)):
            lines = ax.plot(latest_all_trajs[i, :, 0], latest_all_trajs[i, :, 1], 
                            color='red', linestyle='-', linewidth=1.0, alpha=0.25, 
                            visible=VIS_STATE['All MPD'])
            current_lines['All MPD'].extend(lines)

    # 4. 渲染所有备选拓扑分支
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
    
    # handles, labels = ax.get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

@single_experiment_yaml
def experiment(
    model_id: str = 'EnvDense2D-RobotPointMass',
    planner_alg: str = 'mpd',
    use_guide_on_extra_objects_only: bool = False,
    n_samples: int = 70, 
    start_guide_steps_fraction: float = 0.1,
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
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'##########################################################################################################')
    print(f'Model -- {model_id} | Algorithm -- {planner_alg}')
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, 'results_inference', str(seed))
    save_data_dir = os.path.join(results_dir, 'trajectory_data')
    os.makedirs(results_dir, exist_ok=True)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    # 1. 挂载 Dataset 获取底层结构
    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05,
        **args, tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    robot = dataset.robot
    base_task = dataset.task # 先暂存基础 Task 类

    # ==========================================================
    # 【核心修复】：强行注入动态密林环境 (EnvDense2DExtraObjects)
    # ==========================================================
    env = EnvDense2DExtraObjects(
        tensor_args=tensor_args, 
        drop_old_num=2, 
        num_extra_spheres=24, 
        num_extra_boxes=0, 
        seed=42
    )
    # 基于新环境构建具备正确碰撞场（SDF）的 Task
    task = type(base_task)(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
    
    # 覆盖 dataset 的 env 和 task，防止扩散模型和评价器引用错地图
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

    # 固定初始起点
    q_dim = getattr(env, 'q_dim', 2)
    start_state_pos = torch.zeros(q_dim, **tensor_args)
    # 为了避免写死的坐标正好砸在随机生成的新方块上，如果出现开局报错可以微调这个点
    custom_start_coords = [-0.9, -0.9]
    start_state_pos[:2] = torch.tensor(custom_start_coords, **tensor_args)
    
    # ... 后续交互及推理逻辑无需改动 ...
    history_path_list = []  
    current_start_pos = start_state_pos.clone()
    
    latest_all_trajs = None
    latest_best_traj = None
    latest_optimized_traj = None
    latest_best_energy = None

    print("\nInitialization finished, model loaded...")
    iteration_count = 0

    plt.ion()  
    fig_topo, ax_topo = plt.subplots(figsize=(13, 11))
    fig_topo.canvas.manager.set_window_title("MPD Interactive Topology Lab")
    
    ax_topo = fig_topo.add_axes([0.22, 0.05, 0.75, 0.9])
    ax_check = fig_topo.add_axes([0.02, 0.45, 0.16, 0.25])
    ax_check.set_title("Display Control", weight='bold')
    ax_check.set_xticks([])  
    ax_check.set_yticks([])  
    for spine in ax_check.spines.values():
        spine.set_visible(False)  
        
    VIS_STATE = {
        'Raw History': True,
        'Global H1': True,
        'MPD Candidate': True,
        'All MPD': True,
        'Ref Topo': True,
        'All Topo': False
    }
    current_lines = {k: [] for k in VIS_STATE.keys()}

    labels = list(VIS_STATE.keys())
    actives = list(VIS_STATE.values())
    check_buttons = CheckButtons(ax_check, labels, actives)

    def toggle_vis(label):
        VIS_STATE[label] = not VIS_STATE[label]
        for line_obj in current_lines[label]:
            if line_obj is not None:
                line_obj.set_visible(VIS_STATE[label])
        fig_topo.canvas.draw_idle()

    check_buttons.on_clicked(toggle_vis)
    plt.show(block=False) 

    current_homotopy_classes = None
    current_homotopy_energies = None
    current_homotopy_indices = None
    all_iterations_data = [] # 【新增】用于存放所有步数的轨迹容器
    
    # 定义唯一的保存路径
    combined_save_path = os.path.join(results_dir, f'full_session_data_seed_{seed}_2.pkl')
    while True:
        iteration_count += 1
        print(f"\n====================== [ITERATION-{iteration_count} ] ======================")
        
        render_demo(fig_topo, ax_topo, iteration_count, env, history_path_list, latest_all_trajs, latest_best_traj, current_start_pos,
                    latest_optimized_traj, current_homotopy_classes, current_homotopy_energies, current_homotopy_indices,
                    VIS_STATE, current_lines, latest_best_energy)
        print("waiting for the next goal...")
        
        ax_topo.set_title(f"Iteration {iteration_count}: Click to set NEXT GOAL.", fontsize=13, fontweight='bold')
        pts = []
        def onclick(event):
            if event.inaxes == ax_topo:
                pts.append((event.xdata, event.ydata))
                fig_topo.canvas.stop_event_loop()

        def onclose(event):
            fig_topo.canvas.stop_event_loop()

        cid_click = fig_topo.canvas.mpl_connect('button_press_event', onclick)
        cid_close = fig_topo.canvas.mpl_connect('close_event', onclose)

        fig_topo.canvas.start_event_loop(timeout=0) 

        fig_topo.canvas.mpl_disconnect(cid_click)
        fig_topo.canvas.mpl_disconnect(cid_close)

        if len(pts) == 0 or not plt.fignum_exists(fig_topo.number):
            print("\nTrajectories Planning Finished")
            break

        target_x, target_y = pts[0]
        print(f"Next Goal: ({target_x:.3f}, {target_y:.3f}), Motion Planning Diffusion Running...")
        
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
                # 1. 拼接当前候选与历史轨迹
                if len(hist_for_eval) > 0:
                    combined_raw = np.vstack((hist_for_eval, traj[1:]))
                else:
                    combined_raw = traj
                    
                # 2. 物理拉紧以获取最真实的全局同伦状态
                refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                check_traj = refined_combined if refined_combined is not None else combined_raw
                
                # 3. 计算全局签名
                global_sig = get_trajectory_signature(check_traj, obs_centers_np)
                print("global sig: ", global_sig)
                
                # 综合打分：拓扑势能 + 长度惩罚
                score = energy + 0.8 * np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                combined_scores.append(score)
            
            sorted_data = sorted(zip(range(1, len(unique_mpd_classes) + 1), unique_mpd_classes, mpd_energies, combined_scores), key=lambda x: x[3])
            
            current_homotopy_indices = [x[0] for x in sorted_data]
            current_homotopy_classes = [x[1] for x in sorted_data]
            current_homotopy_energies = [x[2] for x in sorted_data]
            
            # 对前3名轨迹分别执行平滑与重采样，保证它们都是高质量的 64 点轨迹

            latest_best_traj = current_homotopy_classes[0]
            latest_best_energy = current_homotopy_energies[0]
            # 1. 空间平滑
            latest_best_traj = apply_safe_post_processing(latest_best_traj, task, tensor_args, n_support_points, max_iters=5)

            print(f"🔥 Generation Finished!  {len(current_homotopy_classes)} Topology found in total, select H{current_homotopy_indices[0]} (Energy: {latest_best_energy:.2f})")
            key_steps_for_paper = [4] # 只有第 4步
            is_key_step = (iteration_count in key_steps_for_paper)
            
            save_candidates = [] # 默认空列表
            
            if is_key_step and len(trajs_free_np) > 1:
                # 【修改点】：直接从扩散模型生成的“所有无碰撞轨迹池”中提取前 10 条
                # 这将展示出 Diffusion Model 极具特征的“不确定性探索簇”
                for traj in trajs_free_np[:10]: 
                    
                    # 简单判断一下，如果这条原始轨迹碰巧跟你的最优轨迹一模一样，就跳过不画
                    if np.allclose(traj, latest_best_traj, atol=1e-3):
                        continue
                    tmp_traj = apply_safe_post_processing(traj, task, tensor_args, n_support_points, max_iters=5)
                    save_candidates.append(tmp_traj.copy())
            path_length = np.sum(np.linalg.norm(np.diff(latest_best_traj, axis=0), axis=1))

            sm_tensor = torch.tensor(latest_best_traj, **tensor_args).unsqueeze(0)
            smoothness = compute_smoothness(sm_tensor, robot).item()

            print("\n" + "="*55)
            print("📊 [ 轨迹生成评估报告 | METRICS REPORT ]")
            print("="*55)
            print(f"📏 轨迹总长度 (Length)    : {path_length:.3f} m")
            print(f"🌊 运动学平滑度 (Smoothness): {smoothness:.2f} ")
            print(f"🪢 拓扑势能 (Topo Energy) : {latest_best_energy:.3f}")
            print("="*55 + "\n")

            if len(hist_for_eval) > 0:
                combined_raw_traj = np.vstack((hist_for_eval, latest_best_traj[1:]))
            else:
                combined_raw_traj = latest_best_traj
                
            refined_global = get_simplest_homotopy_curve(combined_raw_traj, obs_centers_np, obs_types, obs_dims)
            latest_optimized_traj = refined_global if refined_global is not None else combined_raw_traj

            history_path_list.append(latest_best_traj)
            current_start_pos[:2] = torch.tensor(latest_best_traj[-1], dtype=torch.float32, device=device)
            latest_all_trajs = trajs_free_np
            # ==========================================================
            # 【集中保存版】将当前步数据追加到总列表，并保存至单一文件
            # ==========================================================
            current_step_data = {
                'iteration': iteration_count,
                'waypoints': latest_best_traj.copy(), # 使用 copy 确保数据一致性
                'taut_cable_global': latest_optimized_traj.copy() if latest_optimized_traj is not None else None,
                'top_3_candidates': save_candidates,
                'metrics': {
                    'length': path_length,
                    'smoothness': smoothness,
                    'energy': latest_best_energy
                }
            }
            
            # 将当前步加入总表
            all_iterations_data.append(current_step_data)
            
            # 构造最终保存的字典（包含环境信息，因为环境是静态的，存一份即可）
            final_save_payload = {
                'metadata': {
                    'seed': seed,
                    'model_id': model_id,
                    'obstacles': {
                        'centers': obs_centers_np,
                        'types': obs_types,
                        'dims': obs_dims
                    }
                },
                'steps': all_iterations_data # 所有的迭代轨迹都在这里
            }
            
            # 每次迭代都覆盖保存，确保即便仿真中途断掉，数据也是全的
            with open(combined_save_path, 'wb') as f:
                pickle.dump(final_save_payload, f)
            
            print(f"💾 Full session data updated at: {combined_save_path}")
        else:
            print("All collision!!!")
            latest_all_trajs, latest_best_traj, latest_optimized_traj, latest_best_energy = None, None, None, None

if __name__ == '__main__':
    run_experiment(experiment)