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

from tmpd_baselines.environment.env_simple_2d_extra_objects import EnvSimple2DExtraObjects
from mpd.utils.topology_utils import (
    get_trajectory_signature,
    prune_self_intersections,
    get_simplest_homotopy_curve,
    evaluate_homotopy_topological_energy
)
TRAINED_MODELS_DIR = '../../data_trained_models/'

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
    """
    封装的可视化渲染引擎：支持交互式显隐控制
    """
    ax.clear()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    spine_width = 4.0  # 你可以根据需要调整这个数字，通常 2.0 在论文里很醒目
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    ax.set_title(f"Iteration {iteration_count}: Click to set NEXT GOAL.", fontsize=13, fontweight='bold')
    ax.grid(False, linestyle='--', alpha=0.5)

    # 每次重绘前，清空旧的线条对象缓存
    if current_lines is not None:
        for k in current_lines.keys():
            current_lines[k].clear()
    env.render(ax)
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    
    # 1. 渲染历史轨迹 (绑定 Raw History)
    if len(history_path_list) > 0:
        full_hist_np = np.concatenate(history_path_list, axis=0)
        lines = ax.plot(full_hist_np[:, 0], full_hist_np[:, 1], color='blue', linestyle='-', linewidth=1.6, alpha=0.9, label='Raw History', visible=VIS_STATE['Raw History'])
        current_lines['Raw History'].extend(lines)
        
        initial_pt = history_path_list[0][0]
        ax.plot(initial_pt[0], initial_pt[1], 'bs', markersize=8, markeredgecolor='white', zorder=12)
        
        reached_pts = np.array([traj[-1] for traj in history_path_list])
        ax.plot(reached_pts[:, 0], reached_pts[:, 1], 'bo', markersize=6, markeredgecolor='white', zorder=12)

    # 2. 渲染全局拓扑参考线 (绑定 Global H1)
    if latest_optimized_traj is not None:
        lines = ax.plot(latest_optimized_traj[:, 0], latest_optimized_traj[:, 1], 
                        color='cyan', linestyle='-', linewidth=4.0, alpha=0.4, 
                        solid_capstyle='round', label='Global Taut Cable', 
                        visible=VIS_STATE['Global H1'])
        current_lines['Global H1'].extend(lines)

    # 3. 渲染 MPD 轨迹 (绑定 MPD Raw)
    if latest_best_traj is not None:
        lines = ax.plot(latest_best_traj[:, 0], latest_best_traj[:, 1], 
                        color='blue', linewidth=2.5, label='MPD Candidate', 
                        visible=VIS_STATE['MPD Candidate'])
        current_lines['MPD Candidate'].extend(lines)
        # --- [新增] Waypoint 标注逻辑 ---
        # 遍历轨迹中的每一个点 (g_0 到 g_i)
        for i, pt in enumerate(latest_best_traj):
            # 为了美观，我们不一定每个点都标（如果点太密），可以每隔 n 个点标一个，或者全标
            # 这里设为全标：
            txt = ax.text(pt[0] + 0.02, pt[1] + 0.02, f'$g_{{{i}}}$', 
                          fontsize=9, color='darkred', 
                          fontweight='bold',
                          # 使用 zorder 确保文字在最上层
                          zorder=35,
                          visible=VIS_STATE['MPD Candidate'])
            current_lines['MPD Candidate'].append(txt)
        # [新增] 加上带有能量的专属标签
        if latest_best_energy is not None:
            mid_idx = len(latest_best_traj) // 2
            label_text = "Candidate"
            #  (E:{latest_best_energy:.1f})
            txt = ax.text(latest_best_traj[mid_idx, 0], latest_best_traj[mid_idx, 1], label_text, 
                          color='fuchsia', weight='bold', fontsize=10, 
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), 
                          zorder=30, visible=VIS_STATE['MPD Candidate'])
            current_lines['MPD Candidate'].append(txt)

    if latest_all_trajs is not None:
        for i in range(len(latest_all_trajs)):
            lines = ax.plot(latest_all_trajs[i, :, 0], latest_all_trajs[i, :, 1], 
                            color='red', linestyle='-', linewidth=1.0, alpha=0.2, 
                            visible=VIS_STATE['All MPD'])
            current_lines['All MPD'].extend(lines)

    # 4. 渲染所有备选拓扑分支 (绑定 Topo Branches)
    if unique_homotopy_classes is not None and homotopy_energies is not None:
        colors = ['magenta', 'orange', 'lime', 'cyan', 'yellow', 'pink']
        for i, (traj, energy) in enumerate(zip(unique_homotopy_classes, homotopy_energies)):
            c = colors[i % len(colors)]
            if i == 0:
                vis_key = 'Ref Topo'
                lw = 3.0       # 最优解画粗一点
                ls = '--'
            else:
                vis_key = 'All Topo'
                lw = 1.8       # 备选解画细一点
                ls = ':'       # 备选解用密集虚线，弱化视觉存在感
            lines = ax.plot(traj[:, 0], traj[:, 1], color=c, linestyle='--', linewidth=2.0, alpha=0.9, solid_capstyle='round', visible=VIS_STATE[vis_key])
            current_lines[vis_key].extend(lines)
            h_id = homotopy_indices[i] if homotopy_indices is not None else i + 1
            label_text = f"H{h_id} (E:{energy:.1f})"
            mid_idx = len(traj) // 2
            txt = ax.text(traj[mid_idx, 0], traj[mid_idx, 1], label_text, color=c, weight='bold', fontsize=10, bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'), zorder=25, visible=VIS_STATE[vis_key])
            current_lines[vis_key].append(txt)

    ax.plot(current_start_pos[0].item(), current_start_pos[1].item(), 'go', markersize=12, label='Current Pos')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax.legend(by_label.values(), by_label.keys(), loc='lower right', framealpha=0.9)
    
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

@single_experiment_yaml
def experiment(
    model_id: str = 'EnvSimple2D-RobotPointMass',
    planner_alg: str = 'mpd',
    use_guide_on_extra_objects_only: bool = False,
    n_samples: int = 50, # 采样数量
    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 10,
    n_diffusion_steps_without_noise: int = 20, # 无噪声引导时间步
    weight_grad_cost_collision: float = 1e-2,
    weight_grad_cost_smoothness: float = 1e-7,
    factor_num_interpolated_points_for_collision: float = 1.5,
    trajectory_duration: float = 5.0,  # currently fixed
    device: str = 'cuda',
    debug: bool = True,
    render: bool = False, # 强制关闭多余渲染
    seed: int = 30,
    results_dir: str = 'logs',
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print(f'##########################################################################################################')
    print(f'Model -- {model_id} | Algorithm -- {planner_alg}')
    
    # run_prior_only = (planner_alg == 'diffusion_prior')
    # run_prior_then_guidance = (planner_alg == 'diffusion_prior_then_guide')

    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(model_dir, 'results_inference', str(seed))
    os.makedirs(results_dir, exist_ok=True)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    # Load dataset & model
    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05,
        **args, tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    env = dataset.env
    robot = dataset.robot
    base_task = dataset.task
    env = EnvSimple2DExtraObjects(
        tensor_args=tensor_args, 
        drop_old_num=0, 
        num_extra_spheres=0, 
        num_extra_boxes=0, 
        seed=seed
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

    # =====================================================================
    # 固定初始起点
    # =====================================================================
    q_dim = getattr(env, 'q_dim', 2)
    start_state_pos = torch.zeros(q_dim, **tensor_args)
    custom_start_coords = [0, -0.25]

    start_state_pos[:2] = torch.tensor(custom_start_coords, **tensor_args)
    
    # =======================================================================================
    # 【统一交互架构】：动态界面与拓扑势场渲染闭环
    # =======================================================================================
    history_path_list = []  
    current_start_pos = start_state_pos.clone()
    
    # 用于记录和显示最新生成的轨迹
    latest_all_trajs = None
    latest_best_traj = None
    latest_optimized_traj = None
    latest_best_energy = None

    print("\nInitialization finished, model loaded...")
    iteration_count = 0

    plt.ion()  
    fig_topo, ax_topo = plt.subplots(figsize=(13, 11))
    fig_topo.canvas.manager.set_window_title("MPD Interactive Topology Lab")
    
    # 主地图区 (右侧)
    ax_topo = fig_topo.add_axes([0.22, 0.05, 0.75, 0.9])
    # 按钮控制区 (左侧)
    ax_check = fig_topo.add_axes([0.02, 0.45, 0.16, 0.25])
    ax_check.set_title("Display Control", weight='bold')
    ax_check.set_xticks([])  # 删掉 X 轴刻度和数字
    ax_check.set_yticks([])  # 删掉 Y 轴刻度和数字
    for spine in ax_check.spines.values():
        spine.set_visible(False)  # 删掉外围的黑色边框，让它彻底变成一个悬浮菜单
    # 2. 定义可视化状态与线条容器
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

    # 3. 按钮点击回调函数
    def toggle_vis(label):
        VIS_STATE[label] = not VIS_STATE[label]
        # 实时更新所有关联对象的可见性
        for line_obj in current_lines[label]:
            if line_obj is not None:
                line_obj.set_visible(VIS_STATE[label])
        fig_topo.canvas.draw_idle()

    check_buttons.on_clicked(toggle_vis)
    plt.show(block=False) 

    current_homotopy_classes = None
    current_homotopy_energies = None
    current_homotopy_indices = None

    while True:
        iteration_count += 1
        print(f"\n====================== [ITERATION-{iteration_count} ] ======================")
        
        render_demo(fig_topo, ax_topo, iteration_count, env, history_path_list, latest_all_trajs, latest_best_traj, current_start_pos,
                    latest_optimized_traj, current_homotopy_classes, current_homotopy_energies, current_homotopy_indices,
                    VIS_STATE, current_lines, latest_best_energy)
        print("waiting for the next goal...")
        
    
        pts = []
        def onclick(event):
            # 只有当鼠标点在右侧的主地图 (ax_topo) 内时，才视为设定了新目标
            if event.inaxes == ax_topo:
                pts.append((event.xdata, event.ydata))
                fig_topo.canvas.stop_event_loop()
            # 如果点在了 ax_check，什么都不做，放行给 CheckButtons 自己的回调

        def onclose(event):
            fig_topo.canvas.stop_event_loop()

        cid_click = fig_topo.canvas.mpl_connect('button_press_event', onclick)
        cid_close = fig_topo.canvas.mpl_connect('close_event', onclose)

        # 挂起程序，直到 stop_event_loop 被调用
        fig_topo.canvas.start_event_loop(timeout=0) 

        # 退出循环后解绑事件
        fig_topo.canvas.mpl_disconnect(cid_click)
        fig_topo.canvas.mpl_disconnect(cid_close)

        if len(pts) == 0 or not plt.fignum_exists(fig_topo.number):
            print("\nTrajectories Planning Finished")
            break

        # 获取新终点，执行 MPD 推理
        target_x, target_y = pts[0]
        print(f"Next Goal: ({target_x:.3f}, {target_y:.3f}), Motion Planning Diffusion Running...")
        
        goal_state_pos = torch.zeros_like(current_start_pos)
        goal_state_pos[0], goal_state_pos[1] = target_x, target_y
        start_state_pos = current_start_pos.clone()
        
        hard_conds = dataset.get_hard_conditions(torch.vstack((start_state_pos, goal_state_pos)), normalize=True)
        
        # ====================================================================
        # 物理状态估计：构建历史评估线缆 (Taut-cable footprint C_past)
        # ====================================================================
        hist_for_eval = np.concatenate(history_path_list, axis=0) if history_path_list else np.array([])
        if len(hist_for_eval) > 0 and len(obs_centers_np) > 0:
            refined_hist = get_simplest_homotopy_curve(hist_for_eval, obs_centers_np, obs_types, obs_dims)
            # 显式判断：只有当成功返回了收缩后的数组时，才覆盖原变量
            if refined_hist is not None:
                hist_for_eval = refined_hist
        # ====================================================================
        # STAGE 1: 纯生成器 (High-Temperature Delayed-Guidance Diffusion)
        # 彻底去除了前置启发式引导，完全依赖模型自身的多模态发散
        # ====================================================================
        
        cost_collision_l = [CostCollision(robot, n_support_points, field=f, sigma_coll=1.0, tensor_args=tensor_args) 
                            for f in (task.get_collision_fields_extra_objects() if use_guide_on_extra_objects_only else task.get_collision_fields())]
        cost_composite = CostComposite(robot, n_support_points, [*cost_collision_l, CostGPTrajectory(robot, n_support_points, trajectory_duration/n_support_points, sigma_gp=1.0, tensor_args=tensor_args)], 
                                       weights_cost_l=[weight_grad_cost_collision]*len(cost_collision_l) + [weight_grad_cost_smoothness], tensor_args=tensor_args)

        guide = GuideManagerTrajectoriesWithVelocity(dataset, cost_composite, clip_grad=True, interpolate_trajectories_for_collision=True, num_interpolated_points=ceil(n_support_points * factor_num_interpolated_points_for_collision), tensor_args=tensor_args)
        # 延迟引导触发点 (例如倒数20步才介入)
        t_start_guide = ceil(start_guide_steps_fraction * model.n_diffusion_steps)
        sample_fn_kwargs = dict(guide=guide, n_guide_steps=n_guide_steps, t_start_guide=t_start_guide, noise_std_extra_schedule_fn=lambda x: 1.8)

        with TimerCUDA() as timer_model_sampling:
            trajs_normalized_iters = model.run_inference(None, hard_conds, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, **sample_fn_kwargs, n_diffusion_steps_without_noise=n_diffusion_steps_without_noise)
        
        _, _, trajs_final_free, _, _ = task.get_trajs_collision_and_free(dataset.unnormalize_trajectories(trajs_normalized_iters)[-1], return_indices=True)

        # ====================================================================
        # STAGE 2: 无监督拓扑后验过滤 (Unsupervised Topological Posterior Filter)
        # 从扩散模型的轨迹束中提取同伦类
        # ====================================================================
        if trajs_final_free is not None:
            trajs_free_np = trajs_final_free[..., :2].cpu().numpy()
            unique_mpd_classes, unique_sigs = [], []
            
            # 拓扑聚类：依靠包角签名将 MPD 轨迹束归类
            for traj in trajs_free_np:
                sig = get_trajectory_signature(traj, obs_centers_np)
                # 容差设为 0.3，吸收属于同一个通道内的平滑度波动
                if not any(np.all(np.abs(sig - ext_sig) < 0.3) for ext_sig in unique_sigs):
                    unique_sigs.append(sig)
                    unique_mpd_classes.append(prune_self_intersections(traj))
                    
            # 能量评估与帕累托最优排序
            mpd_energies, _ = evaluate_homotopy_topological_energy(hist_for_eval, unique_mpd_classes, obs_centers_np, w_max=0.8)
            combined_scores = []
            for traj, energy in zip(unique_mpd_classes, mpd_energies):
                # 1. 拼接当前候选与历史轨迹
                if len(hist_for_eval) > 0:
                    combined_raw = np.vstack((hist_for_eval, traj[1:]))
                else:
                    combined_raw = traj
                    
                # 2. 物理拉紧以获取最真实的全局同伦状态
                # 注意：确保你的脚本前面获取了 obs_types 和 obs_dims，如果没有，可以用默认值
                refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                check_traj = refined_combined if refined_combined is not None else combined_raw
                
                # 3. 计算全局签名
                global_sig = get_trajectory_signature(check_traj, obs_centers_np)
                
                # 4. 熔断判定：只要在任何一个障碍物上绕圈 >= 1 (0.95容差)，直接宣判死刑
                if np.any(np.abs(global_sig) >= 0.95):
                    tangle_penalty = 10000.0  # 极高惩罚，确保打结轨迹绝对沉底
                else:
                    tangle_penalty = 0.0
                
                # 综合打分：拓扑势能 + 长度惩罚 + 缠绕熔断惩罚
                score = energy + 0.8 * np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) + tangle_penalty
                combined_scores.append(score)
            
            sorted_data = sorted(zip(range(1, len(unique_mpd_classes) + 1), unique_mpd_classes, mpd_energies, combined_scores), key=lambda x: x[3])
            
            current_homotopy_indices = [x[0] for x in sorted_data]
            current_homotopy_classes = [x[1] for x in sorted_data]
            current_homotopy_energies = [x[2] for x in sorted_data]
            
            latest_best_traj = current_homotopy_classes[0]
            latest_best_energy = current_homotopy_energies[0]
            
            print(f"🔥 Generation Finished!  {len(current_homotopy_classes)} Topology found in total, select H{current_homotopy_indices[0]} (Energy: {latest_best_energy:.2f})")
            # ====================================================================
            # [新增] 实时计算并打印 Metrics 评估面板
            # ====================================================================

            # 1. 轨迹长度 (Path Length)
            path_length = np.sum(np.linalg.norm(np.diff(latest_best_traj, axis=0), axis=1))

            # 2. 运动学平滑度 (Kinematic Smoothness) 
            # 使用离散加速度平方的积分来严谨逼近: Integral( ||a(t)||^2 dt )
            if len(latest_best_traj) >= 3:
                # dt 在前面已经定义了: dt = trajectory_duration / n_support_points
                accels = np.diff(latest_best_traj, n=2, axis=0) / (dt ** 2)
                smoothness = np.sum(np.linalg.norm(accels, axis=1)**2) * dt
            else:
                smoothness = 0.0

            print("\n" + "="*55)
            print("📊 [ 轨迹生成评估报告 | METRICS REPORT ]")
            print("="*55)
            print(f"📏 轨迹总长度 (Length)    : {path_length:.3f} m")
            print(f"🌊 运动学平滑度 (Smoothness): {smoothness:.2f} ")
            print(f"🪢 拓扑势能 (Topo Energy) : {latest_best_energy:.3f}")
            print("="*55 + "\n")

            # ====================================================================
            # [新增] 计算全局物理拉紧线 (Global Taut Cable) 用于视觉基准对比
            # ====================================================================
            if len(hist_for_eval) > 0:
                combined_raw_traj = np.vstack((hist_for_eval, latest_best_traj[1:]))
            else:
                combined_raw_traj = latest_best_traj
                
            refined_global = get_simplest_homotopy_curve(combined_raw_traj, obs_centers_np, obs_types, obs_dims)
            latest_optimized_traj = refined_global if refined_global is not None else combined_raw_traj

            # 状态更迭
            history_path_list.append(latest_best_traj)
            current_start_pos[:2] = torch.tensor(latest_best_traj[-1], dtype=torch.float32, device=device)
            latest_all_trajs = trajs_free_np
        else:
            print("All collision!!!")
            latest_all_trajs, latest_best_traj, latest_optimized_traj, latest_best_energy = None, None, None, None

if __name__ == '__main__':
    run_experiment(experiment)