import os
import pickle
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
from matplotlib.widgets import CheckButtons

from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness


from mpd.environments.env_dense_2d_extra_objects import EnvDense2DExtraObjects

from mpd.utils.topology_utils import (
    get_trajectory_signature,
    prune_self_intersections,
    get_simplest_homotopy_curve,
    evaluate_homotopy_topological_energy
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "data_trained_models")
INFERENCE_RESULTS_DIR = os.path.join(RESULTS_ROOT, "inference")


def parse_auto_goals(auto_goals: str):
    """Parse auto goals from 'x1,y1;x2,y2;...'."""
    if auto_goals is None:
        return []
    text = auto_goals.strip()
    if not text:
        return []

    goals = []
    for token in text.split(';'):
        token = token.strip()
        if not token:
            continue
        xy = [v.strip() for v in token.split(',')]
        if len(xy) != 2:
            raise ValueError(f"Invalid auto goal token '{token}', expected 'x,y'.")
        goals.append((float(xy[0]), float(xy[1])))
    return goals


def parse_polyline(polyline_str: str):
    """Parse a 2D polyline from 'x1,y1;x2,y2;...'."""
    if polyline_str is None:
        return None
    text = polyline_str.strip()
    if not text:
        return None

    pts = []
    for token in text.split(';'):
        token = token.strip()
        if not token:
            continue
        xy = [v.strip() for v in token.split(',')]
        if len(xy) != 2:
            raise ValueError(f"Invalid history point '{token}', expected 'x,y'.")
        pts.append([float(xy[0]), float(xy[1])])

    if len(pts) < 2:
        raise ValueError("fixed_history must contain at least two points.")
    return np.asarray(pts, dtype=np.float32)


def save_denoise_animation_multi(
    env,
    candidate_chains: list,
    candidate_finals: list,
    candidate_labels: list,
    candidate_colors: list,
    best_candidate_idx: int,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    output_path: str,
    history_xy: np.ndarray = None,
    fps: int = 10,
    stride: int = 2,
):
    """
    Save denoising animation for multiple feasible candidates.
    Returns the final saved path.
    """
    if len(candidate_chains) == 0:
        return None

    stride = max(1, int(stride))
    fps = max(1, int(fps))
    n_steps = candidate_chains[0].shape[0]
    frame_indices = list(range(0, n_steps, stride))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)

    chain_frames = [chain[frame_indices] for chain in candidate_chains]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    def draw(frame_id):
        ax.clear()
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(False)
        env.render(ax)

        if history_xy is not None and len(history_xy) > 1:
            ax.plot(
                history_xy[:, 0], history_xy[:, 1],
                color='black', linewidth=3.2, alpha=0.85, label='Fixed History'
            )

        for i, frames in enumerate(chain_frames):
            lw = 2.8 if i == best_candidate_idx else 1.7
            alpha = 0.95 if i == best_candidate_idx else 0.55
            style = '-' if i == best_candidate_idx else '--'
            current = frames[frame_id]
            ax.plot(
                current[:, 0], current[:, 1],
                color=candidate_colors[i], linewidth=lw, alpha=alpha, linestyle=style,
                label=f"{candidate_labels[i]} Denoising"
            )

        ax.plot(start_xy[0], start_xy[1], 'go', markersize=9, label='Start')
        ax.plot(goal_xy[0], goal_xy[1], 'rx', markersize=10, markeredgewidth=2.4, label='Goal')

        if frame_id == len(frame_indices) - 1:
            for i, final_traj in enumerate(candidate_finals):
                if final_traj is None:
                    continue
                if i == best_candidate_idx:
                    # Glow effect for selected trajectory.
                    ax.plot(
                        final_traj[:, 0], final_traj[:, 1],
                        color='gold', linewidth=8.0, alpha=0.35
                    )
                    ax.plot(
                        final_traj[:, 0], final_traj[:, 1],
                        color='orangered', linewidth=3.0, alpha=1.0,
                        label='Selected Trajectory'
                    )
                else:
                    ax.plot(
                        final_traj[:, 0], final_traj[:, 1],
                        color=candidate_colors[i], linewidth=2.0, alpha=0.8, linestyle=':'
                    )

        ax.set_title(f"Denoising {frame_id + 1}/{len(frame_indices)}")
        ax.legend(loc='upper right', framealpha=0.9, fontsize=8)

    ani = animation.FuncAnimation(
        fig, draw, frames=len(frame_indices), interval=int(1000 / fps), repeat=False
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()

    try:
        if ext == '.gif':
            ani.save(output_path, writer=animation.PillowWriter(fps=fps))
            final_path = output_path
        else:
            ani.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
            final_path = output_path
    except Exception:
        fallback_path = os.path.splitext(output_path)[0] + '.gif'
        ani.save(fallback_path, writer=animation.PillowWriter(fps=fps))
        final_path = fallback_path

    plt.close(fig)
    return final_path


# Collision-safe post-processing for candidate trajectories.
def apply_safe_post_processing(traj_np, task, tensor_args, n_points, max_iters=5):
    """
    Smooth and resample a trajectory while guarding against collisions.
    """
    current_safe_traj = np.copy(traj_np)
    
    # Laplacian smoothing with collision rollback.
    for i in range(max_iters):
        smoothed = np.copy(current_safe_traj)
        # Keep endpoints fixed.
        smoothed[1:-1] = 0.5 * smoothed[1:-1] + 0.25 * smoothed[:-2] + 0.25 * smoothed[2:]
        
        smoothed_t = torch.tensor(smoothed, **tensor_args).unsqueeze(0)
        
        # Any positive collision cost marks this candidate as invalid.
        if task.compute_collision(smoothed_t).sum().item() > 0:
            break
        else:
            current_safe_traj = smoothed

    # Uniform resampling with a second collision check.
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
    
    final_resampled_t = torch.tensor(final_resampled, **tensor_args).unsqueeze(0)
    
    if task.compute_collision(final_resampled_t).sum().item() > 0:
        return current_safe_traj
        
    return final_resampled

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
    spine_width = 3.0
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
    
    ax.tick_params(axis='both', which='major', labelsize=10, width=spine_width)

    if current_lines is not None:
        for k in current_lines.keys():
            current_lines[k].clear()

    env.render(ax)
    
    # Label obstacles for quick topology debugging.
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    for k, center in enumerate(obs_centers_np):
        ax.text(center[0], center[1], f"O{k+1}", color='white', ha='center', va='center', weight='bold', fontsize=9, zorder=10)

    # Raw history trajectory.
    if len(history_path_list) > 0:
        full_hist_np = np.concatenate(history_path_list, axis=0)
        lines = ax.plot(full_hist_np[:, 0], full_hist_np[:, 1], color='blue', linestyle='-', linewidth=1.6, alpha=0.9, label='Raw History', visible=VIS_STATE['Raw History'])
        current_lines['Raw History'].extend(lines)
        
        initial_pt = history_path_list[0][0]
        ax.plot(initial_pt[0], initial_pt[1], 'bs', markersize=8, markeredgecolor='white', zorder=12)
        
        reached_pts = np.array([traj[-1] for traj in history_path_list])
        ax.plot(reached_pts[:, 0], reached_pts[:, 1], 'bo', markersize=6, markeredgecolor='white', zorder=12)

    # Global taut homotopy reference.
    if latest_optimized_traj is not None:
        lines = ax.plot(latest_optimized_traj[:, 0], latest_optimized_traj[:, 1], 
                        color='cyan', linestyle='-', linewidth=4.0, alpha=0.4, 
                        solid_capstyle='round', label='Global Taut Cable', 
                        visible=VIS_STATE['Global H1'])
        current_lines['Global H1'].extend(lines)

    # Current MPD candidate.
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

    # Alternative homotopy branches.
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
    render: bool = True, 
    seed: int = 42,
    results_dir: str = 'logs',
    auto_goals: str = '',
    max_auto_iterations: int = 0,
    save_denoise_animation_flag: bool = False,
    animation_fps: int = 10,
    animation_stride: int = 2,
    animation_format: str = 'gif',
    max_animation_candidates: int = 4,
    fixed_history: str = '',
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    print('##########################################################################################################')
    print(f'Model -- {model_id} | Algorithm -- {planner_alg}')
    
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    results_dir = os.path.join(INFERENCE_RESULTS_DIR, "interactive_inference", str(seed))
    os.makedirs(results_dir, exist_ok=True)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))

    # Load dataset and planning primitives.
    train_subset, _, _, _ = get_dataset(
        dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05,
        **args, tensor_args=tensor_args
    )
    dataset = train_subset.dataset
    n_support_points = dataset.n_support_points
    robot = dataset.robot
    base_task = dataset.task  # Keep task type for rebuilding with the dynamic env.

    # Replace base env with the dynamic obstacle environment.
    env = EnvDense2DExtraObjects(
        tensor_args=tensor_args, 
        drop_old_num=2, 
        num_extra_spheres=24, 
        num_extra_boxes=0, 
        seed=42
    )
    # Rebuild task so collision fields/SDF match the dynamic map.
    task = type(base_task)(env=env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
    
    # Update dataset references to keep model/guide/task consistent.
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

    # Fixed initial state.
    q_dim = getattr(env, 'q_dim', 2)
    start_state_pos = torch.zeros(q_dim, **tensor_args)
    # Tweak this point if it spawns inside newly added obstacles.
    custom_start_coords = [-0.9, -0.9]
    start_state_pos[:2] = torch.tensor(custom_start_coords, **tensor_args)

    fixed_history_np = parse_polyline(fixed_history)
    history_path_list = []
    if fixed_history_np is not None:
        history_path_list.append(fixed_history_np.copy())
        start_state_pos[:2] = torch.tensor(fixed_history_np[-1], **tensor_args)
        print(f"Fixed history loaded with {len(fixed_history_np)} points.")
    current_start_pos = start_state_pos.clone()
    
    latest_all_trajs = None
    latest_best_traj = None
    latest_optimized_traj = None
    latest_best_energy = None

    print("\nInitialization finished, model loaded...")
    iteration_count = 0

    auto_goal_list = parse_auto_goals(auto_goals)
    if auto_goal_list:
        print(f"Auto-goal mode enabled with {len(auto_goal_list)} goals.")
    elif not render:
        raise ValueError("render=False requires --auto_goals (e.g., --auto_goals '0.6,0.6').")

    animation_format = animation_format.lower().strip()
    if animation_format not in {'gif', 'mp4'}:
        raise ValueError("animation_format must be 'gif' or 'mp4'.")
    animations_dir = os.path.join(results_dir, "animations")
    if save_denoise_animation_flag:
        os.makedirs(animations_dir, exist_ok=True)

    VIS_STATE = {
        'Raw History': True,
        'Global H1': True,
        'MPD Candidate': True,
        'All MPD': True,
        'Ref Topo': True,
        'All Topo': False
    }
    current_lines = {k: [] for k in VIS_STATE.keys()}

    fig_topo, ax_topo = None, None
    if render:
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
    all_iterations_data = []  # Collected per-iteration outputs.
    
    # Single file that stores the whole interactive session.
    combined_save_path = os.path.join(results_dir, f'full_session_data_seed_{seed}_2.pkl')
    while True:
        if max_auto_iterations > 0 and iteration_count >= max_auto_iterations:
            print(f"Reached max_auto_iterations={max_auto_iterations}, stopping.")
            break

        iteration_count += 1
        print(f"\n====================== [ITERATION-{iteration_count} ] ======================")

        if render:
            render_demo(fig_topo, ax_topo, iteration_count, env, history_path_list, latest_all_trajs, latest_best_traj, current_start_pos,
                        latest_optimized_traj, current_homotopy_classes, current_homotopy_energies, current_homotopy_indices,
                        VIS_STATE, current_lines, latest_best_energy)

        if auto_goal_list:
            if iteration_count > len(auto_goal_list):
                print("\nAuto goals consumed, planning finished.")
                break
            target_x, target_y = auto_goal_list[iteration_count - 1]
            print(f"Auto goal {iteration_count}/{len(auto_goal_list)} selected.")
        else:
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

        with TimerCUDA():
            trajs_normalized_iters = model.run_inference(None, hard_conds, n_samples=n_samples, horizon=n_support_points, return_chain=True, sample_fn=ddpm_sample_fn, **sample_fn_kwargs, n_diffusion_steps_without_noise=n_diffusion_steps_without_noise)

        trajs_chain_unnorm = dataset.unnormalize_trajectories(trajs_normalized_iters)
        _, _, trajs_final_free, trajs_free_idxs, _ = task.get_trajs_collision_and_free(
            trajs_chain_unnorm[-1], return_indices=True
        )

        if trajs_final_free is not None:
            trajs_free_np = trajs_final_free[..., :2].cpu().numpy()
            unique_mpd_classes, unique_sigs = [], []
            class_source_free_idx = []

            for free_idx, traj in enumerate(trajs_free_np):
                sig = get_trajectory_signature(traj, obs_centers_np)
                if not any(np.all(np.abs(sig - ext_sig) < 0.3) for ext_sig in unique_sigs):
                    unique_sigs.append(sig)
                    unique_mpd_classes.append(prune_self_intersections(traj))
                    class_source_free_idx.append(free_idx)
                    
            mpd_energies, _ = evaluate_homotopy_topological_energy(hist_for_eval, unique_mpd_classes, obs_centers_np, w_max=0.8)
            combined_scores = []
            for traj, energy in zip(unique_mpd_classes, mpd_energies):
                # Merge history with current candidate.
                if len(hist_for_eval) > 0:
                    combined_raw = np.vstack((hist_for_eval, traj[1:]))
                else:
                    combined_raw = traj
                    
                # Build a taut global reference for topology evaluation.
                refined_combined = get_simplest_homotopy_curve(combined_raw, obs_centers_np, obs_types, obs_dims)
                check_traj = refined_combined if refined_combined is not None else combined_raw
                
                # Evaluate global winding signature.
                global_sig = get_trajectory_signature(check_traj, obs_centers_np)
                print("global sig: ", global_sig)
                
                # Combined score: topology energy + path length penalty.
                score = energy + 0.8 * np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                combined_scores.append(score)
            
            sorted_data = sorted(
                zip(
                    range(1, len(unique_mpd_classes) + 1),
                    unique_mpd_classes,
                    mpd_energies,
                    combined_scores,
                    class_source_free_idx,
                ),
                key=lambda x: x[3]
            )
            
            current_homotopy_indices = [x[0] for x in sorted_data]
            current_homotopy_classes = [x[1] for x in sorted_data]
            current_homotopy_energies = [x[2] for x in sorted_data]

            latest_best_traj_raw = current_homotopy_classes[0].copy()
            latest_best_traj = current_homotopy_classes[0]
            latest_best_energy = current_homotopy_energies[0]
            # Apply collision-safe smoothing/resampling.
            latest_best_traj = apply_safe_post_processing(latest_best_traj, task, tensor_args, n_support_points, max_iters=5)

            print(f"Generation Finished!  {len(current_homotopy_classes)} Topology found in total, select H{current_homotopy_indices[0]} (Energy: {latest_best_energy:.2f})")
            key_steps_for_paper = [4]  # Save candidates for selected iterations only.
            is_key_step = (iteration_count in key_steps_for_paper)
            
            save_candidates = []
            
            if is_key_step and len(trajs_free_np) > 1:
                # Save a subset of collision-free candidates for paper figures.
                for traj in trajs_free_np[:10]: 
                    
                    # Skip candidates that duplicate the best trajectory.
                    if np.allclose(traj, latest_best_traj, atol=1e-3):
                        continue
                    tmp_traj = apply_safe_post_processing(traj, task, tensor_args, n_support_points, max_iters=5)
                    save_candidates.append(tmp_traj.copy())
            path_length = np.sum(np.linalg.norm(np.diff(latest_best_traj, axis=0), axis=1))

            sm_tensor = torch.tensor(latest_best_traj, **tensor_args).unsqueeze(0)
            smoothness = compute_smoothness(sm_tensor, robot).item()

            print("\n" + "="*55)
            print("[ 轨迹生成评估报告 | METRICS REPORT ]")
            print("="*55)
            print(f"轨迹总长度 (Length)    : {path_length:.3f} m")
            print(f"运动学平滑度 (Smoothness): {smoothness:.2f} ")
            print(f"拓扑势能 (Topo Energy) : {latest_best_energy:.3f}")
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

            denoise_animation_path = None
            if save_denoise_animation_flag and trajs_free_idxs is not None and trajs_free_idxs.nelement() > 0:
                free_batch_indices = trajs_free_idxs[:, -1].detach().cpu().numpy().reshape(-1).astype(int)
                if len(free_batch_indices) == len(trajs_free_np):
                    max_animation_candidates = max(1, int(max_animation_candidates))
                    animation_candidates = []
                    for rank, (h_id, h_traj, _, _, source_free_idx) in enumerate(sorted_data[:max_animation_candidates]):
                        if rank == 0:
                            final_traj = latest_best_traj
                            label = f"H{h_id} Best"
                        else:
                            final_traj = apply_safe_post_processing(
                                h_traj, task, tensor_args, n_support_points, max_iters=3
                            )
                            label = f"H{h_id}"
                        animation_candidates.append(
                            dict(
                                source_free_idx=int(source_free_idx),
                                final_traj=np.asarray(final_traj),
                                label=label,
                                is_best=(rank == 0),
                            )
                        )

                    # Fill with extra feasible samples when unique homotopies are not enough.
                    used_free_idxs = {c['source_free_idx'] for c in animation_candidates}
                    if len(animation_candidates) < max_animation_candidates:
                        for free_idx, free_traj in enumerate(trajs_free_np):
                            if free_idx in used_free_idxs:
                                continue
                            mean_dist = np.mean(np.linalg.norm(free_traj - latest_best_traj_raw, axis=1))
                            if mean_dist < 0.03:
                                continue
                            extra_traj = apply_safe_post_processing(
                                free_traj, task, tensor_args, n_support_points, max_iters=2
                            )
                            animation_candidates.append(
                                dict(
                                    source_free_idx=int(free_idx),
                                    final_traj=np.asarray(extra_traj),
                                    label=f"Feasible {len(animation_candidates) + 1}",
                                    is_best=False,
                                )
                            )
                            used_free_idxs.add(free_idx)
                            if len(animation_candidates) >= max_animation_candidates:
                                break

                    candidate_chains = []
                    candidate_finals = []
                    candidate_labels = []
                    candidate_colors = []
                    best_candidate_idx = 0
                    palette = ['orangered', 'deepskyblue', 'mediumseagreen', 'goldenrod', 'mediumpurple', 'hotpink']
                    for i, candidate in enumerate(animation_candidates):
                        source_idx = candidate['source_free_idx']
                        if source_idx < 0 or source_idx >= len(free_batch_indices):
                            continue
                        batch_idx = int(free_batch_indices[source_idx])
                        chain_xy = trajs_chain_unnorm[:, batch_idx, :, :2].detach().cpu().numpy()
                        candidate_chains.append(chain_xy)
                        candidate_finals.append(candidate['final_traj'])
                        candidate_labels.append(candidate['label'])
                        candidate_colors.append(palette[i % len(palette)])
                        if candidate['is_best']:
                            best_candidate_idx = len(candidate_chains) - 1

                    hist_np = np.concatenate(history_path_list[:-1], axis=0) if len(history_path_list) > 1 else None
                    anim_ext = 'gif' if animation_format == 'gif' else 'mp4'
                    animation_output = os.path.join(
                        animations_dir, f"iter_{iteration_count:03d}_seed_{seed}.{anim_ext}"
                    )
                    denoise_animation_path = save_denoise_animation_multi(
                        env=env,
                        candidate_chains=candidate_chains,
                        candidate_finals=candidate_finals,
                        candidate_labels=candidate_labels,
                        candidate_colors=candidate_colors,
                        best_candidate_idx=best_candidate_idx,
                        start_xy=start_state_pos[:2].detach().cpu().numpy(),
                        goal_xy=goal_state_pos[:2].detach().cpu().numpy(),
                        output_path=animation_output,
                        history_xy=hist_np,
                        fps=animation_fps,
                        stride=animation_stride,
                    )
                    print(f"🎬 Denoising animation saved at: {denoise_animation_path}")
                else:
                    print("⚠️ Skip denoising animation: free index count mismatch.")

            # Append this step and overwrite the session file.
            current_step_data = {
                'iteration': iteration_count,
                'waypoints': latest_best_traj.copy(),  # Copy to keep payload immutable.
                'taut_cable_global': latest_optimized_traj.copy() if latest_optimized_traj is not None else None,
                'top_3_candidates': save_candidates,
                'denoising_animation_path': denoise_animation_path,
                'metrics': {
                    'length': path_length,
                    'smoothness': smoothness,
                    'energy': latest_best_energy
                }
            }
            
            all_iterations_data.append(current_step_data)
            
            # Environment metadata is static; store once with all steps.
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
                'steps': all_iterations_data
            }
            
            with open(combined_save_path, 'wb') as f:
                pickle.dump(final_save_payload, f)
            
            print(f"💾 Full session data updated at: {combined_save_path}")
        else:
            print("All collision!!!")
            latest_all_trajs, latest_best_traj, latest_optimized_traj, latest_best_energy = None, None, None, None

if __name__ == '__main__':
    run_experiment(experiment)
