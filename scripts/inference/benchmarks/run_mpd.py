import os
import time
import numpy as np
import pandas as pd
import torch
from math import ceil
import matplotlib
matplotlib.use('Agg')  # Headless plotting backend.

# Core dependencies and MPD model stack.
from experiment_launcher import single_experiment_yaml, run_experiment
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostGPTrajectory
from mpd.models import TemporalUnet, UNET_DIM_MULTS
from mpd.models.diffusion_models.guides import GuideManagerTrajectoriesWithVelocity
from mpd.models.diffusion_models.sample_functions import ddpm_sample_fn
from mpd.trainer import get_dataset, get_model
from mpd.utils.bench_plotting import render_segmented_trial_plot
from mpd.utils.bench_io import ensure_dir, resolve_output_root
from mpd.utils.bench_metrics import format_time_to_success_summary
from mpd.utils.loading import load_params_from_yaml
from mpd.utils.waypoints import generate_sequential_waypoints
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, freeze_torch_model_params
from torch_robotics.trajectory.metrics import compute_smoothness, compute_path_length

# Dynamic environment.
from mpd.environments.env_dense_2d_extra_objects import EnvDense2DExtraObjects

from mpd.utils.topology_utils import (
    get_trajectory_signature,
    get_simplest_homotopy_curve,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, "..", ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "data_trained_models")
RESULTS_DIR = os.path.join(RESULTS_ROOT, "benchmark_time_to_success_official_mpd")

# Benchmark entrypoint.
@single_experiment_yaml
def run_benchmark_official_mpd(
    model_id: str = 'EnvDense2D-RobotPointMass',
    num_trials: int = 100,
    num_segments: int = 5,
    n_samples: int = 50,  # Match TMPD compute budget for fair comparison.
    n_diffusion_steps_without_noise: int = 5, 
    start_guide_steps_fraction: float = 0.25,
    n_guide_steps: int = 5,
    device: str = 'cuda',
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

    print('Initializing Benchmark (Official MPD ONLY - HEADLESS MODE)...')
    print(f'Plots will be saved to: {plots_dir}')
    
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

        # Keep a failed sample for visualization when all candidates collide.
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
                
                # Official MPD sampling setup.
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
                    tracker["failed_traj"] = t_unnorm[0, ..., :2].cpu().numpy()
                    print(f"    [Official MPD] Segment {seg+1}: Collision detected. Resampling (Attempt {attempts}/{max_mpd_attempts})...")

            if best_traj_np is None:
                print(f"    [Official MPD] FATAL: Stuck in collision after {max_mpd_attempts} attempts.")
                tracker["collision"] = True

            seg_time = time.time() - t0
            tracker["time"] += seg_time

            if best_traj_np is not None:
                tracker["history"].append(best_traj_np)
                tracker["sr"] += 1
                tracker["pl_list"].append(np.sum(np.linalg.norm(np.diff(best_traj_np, axis=0), axis=1)))
                tracker["sm_list"].append(compute_smoothness(torch.tensor(best_traj_np, **tensor_args).unsqueeze(0), robot).item())
                
                # Evaluate global winding and accumulated energy.
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

        if tracker["collision"]:
            title_color = 'red'
            status_txt = f"Collision at Seg {tracker['sr']+1}"
        elif tracker["tangled"]:
            title_color = 'darkorange'
            status_txt = f"Tangled (Energy: {tracker['final_energy']:.2f})"
        else:
            title_color = 'green'
            status_txt = f"Success (Energy: {tracker['final_energy']:.2f})"

        failed_wp = waypoints_np[tracker["sr"] + 1] if tracker["collision"] else None
        render_segmented_trial_plot(
            env=dynamic_env,
            waypoints_np=waypoints_np,
            history_trajs=tracker["history"],
            trial_idx=trial + 1,
            method_label="MPD",
            status_txt=status_txt,
            title_color=title_color,
            output_path=os.path.join(plots_dir, f"mpd_trial_{trial+1:03d}.png"),
            taut_traj=taut_traj,
            is_tangled=tracker["tangled"],
            failed_goal=failed_wp,
            failed_goal_label=None,
            failed_traj=tracker["failed_traj"] if tracker["collision"] else None,
            dpi=150,
        )

        # Quality metrics require full success; time includes all attempts.
        is_fully_successful = (not tracker["tangled"]) and (not tracker["collision"]) and (tracker["sr"] == num_segments)
        attempted_segs = tracker["sr"] + (1 if tracker["collision"] else 0)

        all_results.append({
            "Method": "Official MPD", "Trial": trial + 1,
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
    
    print("\n" + "="*95)
    print("OFFICIAL MPD REPORT")
    print("="*95)
    print(summary.to_string())
    print("="*95)

if __name__ == '__main__':
    run_experiment(run_benchmark_official_mpd)
