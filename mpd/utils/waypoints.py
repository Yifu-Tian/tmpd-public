import numpy as np
import torch

from mpd.utils.topology_utils import is_trajectory_safe


def generate_sequential_waypoints(
    env,
    task,
    q_dim,
    tensor_args,
    num_segments=5,
    max_attempts=1000,
    min_segment_distance=0.6,
    sample_bound=0.85,
):
    obs_centers_np = getattr(env, "active_obs_centers", [])
    obs_types = getattr(env, "active_obs_types", ["sphere"] * len(obs_centers_np))
    obs_dims = getattr(env, "active_obs_dims", [np.array([0.125])] * len(obs_centers_np))
    waypoints_np, waypoints_t = [], []

    while True:
        p0 = np.random.uniform(-sample_bound, sample_bound, size=q_dim)
        t0 = torch.tensor(p0, **tensor_args)
        if task.compute_collision(t0.unsqueeze(0)).item() == 0:
            waypoints_np.append(p0)
            waypoints_t.append(t0)
            break

    for _ in range(num_segments):
        curr_p = waypoints_np[-1]
        found = False
        for _ in range(max_attempts):
            next_p = np.random.uniform(-sample_bound, sample_bound, size=q_dim)
            next_t = torch.tensor(next_p, **tensor_args)
            if task.compute_collision(next_t.unsqueeze(0)).item() > 0:
                continue
            if np.linalg.norm(next_p - curr_p) < min_segment_distance:
                continue
            # "Hard" segment: straight line should intersect obstacles.
            if not is_trajectory_safe(np.array([curr_p, next_p]), obs_centers_np, obs_types, obs_dims):
                waypoints_np.append(next_p)
                waypoints_t.append(next_t)
                found = True
                break
        if not found:
            return None, None

    return waypoints_t, waypoints_np


def sample_collision_free_start_goal(task, tensor_args, min_dist=0.8, max_attempts=5000, sample_bound=0.85):
    for _ in range(max_attempts):
        start_np = np.random.uniform(-sample_bound, sample_bound, size=2)
        goal_np = np.random.uniform(-sample_bound, sample_bound, size=2)

        if np.linalg.norm(start_np - goal_np) < min_dist:
            continue

        start_t = torch.tensor(start_np, **tensor_args)
        goal_t = torch.tensor(goal_np, **tensor_args)
        if task.compute_collision(start_t.unsqueeze(0)).item() == 0 and task.compute_collision(goal_t.unsqueeze(0)).item() == 0:
            return start_np, goal_np, start_t, goal_t

    return None, None, None, None


def resample_trajectory(traj_np, n_points):
    if len(traj_np) == n_points:
        return traj_np
    if len(traj_np) < 2:
        return np.zeros((n_points, traj_np.shape[1]))

    diffs = np.linalg.norm(np.diff(traj_np, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(diffs), 0, 0)
    if cum_dists[-1] == 0:
        return np.tile(traj_np[0], (n_points, 1))

    resampled = np.zeros((n_points, traj_np.shape[1]))
    target_dists = np.linspace(0, cum_dists[-1], n_points)
    for i in range(traj_np.shape[1]):
        resampled[:, i] = np.interp(target_dists, cum_dists, traj_np[:, i])
    return resampled
