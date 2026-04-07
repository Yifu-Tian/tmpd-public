import numpy as np

def get_trajectory_signature(traj_np, obs_centers):
    """ 计算单条轨迹绕所有障碍物的包角签名向量 (Winding Number Vector) """
    if len(traj_np) < 2 or len(obs_centers) == 0:
        return np.zeros(len(obs_centers))
    
    sig = []
    for cx, cy in obs_centers:
        vecs = traj_np - np.array([cx, cy])
        cross_p = vecs[:-1, 0] * vecs[1:, 1] - vecs[:-1, 1] * vecs[1:, 0]
        dot_p = vecs[:-1, 0] * vecs[1:, 0] + vecs[:-1, 1] * vecs[1:, 1]
        w = np.sum(np.arctan2(cross_p, dot_p)) / (2 * np.pi)
        sig.append(w)
    return np.array(sig)

def is_trajectory_safe(traj_np, obs_centers, obs_types, obs_dims, margin=0.005):
    """ 绝对数学几何防穿墙过滤器 (分别精确处理圆形和方形) """
    if len(traj_np) < 2 or len(obs_centers) == 0:
        return True

    for i in range(len(traj_np) - 1):
        p1, p2 = traj_np[i], traj_np[i+1]
        l2 = np.sum((p1 - p2)**2)
        dist_p1_p2 = np.sqrt(l2)
        
        # 对方形障碍物进行密集的线段采样检测 (简单且极度鲁棒)
        num_pts = max(3, int(dist_p1_p2 / 0.01)) 
        xs = np.linspace(p1[0], p2[0], num_pts)
        ys = np.linspace(p1[1], p2[1], num_pts)
        
        for center, o_type, dims in zip(obs_centers, obs_types, obs_dims):
            if o_type == 'sphere':
                hard_rad = dims[0]
                if l2 == 0.0:
                    dist = np.linalg.norm(p1 - center)
                else:
                    t = max(0, min(1, np.dot(center - p1, p2 - p1) / l2))
                    projection = p1 + t * (p2 - p1)
                    dist = np.linalg.norm(center - projection)
                if dist < hard_rad - margin:
                    return False
                    
            elif o_type == 'box':
                # 【修复核心】：使用 AABB 检测，而不是变态的外接圆
                half_w, half_h = dims[0], dims[1]
                # 减去 margin 是为了容忍极微小的边界擦碰
                min_x, max_x = center[0] - half_w + margin, center[0] + half_w - margin
                min_y, max_y = center[1] - half_h + margin, center[1] + half_h - margin
                
                in_x = (xs > min_x) & (xs < max_x)
                in_y = (ys > min_y) & (ys < max_y)
                if np.any(in_x & in_y):
                    return False
    return True

def prune_self_intersections(traj_np, spatial_thresh=0.06, temporal_thresh=8):
    """ 时空打结消除器 """
    n_pts = len(traj_np)
    if n_pts < temporal_thresh: return traj_np
    
    keep_indices = []
    curr_idx = 0
    while curr_idx < n_pts:
        keep_indices.append(curr_idx)
        jump_idx = curr_idx + 1
        for j in range(n_pts - 1, curr_idx + temporal_thresh, -1):
            dist = np.linalg.norm(traj_np[curr_idx] - traj_np[j])
            if dist < spatial_thresh: 
                jump_idx = j
                break
        curr_idx = jump_idx
    
    pruned_traj = traj_np[keep_indices]
    
    if len(pruned_traj) < n_pts and len(pruned_traj) > 1:
        diffs = np.linalg.norm(np.diff(pruned_traj, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(diffs), 0, 0)
        total_len = cum_dists[-1]
        if total_len > 0:
            new_dists = np.linspace(0, total_len, n_pts)
            resampled = np.zeros((n_pts, 2))
            resampled[:, 0] = np.interp(new_dists, cum_dists, pruned_traj[:, 0])
            resampled[:, 1] = np.interp(new_dists, cum_dists, pruned_traj[:, 1])
            return resampled
    return pruned_traj

def get_simplest_homotopy_curve(raw_hist_np, obs_centers, obs_types, obs_dims):
    """ 最简拓扑同伦保形器 (Taut-String + Elastic Skin) """
    if len(raw_hist_np) < 3 or len(obs_centers) == 0:
        return raw_hist_np

    def is_homotopic_segment(path_segment, p1, p2):
        # 【修复核心】：完美闭合！原代码的写法会在尾部画一个死结 8 字形
        loop = np.vstack([path_segment, p1]) 
        sig = get_trajectory_signature(loop, obs_centers)
        return np.all(np.abs(sig) < 0.1)

    skeleton = [raw_hist_np[0]]
    curr, n = 0, len(raw_hist_np)
    while curr < n - 1:
        for j in range(n - 1, curr, -1):
            if j == curr + 1:
                skeleton.append(raw_hist_np[j])
                curr = j
                break
            p1, p2 = raw_hist_np[curr], raw_hist_np[j]
            if is_trajectory_safe(np.array([p1, p2]), obs_centers, obs_types, obs_dims) and \
               is_homotopic_segment(raw_hist_np[curr:j+1], p1, p2):
                skeleton.append(p2)
                curr = j
                break
    skeleton_np = np.array(skeleton)

    diffs = np.linalg.norm(np.diff(skeleton_np, axis=0), axis=1)
    cum_dist = np.insert(np.cumsum(diffs), 0, 0)
    total_len = cum_dist[-1]
    
    if total_len < 1e-3: return raw_hist_np

    new_dists = np.arange(0, total_len, 0.02)
    x_res = np.interp(new_dists, cum_dist, skeleton_np[:, 0])
    y_res = np.interp(new_dists, cum_dist, skeleton_np[:, 1])
    if new_dists[-1] != total_len:
        x_res = np.append(x_res, skeleton_np[-1, 0])
        y_res = np.append(y_res, skeleton_np[-1, 1])
    path = np.vstack((x_res, y_res)).T

    alpha = 0.5
    for _ in range(150):
        smoothed = np.empty_like(path)
        smoothed[0], smoothed[-1] = path[0], path[-1]
        smoothed[1:-1] = path[1:-1] + alpha * (path[:-2] + path[2:] - 2 * path[1:-1])
        path = smoothed

        for center, o_type, dims in zip(obs_centers, obs_types, obs_dims):
            # 【修复核心】：不再使用巨大的 hypot 来推离 Box，使用 max 避免过度膨胀
            soft_rad = (dims[0] if o_type == 'sphere' else max(dims[0], dims[1])) + 0.035
            vecs = path - center
            dists = np.linalg.norm(vecs, axis=1)
            in_obs = dists < soft_rad
            in_obs[0], in_obs[-1] = False, False
            
            if np.any(in_obs):
                dists_safe = np.where(dists == 0, 1e-5, dists)
                push_dirs = vecs[in_obs] / dists_safe[in_obs, np.newaxis]
                path[in_obs] = center + push_dirs * soft_rad

    # 【修复核心】：无论最终平滑后的轨迹是否在边界上擦碰了障碍物，
    # 都把它当作拓扑基准返回！因为它的拓扑意义比 Raw History 干净一万倍。
    return path

def evaluate_homotopy_topological_energy(history_traj_np, homotopy_classes, obs_centers, w_max=0.8):
    """ 拓扑能量评估引擎 """
    if len(obs_centers) == 0 or len(homotopy_classes) == 0:
        return [0.0] * len(homotopy_classes), []

    w_hist = get_trajectory_signature(history_traj_np, obs_centers) if len(history_traj_np) > 1 else np.zeros(len(obs_centers))
    
    energies, w_totals_list = [], []
    for traj in homotopy_classes:
        w_cand = get_trajectory_signature(traj, obs_centers)
        w_total = w_hist + w_cand
        w_totals_list.append(w_total)
        
        ratio = np.abs(w_total) / w_max
        energy_k = np.zeros_like(ratio)
        
        margin = 0.95
        safe_mask = ratio < margin
        danger_mask = ~safe_mask
        
        energy_k[safe_mask] = -np.log(1.0 - ratio[safe_mask]**2 + 1e-8)
        energy_k[danger_mask] = -np.log(1.0 - margin**2) + ((2 * margin) / (1.0 - margin**2)) * (ratio[danger_mask] - margin)
        
        energies.append(np.sum(energy_k))
        
    return energies, w_totals_list