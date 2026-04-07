import numpy as np
import heapq
import math
import time
from matplotlib import pyplot as plt
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes

# ⚠️ 注意：这里替换为你的环境文件名
from tmpd_baselines.environment.env_simple_2d_extra_objects import EnvSimple2DExtraObjects

class TopoNode:
    def __init__(self, x, y, g, h, parent, W_array):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.W_array = W_array

    def __lt__(self, other):
        return self.f < other.f
        
    def get_state_key(self):
        quantized_W = tuple(np.round(self.W_array, 1))
        return (round(self.x, 3), round(self.y, 3), quantized_W)

def calc_delta_winding_vectorized(p1, p2, obstacles):
    if len(obstacles) == 0: return np.array([])
    p1, p2 = np.array(p1), np.array(p2)
    v1 = p1 - obstacles
    v2 = p2 - obstacles
    theta1 = np.arctan2(v1[:, 1], v1[:, 0])
    theta2 = np.arctan2(v2[:, 1], v2[:, 0])
    delta_theta = (theta2 - theta1 + np.pi) % (2 * np.pi) - np.pi
    return delta_theta / (2 * np.pi)

def topo_a_star_with_tree(start, goal, env, step_size=0.05, W_th=0.95, robot_radius=0.02):
    obs_centers = env.active_obs_centers
    num_obstacles = len(obs_centers)
    
    def is_collision(px, py):
        if not (-1.0 <= px <= 1.0 and -1.0 <= py <= 1.0): return True
        for i, center in enumerate(obs_centers):
            ctype = env.active_obs_types[i]
            cdim = env.active_obs_dims[i]
            if ctype == 'sphere' and math.hypot(px - center[0], py - center[1]) <= cdim[0] + robot_radius:
                return True
            elif ctype == 'box' and abs(px - center[0]) <= cdim[0] + robot_radius and abs(py - center[1]) <= cdim[1] + robot_radius:
                return True
        return False

    start_node = TopoNode(start[0], start[1], 0, math.hypot(start[0]-goal[0], start[1]-goal[1]), None, np.zeros(num_obstacles))
    open_set = []
    heapq.heappush(open_set, start_node)
    closed_set = set()
    
    # 【新增】记录所有被访问过的坐标，用于画“探索树”
    explored_states = [] 
    
    motions = [
        (0, step_size), (0, -step_size), (step_size, 0), (-step_size, 0),
        (step_size, step_size), (step_size, -step_size), (-step_size, step_size), (-step_size, -step_size)
    ]
    
    while open_set:
        curr = heapq.heappop(open_set)
        
        if math.hypot(curr.x - goal[0], curr.y - goal[1]) <= step_size * 1.5:
            path = [(goal[0], goal[1])]
            while curr:
                path.append((curr.x, curr.y))
                curr = curr.parent
            return path[::-1], explored_states
            
        state_key = curr.get_state_key()
        if state_key in closed_set: continue
        closed_set.add(state_key)
        
        # 记录真实被弹出的节点
        explored_states.append((curr.x, curr.y))
        
        for dx, dy in motions:
            nx, ny = curr.x + dx, curr.y + dy
            if is_collision(nx, ny): continue
            
            delta_W = calc_delta_winding_vectorized((curr.x, curr.y), (nx, ny), obs_centers)
            new_W = curr.W_array + delta_W
            if np.any(np.abs(new_W) >= W_th): continue
                
            new_g = curr.g + math.hypot(dx, dy)
            new_h = math.hypot(nx - goal[0], ny - goal[1])
            heapq.heappush(open_set, TopoNode(nx, ny, new_g, new_h, curr, new_W))
            
    return None, explored_states

# ==========================================
# 绘图与执行
# ==========================================
if __name__ == '__main__':
    env = EnvSimple2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS,
        drop_old_num=8,        # 稍微少删一点旧障碍物，让地图密集一点
        num_extra_spheres=6,   # 多加点新障碍物
        num_extra_boxes=4,
        seed=1024              # 固定种子找一个经典的密集图
    )
    
    start_pt = (-0.8, -0.8) 
    goal_pt = (0.8, 0.8)    
    
    print("Running Topo-A* ...")
    t0 = time.time()
    path, explored = topo_a_star_with_tree(start_pt, goal_pt, env, step_size=0.03, W_th=0.95, robot_radius=0.03)
    t1 = time.time()
    
    # 1. 创建画布与渲染环境障碍物
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    
    # 2. 绘制 A* 漫山遍野的“探索足迹” (淡蓝色小点/细线)
    if explored:
        exp_x = [p[0] for p in explored]
        exp_y = [p[1] for p in explored]
        ax.scatter(exp_x, exp_y, s=2, c='cyan', alpha=0.3, zorder=2, label=f'Explored States ({len(explored)})')

    # 3. 绘制起点和终点
    ax.plot(start_pt[0], start_pt[1], 'go', markersize=10, zorder=5, label='Start')
    ax.plot(goal_pt[0], goal_pt[1], 'r*', markersize=12, zorder=5, label='Goal')
    
    # 4. 绘制最终的无缠绕最优路径
    if path:
        print(f"✅ Success! Path steps: {len(path)} | Time: {(t1-t0)*1000:.1f}ms | Explored: {len(explored)} nodes")
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=3, zorder=4, label='Optimal Tangle-Free Path')
    else:
        print(f"❌ Failed! Time: {(t1-t0)*1000:.1f}ms | Explored: {len(explored)} nodes")
    
    # 5. 学术图表美化与保存
    plt.title(f"Baseline: Topo-A* Search Footprint\nTime: {(t1-t0)*1000:.1f} ms | Nodes Checked: {len(explored)}", fontsize=12, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    
    # 保存为高清 PDF 用于论文
    plt.savefig("topo_astar_footprint.pdf", dpi=300, bbox_inches='tight')
    plt.show()