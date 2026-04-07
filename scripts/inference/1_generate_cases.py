import os
import pickle
import numpy as np
import torch
from experiment_launcher import single_experiment_yaml, run_experiment
from mpd.trainer import get_dataset
from mpd.utils.loading import load_params_from_yaml
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device
from tmpd_baselines.environment.env_simple_2d_extra_objects import EnvSimple2DExtraObjects
from mpd.utils.topology_utils import is_trajectory_safe

TRAINED_MODELS_DIR = '../../data_trained_models/'
CASES_SAVE_PATH = 'hard_cases_100.pkl'

def generate_hard_case_start_goal(env, task, q_dim, tensor_args, max_attempts=1000):
    obs_centers_np = getattr(env, 'active_obs_centers', [])
    obs_types = getattr(env, 'active_obs_types', ['sphere'] * len(obs_centers_np))
    obs_dims = getattr(env, 'active_obs_dims', [np.array([0.125])] * len(obs_centers_np))
    for _ in range(max_attempts):
        start_np = np.random.uniform(-0.85, 0.85, size=q_dim)
        goal_np = np.random.uniform(-0.85, 0.85, size=q_dim)
        start_t, goal_t = torch.tensor(start_np, **tensor_args), torch.tensor(goal_np, **tensor_args)
        
        # 确保起终点无碰撞且距离足够远
        if task.compute_collision(start_t.unsqueeze(0)).item() > 0 or task.compute_collision(goal_t.unsqueeze(0)).item() > 0:
            continue
        if np.linalg.norm(goal_np - start_np) < 0.8:
            continue
            
        # 确保直线发生碰撞 (Hard Case)
        straight_line = np.array([start_np, goal_np])
        if not is_trajectory_safe(straight_line, obs_centers_np, obs_types, obs_dims):
            return start_np, goal_np
    return None, None

@single_experiment_yaml
def generate_dataset(
    model_id: str = 'EnvSimple2D-RobotPointMass',
    num_cases: int = 100,
    device: str = 'cuda',
    seed: int = 42,
    results_dir: str = 'logs',
    **kwargs
):
    fix_random_seed(seed)
    device = get_torch_device(device)
    tensor_args = {'device': device, 'dtype': torch.float32}

    model_dir = os.path.join(TRAINED_MODELS_DIR, model_id)
    args = load_params_from_yaml(os.path.join(model_dir, "args.yaml"))
    train_subset, _, _, _ = get_dataset(dataset_class='TrajectoryDataset', use_extra_objects=True, obstacle_cutoff_margin=0.05, **args, tensor_args=tensor_args)
    robot, base_task = train_subset.dataset.robot, train_subset.dataset.task

    saved_cases = []
    print(f"🚀 开始生成 {num_cases} 个固定 Hard Cases...")

    for i in range(num_cases):
        env_seed = i * 100 # 给每个地图一个固定的种子
        dynamic_env = EnvSimple2DExtraObjects(tensor_args=tensor_args, drop_old_num=7, num_extra_spheres=5, num_extra_boxes=3, seed=env_seed)
        dynamic_task = type(base_task)(env=dynamic_env, robot=robot, tensor_args=tensor_args, obstacle_cutoff_margin=0.05)
        
        start_np, goal_np = generate_hard_case_start_goal(dynamic_env, dynamic_task, 2, tensor_args)
        if start_np is not None:
            saved_cases.append({
                'case_id': i,
                'env_seed': env_seed,
                'start_np': start_np,
                'goal_np': goal_np
            })
            print(f"✅ Case {i+1}/{num_cases} 生成成功")
        else:
            print(f"❌ Case {i+1}/{num_cases} 生成失败 (种子 {env_seed})，请调整参数")

    with open(CASES_SAVE_PATH, 'wb') as f:
        pickle.dump(saved_cases, f)
    print(f"\n🎉 完美！测试集已保存至: {CASES_SAVE_PATH}")

if __name__ == '__main__':
    run_experiment(generate_dataset)