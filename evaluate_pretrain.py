import os
import sys

from stable_baselines3.common.policies import ActorCriticPolicy

from visualize import LogWeight, ActorCriticPolicyForVisualize

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, EventCallback, BaseCallback
import numpy as np
import torch

import time
import datetime
from tempfile import gettempdir
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, unwrap_vec_normalize
from evaluate import evaluate_policy

from envs.kick_to_goal_gym import KickToGoalGym
from metrics import interquartile_mean


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        train_seed, maturity_threshold, test_seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        print("load seeds from args")
    else:
        train_seed, maturity_threshold, test_seed = 53705, 50, 50735
        print("load seeds from default")
    config = {
        "model": PPO,
        "policy": ActorCriticPolicyForVisualize,
        # "policy": ActorCriticPolicy,
        "device": "cpu",
        "total_timesteps": 10000000,
        "batch_size": 512,
        "n_steps": 1024,
        "train_num_envs": 8,
        "eval_num_envs": 8,
        "train_seed": train_seed,
        "test_seed": test_seed,
        # "validate_seed": validate_seed,
        "num_of_eval_episodes": 10240,
        "environment_change_timestep": 1010000,
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "policy_cbp": False,
            "value_cbp": False,
            "replacement_rate": 1e-5,
            "maturity_threshold": maturity_threshold,
            "init": "default"
        }
    }
    print(config)

    print("test standard")
    test_env_standard = make_vec_env(KickToGoalGym, n_envs=config["eval_num_envs"],
                                     vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"},
                                     env_kwargs={"episode_length": 10000, "goal_reward": 1000})
    for env_idx in range(test_env_standard.num_envs):
        test_env_standard.env_method("reset_seed", seed=config["test_seed"] + env_idx, indices=env_idx)
        # new_delta = callback.last_delta
        # eval_env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)

    model_path = f"models/{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}.pt"
    model = config["model"].load(model_path, seed=config["test_seed"], device=config["device"])
    print(f"model loaded {model_path}.")
    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, test_env_standard, n_eval_episodes=config["num_of_eval_episodes"],
                                            return_episode_rewards=True,
                                            infor_keys=["goal", "init_state_x", "init_state_y"], deterministic=True)
    print("test result:")
    print(f"Number of test episodes {len(rews)}")
    print(f"IQM of test rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of test rewards {np.mean(rews):.4f}")
    print(f"Std of test rewards {np.std(rews):.4f}")
    print(f"IQM of test game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal test ratio {np.mean(infors['goal']):.4f}")

    with open(f"{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}_{config['test_seed']}_standard.rewards", "w") as f:
        f.write(",".join([f"{rew:.2f}" for rew in rews]) + "\n")
        f.write(",".join([f"{goal}" for goal in infors['goal']])+ "\n")
        f.write(",".join([f"{infors['init_state_x'][index]:.2f};{infors['init_state_y'][index]:.2f}" for index in range(len(infors['init_state_x']))]) + "\n")

    print("test difficult")
    test_env_difficult = make_vec_env(KickToGoalGym, n_envs=config["eval_num_envs"],
                                      vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"},
                                      env_kwargs={"episode_length": 10000,
                                                  "varying_init_state": True,
                                                  "goal_reward": 1000})
    for env_idx in range(test_env_difficult.num_envs):
        test_env_difficult.env_method("reset_seed", seed=config["test_seed"] + env_idx, indices=env_idx)
        test_env_difficult.env_method("update_attribute", attribute_name="init_ball_to_goal_distance_score",
                                      value=0.95, indices=env_idx)
        # new_delta = callback.last_delta
        # eval_env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)

    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, test_env_difficult, n_eval_episodes=config["num_of_eval_episodes"],
                                            return_episode_rewards=True,
                                            infor_keys=["goal", "init_state_x", "init_state_y"], deterministic=True)
    print("difficult test result:")
    print(f"Number of difficult test episodes {len(rews)}")
    print(f"IQM of difficult test rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of difficult test rewards {np.mean(rews):.4f}")
    print(f"Std of difficult test rewards {np.std(rews):.4f}")
    print(f"IQM of difficult test game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal difficult test ratio {np.mean(infors['goal']):.4f}")

    with open(f"{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}_{config['test_seed']}_difficult.rewards", "w") as f:
        f.write(",".join([f"{rew:.2f}" for rew in rews]) + "\n")
        f.write(",".join([f"{goal}" for goal in infors['goal']]) + "\n")
        f.write(",".join([f"{infors['init_state_x'][index]:.2f};{infors['init_state_y'][index]:.2f}" for index in range(len(infors['init_state_x']))]) + "\n")
