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
from evaluate import evaluate_policy, test_difficult, test_standard

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

    model_path = f"models/{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}.pt"
    model = config["model"].load(model_path, seed=config["test_seed"], device=config["device"])

    test_standard(model, config)
    test_difficult(model, config)
