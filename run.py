import os
import time
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

import supersuit as ss
import argparse
import importlib

from wandb.integration.sb3 import WandbCallback
import wandb

import torch

"""
Important options for training
- batch size
- num policies
- opponent deterministic
- change of random opponent
"""


def parse_args():
    parser = argparse.ArgumentParser()

    # Running options
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--render", action="store_true", default=False)

    parser.add_argument("--train_steps", type=int, default=3000000)

    # Env options
    parser.add_argument("--env", type=str, default="kick_to_goal_gym")

    # Training options
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--vec_envs", type=int, default=16)
    parser.add_argument("--num_cpus", type=int, default=8)

    parser.add_argument("--curriculum_method", type=str, default="close")
    parser.add_argument("--policy_path", type=str, default=None)

    # Wandb options
    parser.add_argument("--wandb", action="store_true", default=False)

    # Render options
    parser.add_argument("--agents", type=str, default=None)
    # Joseph: add Apple Silicon device
    parser.add_argument("--device", type=str, default="mps")

    if not parser.parse_args().env:
        raise ValueError("Must specify env")

    return parser.parse_args()


class LargeMlpPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(LargeMlpPolicy, self).__init__(
            *args, **kwargs, net_arch=dict(pi=[1024, 1024, 1024], vf=[1024, 1024, 1024])
        )


def train(args):
    # Init policy directory
    path = "envs." + args.env
    env = importlib.import_module(path).env

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device

    env = env()
    # env = ss.pettingzoo_env_to_vec_env_v1(env)

    # env = ss.concat_vec_envs_v1(
    #     env,
    #     num_vec_envs=args.vec_envs,
    #     num_cpus=args.num_cpus,
    #     base_class="stable_baselines3",
    # )
    env

    # env = VecMonitor(env)
    # env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10)

    run_name = args.env + "_" + time.strftime("%d_%H_%M_%S")
    if args.wandb:
        run = wandb.init(
            project=run_name,
            config={
                "env": args.env,
            },
            sync_tensorboard=True,
            name=run_name,
        )

    PPO.policy_aliases["LargeMlp"] = LargeMlpPolicy

    if not args.policy_path:
        model = PPO(
            MlpPolicy,
            env,
            batch_size=args.batch_size,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=f"./runs/{run_name}",
            device=device,
        )
    else:
        model = PPO.load(args.policy_path, env=env, device=device)

    if args.wandb:
        model.learn(total_timesteps=args.train_steps, callback=WandbCallback(verbose=1))
    else:
        model.learn(total_timesteps=args.train_steps)

    if not args.policy_path:
        model.save("policies/" + args.env + "_policy")
        print("Finished running")
        print("Saved policy to policies/" + args.env + "_policy")
    else:
        model.save("policies/" + args.env + "_curriculum_" + args.curriculum_method)
        print("Finished running")
        print("Saved policy to policies/" + args.env + "_curriculum_" + args.curriculum_method)

    env.close()

    
    exit()


def render(args):
    if not args.policy_path:
        policy_path = "policies/" + args.env + "_policy.zip"
    else:
        policy_path = args.policy_path

    path = "envs." + args.env
    env = importlib.import_module(path).env

    # Create environment
    env = env(curriculum_method=args.curriculum_method)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    PPO.policy_aliases["LargeMlp"] = LargeMlpPolicy
    model = PPO.load(policy_path)

    # Run models
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = env.step(action)
        env.render()


if __name__ == "__main__":
    args = parse_args()
    args.vec_envs = 4
    args.num_cpus = 2

    if args.train:
        train(args)
    elif args.render:
        render(args)
