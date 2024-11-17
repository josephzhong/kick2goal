import os
import sys
from stable_baselines3.common.policies import ActorCriticPolicy

from visualize import LogWeight, ActorCriticPolicyForVisualize

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import numpy as np
import torch

import time
import datetime
from tempfile import gettempdir
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from evaluate import evaluate_policy

from envs.kick_to_goal_gym import KickToGoalGym
from metrics import interquartile_mean


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        train_seed, test_seed, maturity_threshold = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
        print("load seeds from args")
    else:
        train_seed, test_seed, maturity_threshold = 53705, 50735, 50
        print("load seeds from default")
    config = {
        "model": PPO,
        "policy": ActorCriticPolicyForVisualize,
        # "policy": ActorCriticPolicy,
        "device": "cpu",
        "total_timesteps": 3000000,
        "batch_size": 512,
        "n_steps": 1024,
        "train_num_envs": 8,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "policy_cbp": True,
            "value_cbp": True,
            "replacement_rate": 10e-4,
            "maturity_threshold": maturity_threshold,
            "init": "default"
        }
    }
    print(config)

    print("train")
    train_envs = make_vec_env(KickToGoalGym, n_envs=config["train_num_envs"], vec_env_cls=SubprocVecEnv,
                              vec_env_kwargs={"start_method": "fork"}, env_kwargs={"seed": config["train_seed"]})
    for env_idx in range(train_envs.num_envs):
        train_envs.env_method("reset_seed", seed=config["train_seed"] + env_idx, indices=env_idx)
    print(train_envs.get_attr("reward_dict", 0))
    params = [(config["batch_size"], config["n_steps"])]
    for batch_size, n_step in params:
        model = config["model"](policy=config["policy"], env=train_envs, batch_size=batch_size, n_steps=n_step,
                                tensorboard_log="tb_log", vf_coef=0.05, ent_coef=0.01, device=config["device"],
                                seed=config["train_seed"], policy_kwargs=config["policy_kwargs"])
        model.learn(total_timesteps=config["total_timesteps"],
                    callback=model.policy.callback,
                    progress_bar=True)
        time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # save_path = f"models/kick_2_goal_{config['model'].__name__}_{time_stamp}.pt"
        save_path = f"models/{str(train_seed)}_{str(maturity_threshold)}.pt"
        model.save(save_path)
        print("saved models to path {0}".format(save_path))

    print("test")
    eval_env = make_vec_env(KickToGoalGym, n_envs=8, vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"},
                            env_kwargs={"episode_length": 10000})
    for env_idx in range(eval_env.num_envs):
        eval_env.env_method("reset_seed", seed=config["test_seed"] + env_idx, indices=env_idx)

    # model = config["model"].load("models/kick_2_goal_PPO_20241019200113.pt", seed=config["test_seed"], device=config["device"])
    # print("model loaded.")
    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, eval_env, n_eval_episodes=1024, return_episode_rewards=True,
                                            infor_keys=["goal"], deterministic=True)
    print(f"IQM of rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of rewards {np.mean(rews):.4f}")
    print(f"Std of rewards {np.std(rews):.4f}")
    print(f"IQM of game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal ratio {np.mean(infors['goal']):.4f}")

    # render
    # env = KickToGoalGym(seed=config["test_seed"])
    # obs, _ = env.reset()
    # env.render()
    # values = list()
    # value = 0.0
    # game_len = 0
    # while True:
    #     # By default, deterministic=False, so we use the stochastic policy
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, _, infors = env.step(action)
    #     env.render()
    #     value += rewards
    #     game_len += 1
    #     time.sleep(1.0 / 30 / 100)
    #     if dones:
    #         print("len: {0:.4f}, value: {1:.4f}, goal: {2}".format(game_len, value, infors["goal"]))
    #         # model.logger.record("eval/value", value)
    #         # model.logger.record("eval/game_length", game_len)
    #         # model.logger.dump(step=len(values))
    #         value = 0.0
    #         game_len = 0
    #         obs, _ = env.reset()
    #         env.render()
    # env.close()
