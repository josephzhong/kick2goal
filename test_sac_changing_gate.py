import os
import sys
from SAC_policy_visualize import SACPolicyForVisualize

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, EventCallback, BaseCallback
from stable_baselines3.sac.policies import SACPolicy
import numpy as np

import time
import datetime
from tempfile import gettempdir
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv
from evaluate import evaluate_policy

from envs.kick_to_goal_gym import KickToGoalGym
from metrics import interquartile_mean
import torch

class EveryNTimesteps(EventCallback):
    """
    Trigger a callback every ``n_steps`` timesteps

    :param n_steps: Number of timesteps between two trigger.
    :param callback: Callback that will be called
        when the event is triggered.
    """

    def __init__(self, n_steps: int, callback: BaseCallback):
        super().__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if (self.num_timesteps - self.last_time_trigger) >= self.n_steps:
            self.last_time_trigger = self.num_timesteps

            Goal_area_y_position_delta_min = \
                self.model.env.get_attr("Goal_area_y_position_delta_min", indices=0)[0]
            Goal_area_y_position_delta_max = \
                self.model.env.get_attr("Goal_area_y_position_delta_max", indices=0)[0]
            delta_delta = (Goal_area_y_position_delta_max - Goal_area_y_position_delta_min) / 10
            goal_area_y_position_delta = \
                self.model.env.get_attr("goal_area_y_position_delta", indices=0)[0]
            new_delta = goal_area_y_position_delta + delta_delta
            for env_idx in range(self.model.env.num_envs):
                # self.model.env.env_method("update_attribute", attribute_name="episode_length",
                #                           value=self.model.env.get_attr("episode_length", indices=env_idx)[0] + 1000, indices=env_idx)

                # reward_dict = self.model.env.get_attr("reward_dict", indices=env_idx)[0]
                # reward_dict["goal"] += 100
                # self.model.env.env_method("update_attribute", attribute_name="reward_dict",
                #                           value=reward_dict,
                #                           indices=env_idx)
                # init_ball_to_goal_angle_score = \
                # self.model.env.get_attr("init_ball_to_goal_angle_score", indices=env_idx)[0]
                # self.model.env.env_method("update_attribute", attribute_name="init_ball_to_goal_angle_score",
                #                           value=init_ball_to_goal_angle_score + 0.1, indices=env_idx)
                # init_ball_to_goal_distance_score = \
                # self.model.env.get_attr("init_ball_to_goal_distance_score", indices=env_idx)[0]
                # self.model.env.env_method("update_attribute", attribute_name="init_ball_to_goal_distance_score",
                #                           value=init_ball_to_goal_distance_score + 0.1, indices=env_idx)
                self.model.env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)
            self.last_delta = new_delta
            return self._on_event()
        return True


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        train_seed, test_seed = int(sys.argv[1]), int(sys.argv[2])
        print("load seeds from args")
    else:
        train_seed, test_seed = 53705, 50735
        print("load seeds from default")
    if len(sys.argv) >= 4:
        maturity_threshold = int(sys.argv[3])
    else:
        maturity_threshold = 50
    config = {
        "model": SAC,
        "policy": SACPolicyForVisualize,
        "device": "cpu",
        "total_timesteps": 3000000,
        "batch_size": 512,
        "train_num_envs": 8,
        "train_seed": train_seed,
        "test_seed": test_seed,
        "environment_change_timestep": 300000,
        "policy_kwargs": {
            "activation_fn": torch.nn.ReLU,
            "net_arch": [64, 64],
            "policy_cbp": False,
            "value_cbp": False,
            "replacement_rate": 1e-5,
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
        Goal_area_y_position_delta_min = \
            train_envs.get_attr("Goal_area_y_position_delta_min", indices=env_idx)[0]
        train_envs.env_method("change_goal_position", new_delta=Goal_area_y_position_delta_min, indices=env_idx)
    print(train_envs.get_attr("reward_dict", 0))
    params = [(config["batch_size"])]
    for batch_size in params:
        model = config["model"](policy=config["policy"], env=train_envs, batch_size=batch_size,
                                tensorboard_log="tb_log", device=config["device"],
                                seed=config["train_seed"], policy_kwargs=config["policy_kwargs"])
        callback = EveryNTimesteps(n_steps=config["environment_change_timestep"], callback=None)
        model.learn(total_timesteps=config["total_timesteps"],
                    callback=[model.policy.callback, callback],
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
        new_delta = callback.last_delta
        eval_env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)

    # model = config["model"].load("models/kick_2_goal_SAC_20241008012533.pt", seed=config["test_seed"], device=config["device"])
    # print("model loaded.")
    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, eval_env, n_eval_episodes=1024, return_episode_rewards=True,
                                            infor_keys=["goal"], deterministic=True)
    print(f"IQM of rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of rewards {np.mean(rews):.4f}")
    print(f"Std of rewards {np.std(rews):.4f}")
    print(f"IQM of game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal ratio {np.mean(infors['goal']):.4f}")

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

