import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

from stable_baselines3.common import type_aliases
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped, SubprocVecEnv
from collections import defaultdict

from envs.kick_to_goal_gym import KickToGoalGym
from metrics import interquartile_mean


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
    infor_keys: list = [],
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param infor_keys:
    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_infors = defaultdict(list)

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_infors = [defaultdict(float) for _ in range(n_envs)]
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                for infor_key in infor_keys:
                    current_infors[i][infor_key] += info[infor_key]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                        for infor_key in infor_keys:
                            episode_infors[infor_key].append(current_infors[i][infor_key])
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        for infor_key in infor_keys:
                            episode_infors[infor_key].append(current_infors[i][infor_key])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    for infor_key in infor_keys:
                        current_infors[i][infor_key] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_infors
    return mean_reward, std_reward

def validate(model, config):
    print("validate")
    validate_env = make_vec_env(KickToGoalGym, n_envs=config["eval_num_envs"],
                                vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"},
                                env_kwargs={"episode_length": 10000, "goal_reward": 1000})
    for env_idx in range(validate_env.num_envs):
        validate_env.env_method("reset_seed", seed=config["validate_seed"] + env_idx, indices=env_idx)
        # new_delta = callback.last_delta
        # eval_env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)

    # model = config["model"].load("models/53705_50.pt", seed=config["test_seed"], device=config["device"])
    # print("model loaded.")
    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, validate_env, n_eval_episodes=config["num_of_eval_episodes"],
                                            return_episode_rewards=True,
                                            infor_keys=["goal"], deterministic=True)
    print("validate result:")
    print(f"Number of validation episodes {len(rews)}")
    print(f"IQM of rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of rewards {np.mean(rews):.4f}")
    print(f"Std of rewards {np.std(rews):.4f}")
    print(f"IQM of game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal ratio {np.mean(infors['goal']):.4f}")

def test_difficult(model, config, save_rewards=False):
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

    if save_rewards:
        with open(
                f"{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}_{config['test_seed']}_difficult.rewards",
                "w") as f:
            f.write(",".join(
                [f"{rew:.2f}" for rew in rews]
            ) + "\n")
            f.write(",".join(
                [f"{goal}" for goal in infors['goal']]
            ) + "\n")
            f.write(",".join(
                [f"{infors['init_state_x'][index]:.2f};{infors['init_state_y'][index]:.2f}" for index in
                 range(len(infors['init_state_x']))]
            ) + "\n")

def test_standard(model, config, save_rewards=False):
    print("test standard")
    test_env_standard = make_vec_env(KickToGoalGym, n_envs=config["eval_num_envs"],
                                     vec_env_cls=SubprocVecEnv, vec_env_kwargs={"start_method": "fork"},
                                     env_kwargs={"episode_length": 10000, "goal_reward": 1000})
    for env_idx in range(test_env_standard.num_envs):
        test_env_standard.env_method("reset_seed", seed=config["test_seed"] + env_idx, indices=env_idx)
        # new_delta = callback.last_delta
        # eval_env.env_method("change_goal_position", new_delta=new_delta, indices=env_idx)

    # model = config["model"].load("models/53705_50.pt", seed=config["test_seed"], device=config["device"])
    # print("model loaded.")
    # models.exploration_final_eps = 0.01
    rews, lengths, infors = evaluate_policy(model, test_env_standard, n_eval_episodes=config["num_of_eval_episodes"],
                                            return_episode_rewards=True,
                                            infor_keys=["goal", "init_state_x", "init_state_y"], deterministic=True)
    print("standard test result:")
    print(f"Number of standard test episodes {len(rews)}")
    print(f"IQM of standard test rewards {interquartile_mean(rews):.4f}")
    print(f"Mean of standard test rewards {np.mean(rews):.4f}")
    print(f"Std of standard test rewards {np.std(rews):.4f}")
    print(f"IQM of standard test game_lens {interquartile_mean(lengths):.4f}")
    print(f"goal standard test ratio {np.mean(infors['goal']):.4f}")

    if save_rewards:
        with open(f"{config['train_seed']}_{config['policy_kwargs']['maturity_threshold']}_{config['test_seed']}_standard.rewards", "w") as f:
            f.write(",".join(
                [f"{rew:.2f}" for rew in rews]
            ) + "\n")
            f.write(",".join(
                [f"{goal}" for goal in infors['goal']]
            ) + "\n")
            f.write(",".join(
                [f"{infors['init_state_x'][index]:.2f};{infors['init_state_y'][index]:.2f}" for index in
                 range(len(infors['init_state_x']))]
            ) + "\n")