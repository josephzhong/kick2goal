import numpy as np
import torch.nn
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from torch import nn
import torch as th
from typing import Dict, List, Tuple, Type, Union, Optional, Any
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp, MlpExtractor,
)
import collections
from functools import partial
from multiprocessing import Queue, Manager

import sys
sys.path.append("loss-of-plasticity")
from lop.algos.cbp_linear import CBPLinear


class MlpExtractorForVisualize(MlpExtractor):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def ___setup_model__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto"
    ) -> None:
        super().__init__(
            feature_dim,
            net_arch,
            activation_fn,
            device
        )

    def forward(self, features: th.Tensor, callback: BaseCallback = None) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features, callback), self.forward_critic(features, callback)

    def forward_actor(self, features: th.Tensor, callback: BaseCallback = None) -> th.Tensor:
        if callback is not None:
            return callback.on_policy_inference(self.policy_net, features)
        else:
            return self.policy_net(features)

    def forward_critic(self, features: th.Tensor, callback: BaseCallback = None) -> th.Tensor:
        if callback is not None:
            return callback.on_value_inference(self.value_net, features)
        else:
            return self.value_net(features)


def weight_histograms_conv2d(writer, step, weights, layer_number, model_name):
    weights_shape = weights.shape
    num_kernels = weights_shape[0]
    for k in range(num_kernels):
        flattened_weights = weights[k].flatten()
        tag = f"train/{model_name}_{layer_number}/kernel_{k}"
        writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms_linear(writer, step, weights, layer_number, model_name):
    flattened_weights = weights.flatten()
    tag = f"train/{model_name}_{layer_number}"
    writer.add_histogram(tag, flattened_weights, global_step=step, bins='tensorflow')


def weight_histograms(writer, step, model, model_name):
    # Iterate over all model layers
    if not isinstance(model, torch.nn.Sequential):
        layer = model
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight
            weight_histograms_conv2d(writer, step, weights, "last", model_name)
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            weight_histograms_linear(writer, step, weights, "last", model_name)
    else:
        for layer_number, layer in enumerate(model):
            # Compute weight histograms for appropriate layer
            if isinstance(layer, nn.Conv2d):
                weights = layer.weight
                weight_histograms_conv2d(writer, step, weights, layer_number, model_name)
            elif isinstance(layer, nn.Linear):
                weights = layer.weight
                weight_histograms_linear(writer, step, weights, layer_number, model_name)


def layer_weight_mean_var(logger, model, model_name):
    if not isinstance(model, torch.nn.Sequential):
        last_layer = model
        weight_max = torch.max(torch.abs(last_layer.weight))
        weight_mean = torch.mean(torch.abs(last_layer.weight) / weight_max).item()
        weight_var = torch.std(torch.abs(last_layer.weight) / weight_max).item()
        logger.record(f"train/{model_name}_last_mean", weight_mean)
        logger.record(f"train/{model_name}_last_var", weight_var)
    else:
        for layer_number, layer in enumerate(model):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weight_max = torch.max(torch.abs(layer.weight))
                weight_mean = torch.mean(torch.abs(layer.weight) / weight_max).item()
                weight_var = torch.std(torch.abs(layer.weight) / weight_max).item()
                logger.record(f"train/{model_name}_{layer_number}_mean", weight_mean)
                logger.record(f"train/{model_name}_{layer_number}_var", weight_var)


class LogWeight(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.mp_manager = Manager()
        self.policy_dormant_percentage_list = self.mp_manager.Queue()
        self.value_dormant_percentage_list = self.mp_manager.Queue()

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        logger = self.locals["self"].logger
        layer_weight_mean_var(logger, self.locals["self"].policy.mlp_extractor.policy_net, "policy")
        layer_weight_mean_var(logger, self.locals["self"].policy.mlp_extractor.value_net, "value-0")
        layer_weight_mean_var(logger, self.locals["self"].policy.action_net, "policy")
        layer_weight_mean_var(logger, self.locals["self"].policy.value_net, "value-0")
        dormant_temp_list = list()
        while self.policy_dormant_percentage_list.qsize() > 0:
            dormant_temp_list.append(self.policy_dormant_percentage_list.get())
        logger.record(f"train/policy_dormant_percentage", np.mean(dormant_temp_list))
        dormant_temp_list = list()
        while self.value_dormant_percentage_list.qsize() > 0:
            dormant_temp_list.append(self.value_dormant_percentage_list.get())
        logger.record(f"train/value_dormant_percentage", np.mean(dormant_temp_list))
        # for _format in logger.output_formats:
        #     if isinstance(_format, TensorBoardOutputFormat):
        #         writer = _format.writer
        #         weight_histograms(writer, self.locals["self"].num_timesteps,
        #                           self.locals["self"].policy.action_net, "policy")
        #         weight_histograms(writer, self.locals["self"].num_timesteps,
        #                           self.locals["self"].policy.value_net, "value-0")

    def on_policy_inference(self, model, model_input) -> th.Tensor:
        dormant_percentage, output = LogWeight.dormant_neurons_percentage(model, model_input)
        self.policy_dormant_percentage_list.put(dormant_percentage)
        return output

    def on_value_inference(self, model, model_input) -> th.Tensor:
        dormant_percentage, output = LogWeight.dormant_neurons_percentage(model, model_input)
        self.value_dormant_percentage_list.put(dormant_percentage)
        return output

    @staticmethod
    def dormant_neurons_percentage(model, model_input, tao=0.0):
        num_of_nodes, dormant_count = 0.0, 0.0
        for layer_number, layer in enumerate(model):
            output = layer(model_input)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
                num_of_nodes += output.shape[1]
                output_avg_by_node = torch.divide(torch.sum(output, dim=0), output.shape[0]).numpy(force=True)
                dormant_node_index = np.where(output_avg_by_node <= tao, np.ones(output_avg_by_node.shape),np.zeros(output_avg_by_node.shape))
                dormant_count += np.sum(dormant_node_index)
            model_input = output
        return dormant_count / num_of_nodes, model_input


class ActorCriticPolicyForVisualize(ActorCriticPolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        callback: BaseCallback = LogWeight,
        policy_cbp: bool = False,
        value_cbp: bool = False,
        replacement_rate: float = 10e-4,
        maturity_threshold: int = 1000,
        init: str = "default"
    ):
        self.callback = callback()
        self.policy_cbp = policy_cbp
        self.value_cbp = value_cbp
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.init = init
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )
        if self.policy_cbp:
            self.add_cbp(self.mlp_extractor.policy_net, self.action_net, replacement_rate, maturity_threshold, init)
        if self.value_cbp:
            self.add_cbp(self.mlp_extractor.value_net, self.value_net, replacement_rate, maturity_threshold, init)
            # self.policy_net.to(device=get_device(device))
            # self.value_net.to(device=get_device(device))

    def add_cbp(self, modele_list, next_layer, replacement_rate, maturity_threshold, init):
        cbp_layers = list()
        for layer_index, layer in enumerate(modele_list):
            if isinstance(layer, self.activation_fn):
                if layer_index < len(modele_list) - 1:
                    cbp_layer = CBPLinear(in_layer=modele_list[layer_index - 1], out_layer=modele_list[layer_index + 1],
                                          replacement_rate=replacement_rate,
                                          maturity_threshold=maturity_threshold, init=init)
                else:
                    cbp_layer = CBPLinear(in_layer=modele_list[layer_index - 1], out_layer=next_layer,
                                          replacement_rate=replacement_rate,
                                          maturity_threshold=maturity_threshold, init=init)
                cbp_layers.append([layer_index + 1 + len(cbp_layers), cbp_layer])
        for layer_index, cbp_layer in cbp_layers:
            modele_list.insert(layer_index, cbp_layer)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractorForVisualize(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device
        )

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param callback: callback when doing inference
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features, self.callback)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features, self.callback)
            latent_vf = self.mlp_extractor.forward_critic(vf_features, self.callback)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))

        for layer in self.mlp_extractor.policy_net:
            if isinstance(layer, nn.Linear) and not layer.weight.requires_grad:
                print(layer.weight)

        return actions, values, log_prob


if __name__ == "__main__":
    rewards = [100, 1000]
    x_range = np.arange(2)
    width = 0.2
    sac_goal = [0.949, 0.334]
    ppo_goal = [0.872, 0.914]
    sac_game_lens = [781.75, 8927.22]
    ppo_game_lens = [683.03, 757.52]
    fib, sub_axe = plt.subplots(1, 2)
    sub_axe[0].bar(x_range - width / 2, sac_goal, width, label="sac_goal")
    sub_axe[0].bar(x_range + width / 2, ppo_goal, width, label="ppo_goal")
    sub_axe[0].set_xticks(x_range)
    sub_axe[0].set_xticklabels(rewards)
    sub_axe[0].set_xlabel("goal_reward")
    sub_axe[0].set_ylabel("goal_ratio")
    sub_axe[0].set_ylim([0.0, 1.0])
    sub_axe[0].annotate(f"{sac_goal[0]:.3f}", xy=(x_range[0] - width / 2, sac_goal[0]), xytext=(x_range[0] - width, sac_goal[0] + 0.01))
    sub_axe[0].annotate(f"{sac_goal[1]:.3f}", xy=(x_range[1] - width / 2, sac_goal[1]), xytext=(x_range[1] - width, sac_goal[1] + 0.01))
    sub_axe[0].annotate(f"{ppo_goal[0]:.3f}", xy=(x_range[0] + width / 2, ppo_goal[0]), xytext=(x_range[0], ppo_goal[0] + 0.01))
    sub_axe[0].annotate(f"{ppo_goal[1]:.3f}", xy=(x_range[1] + width / 2, ppo_goal[1]), xytext=(x_range[1], ppo_goal[1] + 0.01))
    sub_axe[1].bar(x_range - width / 2, sac_game_lens, width, label="IQM of game length for SAC")
    sub_axe[1].bar(x_range + width / 2, ppo_game_lens, width, label="IQM of game length for PPO")
    sub_axe[1].set_xticks(x_range)
    sub_axe[1].set_xticklabels(rewards)
    sub_axe[1].set_xlabel("goal_reward")
    sub_axe[1].set_ylabel("game_length")
    sub_axe[1].annotate(f"{sac_game_lens[0]}", xy=(x_range[0] - width / 2, sac_game_lens[0]),
                        xytext=(x_range[0] - width, sac_game_lens[0] + 10))
    sub_axe[1].annotate(f"{sac_game_lens[1]}", xy=(x_range[1] - width / 2, sac_game_lens[1]),
                        xytext=(x_range[1] - width, sac_game_lens[1] + 10))
    sub_axe[1].annotate(f"{ppo_game_lens[0]}", xy=(x_range[0] + width / 2, ppo_game_lens[0]),
                        xytext=(x_range[0], ppo_game_lens[0] + 10))
    sub_axe[1].annotate(f"{ppo_game_lens[1]}", xy=(x_range[1] + width / 2, ppo_game_lens[1]),
                        xytext=(x_range[1], ppo_game_lens[1] + 10))
    sub_axe[0].legend()
    sub_axe[1].legend()
    plt.show()
