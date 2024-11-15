from typing import Any, Dict, List, Optional, Tuple, Type, Union
# import torch
import torch as th
from gymnasium import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.sac.policies import Actor, SACPolicy
from stable_baselines3.common.type_aliases import Schedule
from multiprocessing import Queue, Manager

import sys
sys.path.append("loss-of-plasticity")
from lop.algos.cbp_linear import CBPLinear

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


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
    if not isinstance(model, th.nn.Sequential):
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
    if not isinstance(model, th.nn.Sequential):
        last_layer = model
        weight_max = th.max(th.abs(last_layer.weight))
        weight_mean = th.mean(th.abs(last_layer.weight) / weight_max).item()
        weight_var = th.std(th.abs(last_layer.weight) / weight_max).item()
        logger.record(f"train/{model_name}_last_mean", weight_mean)
        logger.record(f"train/{model_name}_last_var", weight_var)
    else:
        for layer_number, layer in enumerate(model):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                weight_max = th.max(th.abs(layer.weight))
                weight_mean = th.mean(th.abs(layer.weight) / weight_max).item()
                weight_var = th.std(th.abs(layer.weight) / weight_max).item()
                logger.record(f"train/{model_name}_{layer_number}_mean", weight_mean)
                logger.record(f"train/{model_name}_{layer_number}_var", weight_var)


class SACLogWeight(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.mp_manager = Manager()
        self.policy_dormant_percentage_list = self.mp_manager.Queue()
        self.value_dormant_percentage_list = self.mp_manager.Queue()

    def _on_step(self) -> bool:
        return super()._on_step()

    def _on_rollout_end(self) -> None:
        steps = self.locals["self"].num_timesteps
        if steps % 1024 == 0:
            logger = self.locals["self"].logger
            layer_weight_mean_var(logger, self.locals["self"].policy.actor.latent_pi, "policy")
            for index, q_net in enumerate(self.locals["self"].policy.critic.q_networks):
                layer_weight_mean_var(logger, q_net, f"value-{index}")
            layer_weight_mean_var(logger, self.locals["self"].policy.actor.mu, "policy")
            dormant_temp_list = list()
            while self.policy_dormant_percentage_list.qsize() > 0:
                dormant_temp_list.append(self.policy_dormant_percentage_list.get())
            if len(dormant_temp_list) > 0:
                logger.record(f"train/policy_dormant_percentage", np.mean(dormant_temp_list))
            dormant_temp_list = list()
            while self.value_dormant_percentage_list.qsize() > 0:
                dormant_temp_list.append(self.value_dormant_percentage_list.get())
            if len(dormant_temp_list) > 0:
                logger.record(f"train/value_dormant_percentage", np.mean(dormant_temp_list))
            # for _format in logger.output_formats:
            #     if isinstance(_format, TensorBoardOutputFormat):
            #         writer = _format.writer
            #         weight_histograms(writer, self.locals["self"].num_timesteps,
            #                           self.locals["self"].policy.actor, "policy")
            #         for index, q_net in enumerate(self.locals["self"].policy.critic.q_networks):
            #             weight_histograms(writer, self.locals["self"].num_timesteps,
            #                               q_net, f"value-{index}")

    def on_policy_inference(self, model, model_input) -> th.Tensor:
        dormant_percentage, output = SACLogWeight.dormant_neurons_percentage(model, model_input)
        self.policy_dormant_percentage_list.put(dormant_percentage)
        return output

    def on_value_inference(self, model, model_input) -> th.Tensor:
        dormant_percentage, output = SACLogWeight.dormant_neurons_percentage(model, model_input)
        self.value_dormant_percentage_list.put(dormant_percentage)
        return output

    @staticmethod
    def dormant_neurons_percentage(model, model_input, tao=0.0):
        num_of_nodes, dormant_count = 0.0, 0.0
        for layer_number, layer in enumerate(model):
            output = layer(model_input)
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Tanh) or isinstance(layer, nn.Sigmoid):
                num_of_nodes += output.shape[1]
                output_avg_by_node = th.divide(th.sum(output, dim=0), output.shape[0]).numpy(force=True)
                dormant_node_index = np.where(output_avg_by_node <= tao, np.ones(output_avg_by_node.shape), np.zeros(output_avg_by_node.shape))
                dormant_count += np.sum(dormant_node_index)
            model_input = output
        return dormant_count / num_of_nodes, model_input


class ActorForVisualize(Actor):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Box

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            clip_mean,
            normalize_images
        )

    def get_action_dist_params(self, obs: th.Tensor, callback: BaseCallback = None) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :param callback: for visualize nn metrics
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        if callback is not None:
            latent_pi = callback.on_policy_inference(self.latent_pi, features)
        else:
            latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False, callback: BaseCallback = None) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, callback)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)


class ContinuousCriticForVisualize(ContinuousCritic):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.

    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        callback: BaseCallback = None
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor
        )
        self.callback = callback

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        if self.callback is not None:
            return tuple(self.callback.on_value_inference(q_net, qvalue_input) for q_net in self.q_networks)
        else:
            return tuple(q_net(qvalue_input) for q_net in self.q_networks)


class SACPolicyForVisualize(SACPolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: ActorForVisualize
    critic: ContinuousCriticForVisualize
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        callback: BaseCallback = SACLogWeight,
        policy_cbp: bool = False,
        value_cbp: bool = False,
        replacement_rate: float = 10e-4,
        maturity_threshold: int = 1000,
        init: str = "default"
    ):
        self.callback = callback()
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )

        self.policy_cbp = policy_cbp
        self.value_cbp = value_cbp
        if self.policy_cbp:
            self.add_cbp(self.actor.latent_pi, self.actor.mu, replacement_rate, maturity_threshold, init)
            self.actor.latent_pi.to(self.device)
        if self.value_cbp:
            for q_net in self.critic.q_networks:
                self.add_cbp(q_net, None, replacement_rate, maturity_threshold, init)
                q_net.to(self.device)
            for q_net in self.critic_target.q_networks:
                self.add_cbp(q_net, None, replacement_rate, maturity_threshold, init)
                q_net.to(self.device)

    def add_cbp(self, modele_list, next_layer, replacement_rate, maturity_threshold, init):
        cbp_layers = list()
        for layer_index, layer in enumerate(modele_list):
            if isinstance(layer, self.activation_fn):
                if layer_index < len(modele_list) - 1:
                    cbp_layer = CBPLinear(in_layer=modele_list[layer_index - 1], out_layer=modele_list[layer_index + 1],
                                          replacement_rate=replacement_rate,
                                          maturity_threshold=maturity_threshold, init=init)
                    cbp_layers.append([layer_index + 1 + len(cbp_layers), cbp_layer])
                else:
                    if next_layer is not None:
                        cbp_layer = CBPLinear(in_layer=modele_list[layer_index - 1], out_layer=next_layer,
                                              replacement_rate=replacement_rate,
                                              maturity_threshold=maturity_threshold, init=init)
                        cbp_layers.append([layer_index + 1 + len(cbp_layers), cbp_layer])

        for layer_index, cbp_layer in cbp_layers:
            modele_list.insert(layer_index, cbp_layer)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor, target=False)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None, target=False)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None, target=True)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ActorForVisualize:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return ActorForVisualize(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None, target: bool = False) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        if target:
            return ContinuousCritic(**critic_kwargs).to(self.device)
        else:
            critic_kwargs["callback"] = self.callback
            return ContinuousCriticForVisualize(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic, inference=False)

    def _predict(self, observation: th.Tensor, deterministic: bool = False, inference: bool = True) -> th.Tensor:
        if inference:
            return self.actor(observation, deterministic, callback=self.callback)
        else:
            return self.actor(observation, deterministic, callback=None)
