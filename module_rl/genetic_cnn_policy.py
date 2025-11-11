from gymnasium import spaces
from model_geneticCNN import Individual
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.sac.policies import SACPolicy
from torch import nn
from typing import Any, Optional, Union

import gc
import torch as th


class GeneticCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
            self,
            observation_space,
            individual: Individual = None,
            features_dim=256
        ):
        super().__init__(observation_space, features_dim)
        self._features_dim = individual.features_dim
        self.img_input = individual.img_input
        self.Stages = individual.Stages
        self.flatten = individual.flatten

    def forward(self, obs):
        gc.collect()
        th.cuda.empty_cache()
        return Individual.fe(obs, self.flatten, self.img_input, self.Stages)


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GeneticCNNActor(BasePolicy):
    action_space: spaces.Box
    features_extractor: GeneticCNNFeatureExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        individual: Individual = None,
        activation_fn: type[nn.Module] = nn.ReLU,
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
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = individual.activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        self.latent_pi = individual.actor_net
        self.action_dist = individual.action_dist
        self.mu = individual.mu
        self.log_std = individual.log_std

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        features = self.extract_features(obs, self.features_extractor)
        return Individual.actor_get_action_dist_params(features, self.latent_pi, self.mu, self.log_std, self.use_sde)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return Individual.actor_forward(self.action_dist, mean_actions, log_std, kwargs, deterministic=deterministic, only_actions=True)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return Individual.actor_forward(self.action_dist, mean_actions, log_std, kwargs, only_actions=False)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self(observation, deterministic)


class GeneticCNNContinuousCritic(BaseModel):
    features_extractor: GeneticCNNFeatureExtractor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        individual: Individual = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.share_features_extractor = share_features_extractor
        assert n_critics == len(individual.q_networks), "n_critics must eq individual.q_networks"
        self.n_critics = n_critics
        self.q_networks = individual.q_networks
        for idx in range(n_critics):
            self.add_module(f"qf{idx}", self.q_networks[idx])

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, ...]:
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        return Individual.critic_forward(features, actions, self.q_networks)


class GeneticCNNSACPolicy(SACPolicy):
    actor: GeneticCNNActor
    critic: GeneticCNNContinuousCritic
    critic_target: GeneticCNNContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        learning_rate: float,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        individual: Individual = None,
        individual_kwargs: Optional[dict[str, Any]] = {},
        features_extractor_class: type[BaseFeaturesExtractor] = GeneticCNNFeatureExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [256, 256]
        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        individual_kwargs.update({
            "num_stages": (3, 5),
            "input_size": get_obs_shape(self.observation_space),
            "output_size": get_action_dim(action_space),
            "actor_arch": actor_arch,
            "critic_arch": critic_arch,
            "n_critics": n_critics,
            "share_features_extractor": share_features_extractor,
            "optimizer_class": self.optimizer_class,
            "optimizer_kwargs": self.optimizer_kwargs,
            "learning_rate": learning_rate,
            "use_sde": use_sde,
            "use_expln": use_expln,
            "log_std_init": log_std_init,
            "clip_mean": clip_mean,
            "normalize_images": normalize_images,
        })
        individual = individual if individual else Individual(**individual_kwargs)
        self.features_extractor_kwargs.update(dict(individual = individual))

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "individual": individual,
        }
        self.actor_kwargs = self.net_args.copy()

        sde_kwargs = {
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(individual)
        del individual

    def _build(self, individual: Individual) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = individual.actor_optimizer

        self.critic = self.make_critic(features_extractor=self.actor.features_extractor if self.share_features_extractor else None)

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = individual.critic_optimizer

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> GeneticCNNActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return GeneticCNNActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> GeneticCNNContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return GeneticCNNContinuousCritic(**critic_kwargs).to(self.device)
