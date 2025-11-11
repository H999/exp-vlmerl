import gc
import copy
import torch
import random
import warnings
import torch.nn as nn
from itertools import chain

from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution

from typing import TYPE_CHECKING, Union
if TYPE_CHECKING:
    from module_rl.genetic_cnn_policy import GeneticCNNActor, GeneticCNNContinuousCritic

from .Stages import Stages

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Individual(torch.nn.Module):
    Stages: "Stages"
    action_dist: Union[StateDependentNoiseDistribution, SquashedDiagGaussianDistribution]

    def __init__(
            self,
            # geneticCNN params
            num_stages=None,
            gen=None,
            input_size=None, # in this concept is obs shape
            output_size=17, # in this concept is action space dim 
            input_chanel=256,
            output_chanel=256,
            kernel_size=5,
            # SAC model pure init params
            actor_arch=[256, 256],
            critic_arch=[256, 256],
            n_critics=2,
            share_features_extractor=True,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs={},
            learning_rate=3e-4,
            use_sde=False,
            full_std=True,
            use_expln=False,
            log_std_init=-3,
            clip_mean=2.0,
            normalize_images=True,
            # SAC model convert init params
            actor:"GeneticCNNActor"=None,
            critic:"GeneticCNNContinuousCritic"=None,
        ):
        super(Individual, self).__init__()

        if not (actor and critic):
            if actor or critic:
                warnings.warn("Only support convert when have both actor and critic, now create new", UserWarning)
            
            # features_extractor
            self.img_input = nn.Conv2d(input_size["img"][0], input_chanel, 3, padding="same")
            self.Stages = Stages(
                random.randint(2, 10), input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size
            ) if num_stages is None and gen is None else Stages(
                num_stages, gen, input_chanel=input_chanel, output_chanel=output_chanel, kernel_size=kernel_size
            )
            ## sb3 default using FlattenExtractor for vec obs
            self.flatten = nn.Flatten()

            # Compute features_extractor output shape by doing one forward pass
            with torch.no_grad():
                fake_img_obs = torch.rand((1, *input_size["img"])).cuda()
                self.img_input.cuda()
                self.Stages.cuda()
                self.flatten.cuda()
                img_feat = self.img_input(fake_img_obs)
                img_feat = self.Stages(img_feat)
                img_feat = self.flatten(img_feat)
                self.img_input.cpu()
                self.Stages.cpu()
                self.flatten.cpu()
                self.features_dim = img_feat.shape[1] + input_size["vec"][0]
                del img_feat
                del fake_img_obs

            features_extractor_parameters = chain(self.img_input.parameters(), self.Stages.parameters())

            # actor net action is squashed
            self.activation_fn = [nn.ReLU(), nn.Dropout(),]
            self.actor_net = nn.Sequential(*[item for layer in create_mlp(self.features_dim, -1, actor_arch) for item in (self.activation_fn if isinstance(layer, nn.ReLU) else [layer])])
            last_layer_dim = actor_arch[-1] if len(actor_arch) > 0 else self.features_dim
            self.use_sde = use_sde
            if self.use_sde:
                self.action_dist = StateDependentNoiseDistribution(
                    output_size, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
                )
                self.mu, self.log_std = self.action_dist.proba_distribution_net(
                    latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
                )
                if clip_mean > 0.0:
                    self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
            else:
                self.action_dist = SquashedDiagGaussianDistribution(output_size)  # type: ignore[assignment]
                self.mu = nn.Linear(last_layer_dim, output_size)
                self.log_std = nn.Linear(last_layer_dim, output_size)  # type: ignore[assignment]

            self.actor_optimizer = optimizer_class(
                chain(
                    features_extractor_parameters,
                    self.actor_net.parameters(),
                    self.mu.parameters(),
                    self.log_std.parameters(),
                ),
                lr=learning_rate,  # type: ignore[call-arg]
                **optimizer_kwargs,
            )

            # critic net
            self.q_networks = []
            for idx in range(n_critics):
                q_net = nn.Sequential(*[item for layer in create_mlp(self.features_dim + output_size, 1, critic_arch) for item in (self.activation_fn if isinstance(layer, nn.ReLU) else [layer])])
                self.add_module(f"qf{idx}", q_net)
                self.q_networks.append(q_net)

            # this assume only share_features_extractor = True when using with SAC
            self.critic_optimizer = optimizer_class(
                chain(
                    *[] if share_features_extractor else features_extractor_parameters,
                    *[q_net.parameters() for q_net in self.q_networks]
                ),
                lr=learning_rate,  # type: ignore[call-arg]
                **optimizer_kwargs,
            )

            self.output_size = output_size
            self.normalize_images = normalize_images
        else:
            warnings.warn("when actor and critic passed other params useless", UserWarning)
            if not critic.share_features_extractor:
                warnings.warn("If not share the features extractor between the actor and the critic then use the features extractor of the actor for individual", UserWarning)

            # features_extractor
            self.img_input = actor.features_extractor.img_input
            self.Stages = actor.features_extractor.Stages
            self.flatten = actor.features_extractor.flatten
            self.features_dim = actor.features_extractor.features_dim

            # actor net
            self.activation_fn = actor.activation_fn
            self.actor_net = actor.latent_pi
            self.use_sde = actor.use_sde
            self.action_dist = actor.action_dist
            self.mu = actor.mu
            self.log_std = actor.log_std

            self.actor_optimizer = actor.optimizer

            # critic net
            self.q_networks = critic.q_networks
            for idx in range(critic.n_critics):
                self.add_module(f"qf{idx}", self.q_networks[idx])
            
            self.critic_optimizer = critic.optimizer

            self.output_size = actor.mu.out_features
            self.normalize_images = actor.normalize_images

        self.pbest = [None] # disable register with list wrap
        self.fitness = None

        if actor:
            del actor
        if critic:
            del critic

    def update_fitness(self, fitness):
        self.fitness = fitness
        if not self.pbest[0] or fitness > self.pbest[0].fitness:
            self.pbest = [None] # keep only best of alltime, del it for track history update
            self.pbest = [copy.deepcopy(self).cpu()]

    @staticmethod
    def fe(obs, flatten, img_input, Stages):
        vec_obs = obs["vec"]
        img_obs = obs["img"]

        vec_feat = flatten(vec_obs)
        img_feat = img_input(img_obs)
        img_feat = Stages(img_feat)
        img_feat = flatten(img_feat)

        return torch.cat([img_feat, vec_feat], dim=1)

    @staticmethod
    def actor_get_action_dist_params(features, net, mu, log_std, use_sde):
        latent_pi = net(features)
        mean_actions = mu(latent_pi)

        if use_sde:
            return mean_actions, log_std, dict(latent_sde=latent_pi)
        log_std = log_std(latent_pi)  # type: ignore[operator]
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    @staticmethod
    def actor_forward(action_dist, mean_actions, log_std, action_dist_kwargs, deterministic=False, only_actions=True):
        """
        :param only_actions: True : only get actions
                             False: get actions with log
        """
        if only_actions:
            return action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **action_dist_kwargs)
        else:
            return action_dist.log_prob_from_params(mean_actions, log_std, **action_dist_kwargs)

    @staticmethod
    def critic_forward(features, actions, q_networks):
        qvalue_input = torch.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in q_networks)

    def forward(self, obs, observation_space, only_actions=False):
        gc.collect()
        torch.cuda.empty_cache()

        # obs features prepare
        if not isinstance(obs, torch.Tensor): obs = obs_as_tensor(obs, self.img_input.weight.device) 
        obs = preprocess_obs(obs, observation_space, normalize_images=self.normalize_images)
        features = self.fe(obs, self.flatten, self.img_input, self.Stages)

        # actor
        mean_actions, log_std, kwargs = self.actor_get_action_dist_params(features, self.actor_net, self.mu, self.log_std, self.use_sde)
        actions = self.actor_forward(self.action_dist, mean_actions, log_std, kwargs, only_actions=only_actions)
        if only_actions: return actions
        actions, log_prob = actions

        # critic
        q_values = self.critic_forward(features, actions, self.q_networks)
        q_values = torch.min(torch.cat(q_values, dim=1), dim=1, keepdim=True)

        return actions, log_prob, q_values
