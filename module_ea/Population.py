import torch
import random

from copy import deepcopy
from einops import rearrange
from functools import partial
from datetime import datetime
from torch.nn import ModuleList 
from tensordict import TensorDict
from typing import List, Union, cast
from torch.distributed import ProcessGroup
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape

from configs.config import Config
from module_rl.clip_buffer import CLIPReplayBuffer
from module_rl.clip_rewarded_sac import CLIPRewardedSAC
from module_vlm.reward_model import CLIPReward
from model_geneticCNN.Individual import Individual
from utils.ea import get_inds_attr, prod_rewards, flat_gen, normalized_rank, revert_gen, roll, scale_eta, tsd_iter_along_axis0, js_div, update_gen

from .operators import collect_rollouts_with_compute_fitness, params_cxSimulatedBinaryBounded, params_mutPolynomialBounded


class Population:

    gbest: Individual
    individuals: List[Individual]
    individuals_params: Union[TensorDict, List[TensorDict]]
    hdist: Union[TensorDict, List[TensorDict]]

    def __init__(
            self,
            env: VecEnv,
            config: Config,
            reward_model: CLIPReward,
            reward_group: ProcessGroup=None,
            individual_kwargs: dict = dict(),
            individual_group: ProcessGroup=None,
        ):
        super(Population, self).__init__()
        self.env = env
        self.config = config
        self.device = get_device()
        self.reward_model = reward_model
        self.reward_group = reward_group
        self.worker_frames_tensor = torch.zeros(
            (config.reward.batch_size // config.rl.n_workers, *config.render_dim),
            dtype=torch.uint8,
        )
        self.replay_buffer = CLIPReplayBuffer(
            config.rl.buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
            n_envs=env.num_envs,
        )
        self.checkpoint_dir = str(self.config.checkpoints_path)
        self.population_size = self.config.ea.population_size
        individual_kwargs.setdefault("num_stages", config.ea.num_stages)
        individual_kwargs.setdefault("input_size", get_obs_shape(env.observation_space))
        individual_kwargs.setdefault("output_size", get_action_dim(env.action_space))
        self.individual_group = individual_group
        self.individual_kwargs = individual_kwargs
        self.individuals = ModuleList([Individual(**self.individual_kwargs) for _ in range(self.population_size)])
        self.individuals_params = TensorDict.from_modules(*self.individuals,lazy_stack=True,lock=False)
        self.coef_per = 0.5 # personal coefficient
        self.coef_cog = 0.3 # cognitive coefficient
        self.coef_soc = 0.2 # social coefficient
        self.crossover_eta = self.config.ea.crossover_eta
        self.mutation_eta = self.config.ea.mutation_eta
        self.gbest = None
        self.n_hdist = self.config.ea.distribution_history_length
        self.hdist = TensorDict({
            "arch": torch.zeros_like(flat_gen(self.individuals[0].Stages.gen)),
            "model": self.individuals_params[0].new_zeros()
        }).unsqueeze(0).repeat(self.n_hdist).cuda()

    def update_gbest(self, individual: Individual):
        if not self.gbest or individual.fitness > self.gbest.fitness:
            self.gbest = deepcopy(individual)
            self.gbest.pbest = None  # Don't track history update of pbest when was individual

    def compute_hdist(self, tensor_per, tensor_cog, tensor_soc):
        return (self.coef_per * tensor_per +
                self.coef_cog * tensor_cog +
                self.coef_soc * tensor_soc)

    @torch.no_grad()
    def update_hdist(self):
        "add current distribution into index 0 of history distributions on both arch & model"
        roll(self.hdist)
        self.individuals.cuda()
        norm_fitness = normalized_rank(get_inds_attr(self.individuals,"fitness",to_tensor=True))
        gbest_params = TensorDict.from_module(self.gbest).cuda()
        psbest = ModuleList(get_inds_attr(self.individuals, "pbest")).cuda()
        psbest_norm_fitness = normalized_rank(get_inds_attr(psbest, "fitness", to_tensor=True))
        psbest_params = TensorDict.from_modules(*psbest,lazy_stack=True,lock=False)
        # arch
        self.hdist[0].set_("arch", self.compute_hdist(
            prod_rewards(self.individuals, norm_fitness),
            prod_rewards(psbest, psbest_norm_fitness),
            flat_gen(self.gbest.Stages.gen)))
        # model
        tsd_iter_along_axis0(self.individuals_params, lambda k, v: self.compute_hdist(
            prod_rewards(v, norm_fitness),
            prod_rewards(psbest_params[k], psbest_norm_fitness),
            gbest_params[k]
        ), self.hdist[0]["model"], use_key=True)
        self.individuals.cpu()

    @torch.no_grad()
    def fitness_function(self):
        self.individuals.cpu()
        temp_buffs, inds_fitness = collect_rollouts_with_compute_fitness(
            env=self.env,
            individuals=self.individuals,
            individuals_params=self.individuals_params,
            individual_group=self.individual_group,
            individual_device_ids=self.config.ea.device_ids,
            individual_num_workers=self.config.ea.n_workers,
            compute_fitness_method=self.config.ea.compute_fitness_method,
            reward_group=self.reward_group,
            reward_model=self.reward_model,
            reward_num_workers=self.config.rl.n_workers,
            worker_frames_tensor=self.worker_frames_tensor,
            reward_batch_size=self.config.reward.batch_size,
        )
        for ind, ind_fitness, temp_buff_oard, temp_buff_infos in zip(
            self.individuals, inds_fitness, temp_buffs.exclude("infos").tolist(), temp_buffs["infos"].tolist()
        ):
            [self.replay_buffer.add(**oard, infos=infos, use_array=False) for oard, infos in zip(temp_buff_oard, temp_buff_infos)] # loop of each step in temp_buff
            ind.update_fitness(ind_fitness)
            self.update_gbest(ind)

    @torch.no_grad()
    def selection_function(self, n=4):
        self.individuals.cuda()
        # rand_tour = torch.rand(self.population_size, self.population_size).argsort(1)[:, torch.randperm(self.population_size)[:n]]
        rand_tour = torch.randperm(self.population_size)[:n].sort().values.unsqueeze(0)
        while rand_tour.size(0) < self.population_size:
            rand_indx = torch.rand(self.population_size, self.population_size).argsort(1)
            rand_tour = torch.cat((rand_tour, rand_indx[:, torch.randperm(self.population_size)[:n]].sort().values)).unique(dim=0)[:self.population_size]
        weights = normalized_rank(torch.arange(n))[get_inds_attr(self.individuals,"fitness",to_tensor=True)[rand_tour].argsort().argsort()]

        inds_indx = rand_tour[torch.arange(self.population_size), get_inds_attr(self.individuals,"pbest.fitness",to_tensor=True)[rand_tour].argmax(1)]
        new_individuals = ModuleList([deepcopy(self.individuals[idx]) for idx in inds_indx]).cuda()
        new_params = TensorDict.from_modules(*new_individuals,lazy_stack=True,lock=False)
        tsd_iter_along_axis0(self.individuals_params[rand_tour], lambda v: torch.vmap(prod_rewards)(v, weights), new_params)
        update_gen(new_individuals, torch.vmap(prod_rewards)(flat_gen(self.individuals).cuda()[rand_tour], weights).bernoulli())

        self.individuals        = new_individuals
        self.individuals_params = new_params
        self.individuals.cpu()

    @torch.no_grad()
    def crossover_function(self):
        self.individuals.cuda()
        def manual_swap(v:torch.Tensor, cxSimulatedBinaryBounded_args:dict=None, dim_swap:List[int]=[0]):
            """
            manual swap after run cxSimulatedBinaryBounded
            when optimize compute by split (N, ...) => (N//2, 2, ...) (assume N % 2 == 0)
            """
            vs = torch.stack((v[::2], v[1::2]), dim=1)
            if cxSimulatedBinaryBounded_args is None:
                cxSimulatedBinaryBounded_args = {
                    "low": (1 + torch.rand(1).to(vs.device) * 2e-3 - 1e-3) * vs.amin(dim=list(range(1,v.ndim+1))).view(-1, *[1]*(v.ndim-1)),
                    "up" : (1 + torch.rand(1).to(vs.device) * 2e-3 - 1e-3) * vs.amax(dim=list(range(1,v.ndim+1))).view(-1, *[1]*(v.ndim-1))
                }
            ts = torch.stack(params_cxSimulatedBinaryBounded(
                v[::2], v[1::2], always_run = True, eta = self.crossover_eta
                * scale_eta(torch.vmap(js_div)(vs)).view(-1, *[1]*(v.ndim-1)),
                **cxSimulatedBinaryBounded_args
            ), dim=1)
            return ts.where(torch.rand(torch.ones(ts.ndim, dtype=torch.int64).scatter(0,
                torch.arange(ts.ndim)[dim_swap], torch.tensor(ts.shape)[dim_swap]
            ).tolist(), device=ts.device) < 0.5, ts.flip(1)).flatten(0,1)

        tsd_iter_along_axis0(self.individuals_params, manual_swap)
        update_gen(self.individuals, manual_swap(flat_gen(self.individuals), {"low": 0., "up": 1.}, [0, -1]).bernoulli())
        self.individuals.cpu()

    @torch.no_grad()
    def mutation_function(self):
        self.individuals.cuda()
        flattened_gens = flat_gen(self.individuals).cuda()
        dist = self.hdist[0].new_zeros()
        hjsd = self.hdist.apply(partial(js_div,w="desc",reduction="per")).sum()
        pjsd = hjsd.new_zeros()
        pjsd['arch'] = js_div(flattened_gens)
        tsd_iter_along_axis0(self.individuals_params, js_div, pjsd['model'])
        tsd_iter_along_axis0(self.hdist, partial(prod_rewards, rewards=torch.arange(self.n_hdist).flip(0), norm=True), dist)

        hap_jsd = hjsd + pjsd

        self.individuals_params.apply_(
            lambda x, d, hap: params_mutPolynomialBounded(x,
                eta=self.mutation_eta * scale_eta((js_div(torch.stack((x, d))) + hap) / 3),
                low=(1 + torch.rand(1).to(x.device) * 2e-3 - 1e-3) * x.amin(),
                up =(1 + torch.rand(1).to(x.device) * 2e-3 - 1e-3) * x.amax()
            ).lerp(d, torch.rand(1).cuda() * hap / 2),
            dist['model'].expand(self.population_size),
            hap_jsd['model'].expand(self.population_size),
        )

        update_gen(self.individuals, params_mutPolynomialBounded(flattened_gens, eta=self.mutation_eta * scale_eta((
            torch.vmap(js_div)(torch.stack((flattened_gens, dist['arch'].expand(self.population_size,-1)), dim=1)) + hap_jsd['arch']
        ) / 3).unsqueeze(1), low=0., up=1.).lerp(dist['arch'], torch.rand(1).cuda() * hap_jsd['arch'] / 2).bernoulli())
        self.individuals.cpu()

    def run_RL(self):
        print("Setting up RL algorithm", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        cpbest = get_inds_attr(self.individuals,"fitness",to_tensor=True).argmax()
        self.individuals[cpbest].cuda()
        config = deepcopy(self.config)
        config.rl.policy_kwargs.update(dict(individual=self.individuals[cpbest]))
        algo = CLIPRewardedSAC(env=self.env, config=config, reward_model=self.reward_model, group=self.reward_group)
        algo.replay_buffer = self.replay_buffer
        print("Training RL algorithm", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        algo.learn(
            total_timesteps=config.rl.n_steps,
            **(
                dict(log_interval=config.logging.tensorboard_freq // config.rl.n_envs)
                if config.logging.tensorboard_freq
                else dict()
            ),
        )
        self.individuals_params[cpbest].zero_grad()
        self.replay_buffer = algo.replay_buffer
        self.individuals[cpbest].cpu()

    def run(self, generations=3):
        print(f"fitness {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
        self.fitness_function()
        for generation in range(generations):
            print(f"Start generation # {generation} ... {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
            if generation % self.config.ea.rl_freq == 0:
                print(f"run RL {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
                self.run_RL()
            print(f"update hdist {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            self.update_hdist()
            print(f"selection {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            self.selection_function()
            print(f"crossover {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            self.crossover_function()
            print(f"mutation {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            self.mutation_function()
            print(f"fitness {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}")
            self.fitness_function()
            print(f"End generation # {generation} ... {datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n")
            print("---------------------------------------------\n")
