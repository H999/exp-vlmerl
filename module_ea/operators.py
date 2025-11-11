from collections.abc import Sequence as SequenceABC
from datetime import datetime, timedelta
from einops import rearrange, reduce
from functools import partial
from itertools import cycle
from model_geneticCNN import Individual
from module_vlm.reward_model import CLIPReward, compute_rewards
from stable_baselines3.common.vec_env import VecEnv
from tensordict import TensorDict
from typing import List, Literal, Optional, Sequence, Union
from utils.ea import close_together, serialize_model

import numpy as np
import random
import torch
import torch.distributed as dist

epl, epu = 1e-7, 1e7
BOUND_LOW, BOUND_UP = -1., 1.

@torch.no_grad()
def mutPolynomialBounded(
    individual: Union[torch.nn.Module, Individual],
    eta: float = 20.,
    low: Union[float, int, Sequence[float], Sequence[int]] = BOUND_LOW,
    up: Union[float, int, Sequence[float], Sequence[int]] = BOUND_UP,
) -> Union[torch.nn.Module, Individual]:
    """
    - base on original code from https://github.com/DEAP/deap/blob/master/deap/tools/mutation.py
    - mod for in-place change parameters of torch.module
    - handle indpb outside instead

    :param individual: :term:`Individual <individual>` to be mutated.
    :param eta (optional): Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different (default 20. as origin paper).
    :param low (optional): A value or a :term:`python:sequence` of values that
                is the lower bound of the search space (default -10.).
    :param up (optional): A value or a :term:`python:sequence` of values that
               is the upper bound of the search space (default 10.).
    :returns: Individual.
    """
    low = [low] if not isinstance(low, SequenceABC) else low
    up = [up] if not isinstance(up, SequenceABC) else up

    for p, xl, xu in zip(individual.parameters(), cycle(low), cycle(up)):
        p.copy_(params_mutPolynomialBounded(p, eta, xl, xu))
    return individual

@torch.no_grad()
def params_mutPolynomialBounded(
    ts: Union[torch.Tensor, TensorDict],
    eta: float = 20.,
    low: float = BOUND_LOW,
    up: float = BOUND_UP,
) -> Union[torch.Tensor, TensorDict]:
    ts = ts.clamp(min=low+epl, max=up-epl)
    rand = torch.rand_like(ts)
    condition = rand < 0.5
    condition_sign   = 1. - 2. * condition  # True -> -1, False -> 1
    condition_revert = (~condition).float() # True ->  0, False -> 1

    xy  = (rand - 0.5) * (1. - (ts - low).lerp(up - ts, condition_revert) / (  up - low)).pow(       eta + 1. )
    val =       1.     - (2. * ( condition_revert -       condition_sign  * (rand - xy))).pow(1.0 / (eta + 1.))
    delta_q = condition_sign * val
    return (ts + delta_q * (up - low)).clamp(min=low+epl, max=up-epl)

@torch.no_grad()
def cxSimulatedBinaryBounded(
    ind1: Individual,
    ind2: Individual,
    eta: float = 30.,
    low: Union[float, int, Sequence[float], Sequence[int]] = BOUND_LOW,
    up: Union[float, int, Sequence[float], Sequence[int]] = BOUND_UP,
    p: float = 0.5,
) -> tuple[Individual, Individual]:
    """
    - base on original code from https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    - mod for in-place change parameters of torch.module

    :param ind1: The first :term:`Individual <individual>` participating in the crossover.
    :param ind2: The second :term:`Individual <individual>` participating in the crossover.
    :param eta (optional): Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different (default 30. as origin paper).
    :param low (optional): A value or a :term:`python:sequence` of values that is the lower
                bound of the search space (default -10.).
    :param up (optional): A value or a :term:`python:sequence` of values that is the upper
               bound of the search space (default 10.).
    :param p (optional): A value that is the probability for swap (default 0.5).
    :returns: A tuple of two individuals.
    """
    low = [low] if not isinstance(low, SequenceABC) else low
    up = [up] if not isinstance(up, SequenceABC) else up

    for p1, p2, xl, xu in zip(ind1.parameters(), ind2.parameters(), cycle(low), cycle(up)):
        c1, c2 = params_cxSimulatedBinaryBounded(p1, p2, eta, xl, xu, p)
        p1.copy_(c1)
        p2.copy_(c2)

    return ind1, ind2

@torch.no_grad()
def params_cxSimulatedBinaryBounded(
    ts1: Union[torch.Tensor, TensorDict],
    ts2: Union[torch.Tensor, TensorDict],
    eta: float = 30.,
    low: float = BOUND_LOW,
    up: float = BOUND_UP,
    p: float = 0.5,
    always_run: bool = False,
) -> tuple[Union[torch.Tensor, TensorDict], Union[torch.Tensor, TensorDict]]:
    if not close_together(ts1, ts2) or always_run:
        rand = torch.rand_like(ts1)
        x1 = ts1.minimum(ts2).clamp(min=low+epl, max=up-epl)
        x2 = ts1.maximum(ts2).clamp(min=low+epl, max=up-epl)

        def cal_c(beta_numerator: Union[torch.Tensor, TensorDict], beta_q_sign: Literal[-1, 1]) -> Union[torch.Tensor, TensorDict]:
            beta = 1.0 + (2.0 * beta_numerator / (x2 - x1).clamp(min=epl, max=epu)).clamp(min=epl, max=epu)
            alpha = 2.0 - beta.pow(-(eta + 1)).clamp(min=epl, max=1 - epl)
            beta_q = (ra := rand * alpha).lerp(1. / (2. - ra), (ra > 1.).float()).pow(1.0 / (eta + 1))
            return (0.5 * (ts1 + ts2 + beta_q_sign * beta_q * (ts1 - ts2))).clamp(min=low+epl, max=up-epl)

        c1 = cal_c(x1 - low, 1)
        c2 = cal_c(up - x2, -1)
        return (c2, c1) if torch.rand(1) <= p else (c1, c2)
    return (ts2, ts1) if torch.rand(1) <= p else (ts1, ts2)

def collect_rollouts_with_compute_fitness(
    env: VecEnv,
    reward_model: CLIPReward,
    reward_batch_size: int,
    reward_num_workers: int,
    individuals: torch.nn.ModuleList,
    individual_num_workers: int,
    worker_frames_tensor=None,
    reward_group: dist.ProcessGroup=None,
    compute_fitness_method: str = "sum",
    individuals_params: TensorDict = None,
    individual_device_ids: List[int]=None,
    individual_group: dist.ProcessGroup=None,
) -> tuple[TensorDict, torch.Tensor]:
    if individual_device_ids is None:
        individual_device_ids = list(range(individual_num_workers))

    # collect_rollouts
    temp_buff = None
    scatter_list = TensorDict({
        "arch": [list(map(partial(int, base=2), m.Stages.gen)) for m in individuals],
        "model": individuals_params if individuals_params is None else TensorDict.from_modules(*individuals, lazy_stack=True).cpu()
    }, batch_size=[len(individuals)]).split(individual_num_workers)
    prim_rank_model_index = 0
    for chunk_models in scatter_list:
        if individual_num_workers > 1:
            [model.send(i, group=individual_group) for i, model in zip(individual_device_ids[1:], chunk_models[1:])]
        chunk_temp_buff = dist_worker_collect_rollouts(
            rank=0,
            group=individual_group,
            env=env,
            num_workers=individual_num_workers,
            individual=individuals[prim_rank_model_index],
        )
        if temp_buff is None:
            temp_buff = chunk_temp_buff
        else:
            temp_buff = TensorDict.cat([temp_buff, chunk_temp_buff], dim=0)
        prim_rank_model_index += individual_num_workers

    # compute rewards
    print("[Primary] start compute rewards & fitness", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    frames = rearrange(temp_buff["obs"]["img"], "pop_size n_steps n_envs c ... -> (pop_size n_steps n_envs) ... c")
    rewards = compute_rewards(
        model=reward_model,
        frames=frames,
        batch_size=reward_batch_size,
        group=reward_group,
        num_workers=reward_num_workers,
        worker_frames_tensor=worker_frames_tensor,
    )
    temp_buff["reward"] = rearrange(rewards, "(pop_size n_steps n_envs) -> pop_size n_steps n_envs",
        pop_size=len(individuals),
        n_envs=env.num_envs,
    )

    # compute fitness
    inds_fitness = reduce(rewards, "(pop_size n_steps n_envs) -> pop_size", compute_fitness_method, 
        pop_size=len(individuals),
        n_envs=env.num_envs,
    )

    print("[Primary] done compute rewards & fitness", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    return temp_buff, inds_fitness

def dist_worker_collect_rollouts(
    rank: int,
    env: VecEnv,
    num_workers: int,
    individual: Individual,
    tsd: TensorDict = None,
    group: dist.ProcessGroup=None,
) -> Optional[torch.Tensor]:
    if rank != 0:
        if tsd is None:
            raise ValueError("Must pass TensorDict:tsd on rank != 0")
        tsd.recv(0, group=group)
        tsd["model"].to_module(individual)
        individual.Stages.update_stages_state(tuple(
            np.vectorize(np.binary_repr)(
                tsd["arch"], width=list(map(len,individual.Stages.gen))
            ).tolist()
        ))
        print(f"[Worker {rank}] recv model {individual.Stages.gen}", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

    print(f"[{rank}] start collect", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    individual.cuda()
    temp_buff = []
    obs = env.reset()
    for _ in range(200):
        action = individual(obs, env.observation_space, only_actions=True).detach().cpu().numpy()
        next_obs, _, done, info = env.step(action)
        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "next_obs": next_obs,
                "done": done,
                "infos": TensorDict.lazy_stack(list(map(TensorDict,info))).exclude("episode","terminal_observation"),
            },
            batch_size=[]
        )
        temp_buff.append(td)
        obs = next_obs
    temp_buff = TensorDict.stack(temp_buff)
    print(f"[{rank}] done collect", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    individual.cpu()

    if rank == 0:
        rep_temp_buff = temp_buff.clone()
        temp_buff = temp_buff.expand(num_workers-1, *temp_buff.batch_size).contiguous()
    if num_workers > 1:
        temp_buff.gather_and_stack(0, group=group)
    if rank == 0:
        print("[Primary] recv all temp_buff", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        return TensorDict.stack([rep_temp_buff, *temp_buff.unbind(0)], dim=0)


if __name__ == "__main__":
    print("""
          Test for operators
=================================================
          init individuals
          """)
    individual_kwargs ={
        "num_stages": (3, 5),
        "input_size": {"img":(3,16,16),"vec":(367,)},
    }
    inds = [Individual(**individual_kwargs).cuda() for _ in range(2)]
    print("""
                    done
==================================================
          init individual tensordict
          """)
    inds_tsd = TensorDict.from_modules(*inds,lazy_stack=True,lock=False).requires_grad_(False)
    print(f"""
                    done
==================================================
          test mutPolynomialBounded
--------------------------------------------------
    test with def mutPolynomialBounded 
--------------------------------------------------
                  befor run
            {inds[0].mu.bias}
          """)
    mutPolynomialBounded(inds[0])
    print(f"""
                  after run
            {inds[0].mu.bias}
--------------------------------------------------
    test with def params_mutPolynomialBounded 
--------------------------------------------------
                  befor run
            {inds_tsd[0]['mu','bias']}
          """)
    inds_tsd[0]=params_mutPolynomialBounded(inds_tsd[0])
    print(f"""
                  after run
            {inds_tsd[0]['mu','bias']}
==================================================
          test cxSimulatedBinaryBounded
--------------------------------------------------
    test with def cxSimulatedBinaryBounded 
--------------------------------------------------
                  befor run
                    ind1
            {inds[0].mu.bias}
                    ind2
            {inds[1].mu.bias}
          """)
    cxSimulatedBinaryBounded(inds[0], inds[1])
    print(f"""
                  after run
                    ind1
            {inds[0].mu.bias}
                    ind2
            {inds[1].mu.bias}
--------------------------------------------------
    test with def params_cxSimulatedBinaryBounded 
--------------------------------------------------
                  befor run
            {inds_tsd['mu','bias']}
          """)
    inds_tsd[0], inds_tsd[1] = params_cxSimulatedBinaryBounded(inds_tsd[0], inds_tsd[1])
    print(f"""
                  after run
            {inds_tsd['mu','bias']}
          """)
