from __future__ import annotations
from functools import partial
import threading

from einops import rearrange, reduce
import numpy as np

from configs.config import CLIPRewardConfig, Config
from datetime import datetime, timedelta
from model_geneticCNN import Individual
from module_ea import Population
from module_ea.operators import collect_rollouts_with_compute_fitness, dist_worker_collect_rollouts
from module_vlm.reward_model import compute_rewards, dist_worker_compute_reward, load_reward_model_from_config
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from utils import multiprocessing
from utils.helpers import make_vec_env_by_config, set_egl_env_vars
from tensordict import TensorDict

from model_geneticCNN import Individual
# from utils import multiprocessing as mp
from utils.ea import deserialize_model, serialize_model

import os
import torch as th
import torch.distributed as dist
import tensordict
import yaml
import sys

# script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
# log_path = f"{script_name}.txt"
# open(log_path, "w").close()
# sys.stdout = open(log_path, "a")
# sys.stderr = open(log_path, "a")


def primary_worker(
    config: Config,
    reward_group: dist.ProcessGroup,
    individual_group: dist.ProcessGroup,
    stop_event = None,
):
    print("[Primary] Creating environment instance", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    th.cuda.manual_seed(config.seed)
    reward_model = load_reward_model_from_config(config.reward).cuda()
    env = make_vec_env_by_config(config)
    individual_kwargs = {
        "num_stages": config.ea.num_stages,
        "input_size": get_obs_shape(env.observation_space),
        "output_size": get_action_dim(env.action_space),
    }
    p=Population(
        env=env,
        config=config,
        reward_model=reward_model,
        reward_group=reward_group,
        individual_group=individual_group,
        individual_kwargs=individual_kwargs,
    )
    p.run(50)
    if stop_event is not None:
        stop_event.set()
    print("done", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
def individual_worker(rank: int, config: Config, group: dist.ProcessGroup, stop_event):
    print(f"[Worker {rank}] Creating environment instance", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    env = make_vec_env_by_config(config)
    individual_kwargs = {
        "num_stages": config.ea.num_stages,
        "input_size": get_obs_shape(env.observation_space),
        "output_size": get_action_dim(env.action_space),
    }
    individual = Individual(**individual_kwargs).cpu()
    tsd = TensorDict({
        "arch": list(map(partial(int, base=2), individual.Stages.gen)),
        "model": TensorDict.from_module(individual)
    })
    while not stop_event.is_set():
        print(f"[Worker {rank}] Entering wait for collect_rollouts...", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        dist_worker_collect_rollouts(
            rank=rank,
            group=group,
            env=env,
            num_workers=config.ea.n_workers,
            individual=individual,
            tsd=tsd,
        )
    print(f"[Worker {rank}] Received stop event. Exiting worker", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
def clip_inference_worker(rank: int, config: Config, group: dist.ProcessGroup, stop_event):
    assert isinstance(config.reward, CLIPRewardConfig)
    assert config.reward.batch_size % config.rl.n_workers == 0
    print(f"[Worker {rank}] Loading CLIP model....", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    reward_model = load_reward_model_from_config(config.reward).eval().cuda()
    worker_frames_tensor = th.zeros(
        (config.reward.batch_size // config.rl.n_workers, *config.render_dim),
        dtype=th.uint8,
    )
    while not stop_event.is_set():
        print(f"[Worker {rank}] Entering wait for compute_embeddings_dist...", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        dist_worker_compute_reward(
            rank,
            group=group,
            reward_model=reward_model,
            render_dim=config.render_dim,
            batch_size=config.reward.batch_size // config.rl.n_workers,
            num_workers=config.rl.n_workers,
            worker_frames_tensor=worker_frames_tensor,
        )
    print(f"[Worker {rank}] Received stop event. Exiting worker", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))

def init_process(
    rank: int,
    stop_event, /,
    config: Config,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    set_egl_env_vars()
    tensordict.set_list_to_stack(True).set()
    th.cuda.set_device(rank % config.rl.n_workers)
    dist.init_process_group("cuda:nccl,cpu:gloo", rank=rank, world_size=config.world_size
                            # , timeout=timedelta(seconds=50)
                        )
    # dist.init_process_group("cuda:nccl,cpu:gloo", rank=rank, world_size=1)
    reward_group = dist.new_group(config.rl.device_ids)
    individual_group = dist.new_group(config.ea.device_ids)

    if rank == 0:
        primary_worker(config, reward_group, individual_group, stop_event)
    elif rank < config.rl.n_workers:
        clip_inference_worker(rank, config, reward_group, stop_event)
    else:
        individual_worker(rank, config, individual_group, stop_event)

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    print("start", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
    with open("configs/testeapopwithenv.yaml") as stream:
        try:
            config = Config(**yaml.load(stream, Loader=yaml.FullLoader))
            print("load config done", datetime.now().strftime("%d/%m/%Y, %H:%M:%S"))
        except yaml.YAMLError as exc:
            print(exc)
    print(config.render_dim)
    print(f"Started run with id {config.run_name}")
    config.run_path.mkdir(parents=True, exist_ok=True)
    print("Logging experiment metadata")
    config.save()

    multiprocessing.spawn(
        fn=init_process,
        args=(config,),
        nprocs=config.world_size,
        # nprocs=1,
        join=True,
        daemon=False,
        start_method="spawn",
    )

