from datetime import datetime
from operator import methodcaller
from typing import Optional, Tuple, overload

import torch
import torch.distributed as dist
import torch.nn as nn

from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from configs.config import CLIPRewardConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)


class CLIPEmbed(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    @torch.inference_mode()
    def forward(self, x):
        if x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)

        with torch.no_grad(), torch.autocast("cuda:0", enabled=torch.cuda.is_available()):
            x = self.model.get_image_features(self.processor(images=x, padding="max_length", max_length=64, return_tensors="pt")['pixel_values'])
            x = x / x.norm(dim=-1, keepdim=True)
        return x


class CLIPReward(nn.Module):
    def __init__(
        self,
        *,
        model: CLIPEmbed,
        alpha: float,
        target_prompts: str,
        baseline_prompts: str,
    ) -> None:
        super().__init__()
        self.embed_module = model
        target = self.embed_prompts(target_prompts).mean(dim=0, keepdim=True)
        baseline = self.embed_prompts(baseline_prompts).mean(dim=0, keepdim=True)
        direction = target - baseline

        # Register them as buffers so they are automatically moved around.
        self.register_buffer("target", target)
        self.register_buffer("baseline", baseline)
        self.register_buffer("direction", direction)

        self.alpha = alpha
        projection = self.compute_projection(alpha)
        self.register_buffer("projection", projection)

    def compute_projection(self, alpha: float) -> torch.Tensor:
        projection = self.direction.T @ self.direction / torch.norm(self.direction) ** 2
        identity = torch.diag(torch.ones(projection.shape[0])).to(projection.device)
        projection = alpha * projection + (1 - alpha) * identity
        return projection

    def update_alpha(self, alpha: float) -> None:
        self.alpha = alpha
        self.projection = self.compute_projection(alpha)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / torch.norm(x, dim=-1, keepdim=True)
        y = 1 - (torch.norm((x - self.target) @ self.projection, dim=-1) ** 2) / 2
        return y

    def embed_prompts(self, x) -> torch.Tensor:
        """Embed a list of prompts."""
        with torch.no_grad():
            x = self.embed_module.model.get_text_features(**self.embed_module.processor(text=x, padding="max_length", max_length=64, return_tensors="pt").to(self.embed_module.model.device))
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def embed_images(self, x):
        return self.embed_module.forward(x)


def load_reward_model(
    model_name, target_prompts, baseline_prompts, alpha, cache_dir
):
    model = AutoModel.from_pretrained(model_name, device_map="cuda:0", torch_dtype=torch.bfloat16, attn_implementation="sdpa", cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True, cache_dir=cache_dir)

    model = CLIPEmbed(model, processor)
    model = CLIPReward(
        model=model,
        alpha=alpha,
        target_prompts=target_prompts,
        baseline_prompts=baseline_prompts,
    )
    return model.eval()


def load_reward_model_from_config(config: CLIPRewardConfig) -> CLIPReward:
    return load_reward_model(
        model_name=config.pretrained_model,
        target_prompts=config.target_prompts,
        baseline_prompts=config.baseline_prompts,
        alpha=config.alpha,
        cache_dir=config.cache_dir,
    )

def compute_rewards(
    model: CLIPReward,
    frames: torch.Tensor,
    batch_size: int,
    num_workers: int,
    worker_frames_tensor=None,
    group: dist.ProcessGroup=None,
) -> torch.Tensor:
    assert frames.device == torch.device("cpu")
    assert batch_size % num_workers == 0
    n_samples = len(frames)
    rewards = torch.zeros(n_samples, device=torch.device("cpu"))
    model = model.eval()
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            frames_batch = frames[i : i + batch_size]
            rewards_batch = dist_worker_compute_reward(
                rank=0,
                group=group,
                reward_model=model,
                render_dim=frames_batch.shape[1:],
                batch_size=batch_size // num_workers,
                num_workers=num_workers,
                frames=frames_batch,
                worker_frames_tensor=worker_frames_tensor,
            )
            rewards_batch = rewards_batch.cpu()
            rewards[i : i + batch_size] = rewards_batch 
    return rewards


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: torch.Tensor,
) -> torch.Tensor:
    ...


@overload
def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames: None = None,
) -> None:
    ...


def dist_worker_compute_reward(
    rank: int,
    reward_model: CLIPReward,
    render_dim: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    frames=None,
    worker_frames_tensor=None,
    group: dist.ProcessGroup=None,
) -> Optional[torch.Tensor]:
    if rank == 0:
        if frames is None:
            raise ValueError("Must pass render result on rank=0")
        if len(frames) != num_workers * batch_size:
            raise ValueError("Must pass render result with correct batch size")
        scatter_list = list(map(methodcaller('cuda'), torch.chunk(frames, num_workers, dim=0)))
    else:
        scatter_list = []

    worker_frames = (worker_frames_tensor if worker_frames_tensor is not None else torch.zeros((batch_size, *render_dim), dtype=torch.uint8)).cuda()

    dist.scatter(worker_frames, scatter_list=scatter_list, src=0, group=group)
    with torch.no_grad():
        embeddings = reward_model.embed_module(worker_frames)
        rewards = reward_model(embeddings)

    def zero_t():
        return torch.zeros_like(rewards)
    recv_rewards = [zero_t() for _ in range(num_workers)] if rank == 0 else []
    dist.gather(rewards, gather_list=recv_rewards, dst=0, group=group)
    if rank == 0:
        return torch.cat(recv_rewards, dim=0).cuda(rank)
