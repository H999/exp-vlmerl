from typing import Any, Dict, List, Union

from einops import rearrange
import torch as th
from gymnasium import spaces
from numpy.typing import NDArray
from stable_baselines3.common.buffers import DictReplayBuffer


class CLIPReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination,
        )
        self.render_arrays: List[NDArray] = []

    def add(
        self,
        obs: NDArray,
        next_obs: NDArray,
        action: NDArray,
        reward: NDArray,
        done: NDArray,
        infos: List[Dict[str, Any]],
        use_array: bool = True
    ) -> None:
        super().add(
            obs,
            next_obs,
            action,
            reward,
            done,
            infos,
        )

        if use_array:
            assert len(self.render_arrays) < self.buffer_size
            if infos[0].get("render_array") is not None:
                self.render_arrays.append(infos[0]["render_array"])
            else:
                self.render_arrays.append(rearrange(obs['img'], "b c h w -> b h w c"))

    def clear_render_arrays(self) -> None:
        self.render_arrays = []
