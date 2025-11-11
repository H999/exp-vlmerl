from __future__ import annotations
from multiprocessing import cpu_count

from environment.base import RENDER_DIM
from pydantic import (
    BaseModel,
    computed_field,
    Field,
    field_validator,
    model_validator,
    ValidationInfo,
)
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import json
import numpy as np
import pathlib
import torch
import utils

ncpu = cpu_count()
ngpu = torch.cuda.device_count()
# ncpu = 1
# ngpu = 1

class Config(BaseModel):
    env_name: Literal["Humanoid-v4", "MountainCarContinuous-v0"]
    render_dim: Annotated[
                    Optional[Union[int, Tuple[int], Tuple[int, int], Tuple[int, int, int]]],
                    Field(None, validate_default=True)
                ]
    base_path: pathlib.Path
    seed: int
    description: str
    tags: List[str]
    reward: Union[GroundTruthConfig, CLIPRewardConfig]
    ea: EAConfig
    rl: RLConfig
    logging: LoggingConfig

    # Auto-injected properties
    run_hash: str

    def save(self) -> None:
        with open(self.dump_path, "w") as f:
            json.dump(
                self.model_dump(), f, indent=2, cls=utils.PathlibCompatibleJSONEncoder
            )

    @computed_field
    @property
    def world_size(self) -> int:
        return min(ncpu, self.ea.population_size + self.rl.n_workers - 1)

    @computed_field
    @property
    def dump_path(self) -> pathlib.Path:
        return self.run_path / "run_config.json"

    @computed_field
    @property
    def checkpoints_path(self) -> pathlib.Path:
        return self.run_path / "checkpoints"

    @computed_field
    @property
    def is_clip_rewarded(self) -> bool:
        return isinstance(self.reward, CLIPRewardConfig)

    @computed_field
    @property
    def run_name(self) -> str:
        reward_str = "CLIP" if self.is_clip_rewarded else "GT"
        return f"{self.env_name[:-3]}_{reward_str}_{self.run_hash}"

    @computed_field
    @property
    def run_path(self) -> pathlib.Path:
        return (self.base_path / self.run_name).resolve()

    @computed_field
    @property
    def log_file(self) -> pathlib.Path:
        return self.run_path / "info.log"

    @computed_field
    @property
    def tb_dir(self) -> pathlib.Path:
        return self.run_path / "tensorboard"

    @field_validator("render_dim")  
    @classmethod
    def set_or_validate_render_dim(cls, v: Union[None, int, Tuple[int], Tuple[int, int], Tuple[int, int, int]], info: ValidationInfo) -> Tuple[int, int, int]:
        v = np.append(np.resize(RENDER_DIM[info.data["env_name"]] if v is None else v, 2), 3) if np.size(v) <= 2 else v
        assert len(v) == 3 and v[2] == 3, "render_dim must be int or tuple of 1-3 ints ending in 3"
        return tuple(v.tolist() if isinstance(v, np.ndarray) else v)

    @model_validator(mode="before")
    def configure_properties(cls, data: Any) -> Any:
        assert isinstance(data, dict)
        data["run_hash"] = utils.get_run_hash()
        if "base_path" not in data:
            data["base_path"] = pathlib.Path.cwd() / "runs/training"
        return data

    @model_validator(mode="after")
    def check_model(self) -> "Config":
        if self.logging.checkpoint_freq % self.rl.train_freq != 0:
            raise ValueError(
                f"({self.logging.checkpoint_freq=}) must be divisible by "
                f"({self.rl.train_freq=}). Otherwise duplicated checkpoints "
                "are created."
            )
        if self.logging.video_freq % self.rl.n_envs != 0:
            raise ValueError(
                f"({self.logging.video_freq=}) must be divisible by "
                f"({self.rl.n_envs=})"
            )
        if self.logging.checkpoint_freq % self.rl.n_envs != 0:
            raise ValueError(
                f"({self.logging.checkpoint_freq=}) must be divisible by "
                f"({self.rl.n_envs=})"
            )

        if self.is_clip_rewarded:
            assert isinstance(self.reward, CLIPRewardConfig)
            if self.logging.tensorboard_freq is not None:
                raise ValueError(
                    "When doing CLIP-rewarded training, a tensorboard logging "
                    "frequency does not need to be specified."
                )
            if len(self.reward.target_prompts) != len(self.reward.baseline_prompts):
                raise ValueError(
                    f"({self.reward.target_prompts=}) and "
                    f"({self.reward.baseline_prompts=}) must have the same length."
                )

            if len(self.reward.target_prompts) == 0:
                raise ValueError(f"({self.reward.target_prompts=}) must not be empty.")
            if self.rl.train_freq % self.rl.episode_length != 0:
                raise ValueError(
                    f"({self.rl.train_freq=}) must be divisible by "
                    f"({self.rl.episode_length=}), so that training happens after "
                    "full episodes are completed."
                )
            if self.reward.batch_size % self.rl.n_workers != 0:
                raise ValueError(
                    f"({self.reward.batch_size=}) corresponds to the total size of the "
                    " batch do be distributed among workers and therefore must be "
                    f"divisible by ({self.rl.n_workers=})"
                )
            if self.rl.n_envs * self.rl.episode_length % self.reward.batch_size != 0:
                raise ValueError(
                    f"({self.rl.n_envs=}) * ({self.rl.episode_length=}) must be "
                    f"divisible by ({self.reward.batch_size=}) so that all batches"
                    "are of the same size."
                )
        else:
            if self.logging.tensorboard_freq is None:
                raise ValueError(
                    "You must specify a tensorboard logging frequency when"
                    " training on ground-truth rewards."
                )

        return self


class GroundTruthConfig(BaseModel):
    name: Literal["ground_truth"]
    camera_config: Annotated[Optional[Dict[str, Any]], Field(None, validate_default=True)]

    @field_validator("camera_config")  
    @classmethod
    def set_or_validate_camera_config(cls, v: Union[None, Dict[str, Any]], info: ValidationInfo) -> Dict[str, Any]:
        return v if v else {
                        "trackbodyid": 1,
                        "distance": 3.5,
                        "lookat": [0.0, 0.0, 1.0],
                        "elevation": -10.0,
                        "azimuth": 180.0
                    }


class CLIPRewardConfig(BaseModel):
    name: Literal["clip"]
    pretrained_model: str
    batch_size: int
    alpha: float
    target_prompts: List[str]
    baseline_prompts: List[str]
    cache_dir: str
    camera_config: Annotated[Optional[Dict[str, Any]], Field(None, validate_default=True)]
    textured: bool = True

    @field_validator("camera_config")  
    @classmethod
    def set_or_validate_camera_config(cls, v: Union[None, Dict[str, Any]], info: ValidationInfo) -> Dict[str, Any]:
        return v if v else {
                        "trackbodyid": 1,
                        "distance": 3.5,
                        "lookat": [0.0, 0.0, 1.0],
                        "elevation": -10.0,
                        "azimuth": 180.0
                    }

    # @computed_field
    # @property
    # def embed_dim(self) -> int:
    #     use with model.config.vision_config.hidden_size for hf if need


class EAConfig(BaseModel):
    rl_freq: int = 10
    population_size: int = 16
    num_stages: Tuple[int, ...] = (3, 5)
    mutation_eta: Optional[float] = 20.
    crossover_eta: Optional[float] = 30.
    distribution_history_length: Optional[float] = 5
    individual_kwargs: Optional[Dict[str, Any]] = None
    compute_fitness_method: Literal["min", "max", "sum", "mean"] = "sum"

    @computed_field
    @property
    def n_workers(self) -> int:
        return min(ncpu - ngpu + 1, self.population_size)

    @computed_field
    @property
    def device_ids(self) -> List[int]:
        return [0] + list(range(ngpu, self.n_workers + ngpu - 1))



class RLConfig(BaseModel):
    # unuse
    policy_name: str = "MlpPolicy"
    n_steps: int
    n_envs_per_worker: int
    episode_length: int
    learning_starts: int
    train_freq: int
    batch_size: int
    gradient_steps: int
    action_noise: Optional[
        Union[NormalActionNoiseConfig, OrnsteinUhlenbeckActionNoiseConfig]
    ] = None
    tau: float = 0.005
    gamma: float = 0.99
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    ent_coef: Union[str, float] = "auto"
    use_sde: bool = False
    target_update_interval: int = 1
    policy_kwargs: Annotated[Optional[Dict[str, Any]], Field(None, validate_default=True)]
    rl_kwargs: Optional[Dict[str, Any]] = None

    @computed_field
    @property
    def n_workers(self) -> int:
        if ngpu == 0:
            raise RuntimeError("No CUDA device is found.")
        return ngpu

    @computed_field
    @property
    def device_ids(self) -> List[int]:
        return list(range(self.n_workers))

    @computed_field
    @property
    def n_envs(self) -> int:
        return self.n_workers * self.n_envs_per_worker

    @field_validator("policy_kwargs")  
    @classmethod
    def set_or_validate_policy_kwargs(cls, v: Union[None, Dict[str, Any]], info: ValidationInfo) -> Dict[str, Any]:
        if not v: v = dict()
        if "learning_rate" not in v:
            v.update({"learning_rate": info.data["learning_rate"]})
        return v

    @model_validator(mode="after")
    def check_model(self) -> "RLConfig":
        if self.train_freq > self.n_steps:
            raise ValueError(
                f"({self.train_freq=}) cannot be greater than "
                f"({self.n_steps=}), or no training would be performed."
            )
        return self


class LoggingConfig(BaseModel):
    checkpoint_freq: int
    video_freq: int
    tensorboard_freq: Optional[int] = None


class OrnsteinUhlenbeckActionNoiseConfig(BaseModel):
    name: Literal["OrnsteinUhlenbeckActionNoise"]
    mean: float
    sigma: float
    theta: float
    dt: float


class NormalActionNoiseConfig(BaseModel):
    name: Literal["NormalActionNoise"]
    mean: float
    sigma: float
