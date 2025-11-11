
from datetime import datetime
from environment.base import get_clip_rewarded_env_name, get_make_env, is_3d_env
from module_rl.make_vec_env import make_vec_env
from module_rl.subproc_vec_env import SubprocVecEnv, SubprocVecEnvWrap
from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm
from sys import platform
from typing import Any, Dict, TYPE_CHECKING

import json
import os
import secrets

if TYPE_CHECKING:
    from configs.config import Config


def get_run_hash() -> str:
    return (
        f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_'
        f"{secrets.token_hex(4)}"
    )


def save_experiment_config(path: str, experiment_config: Dict) -> None:
    with open(path, "w") as f:
        json.dump(experiment_config, f, indent=2, cls=PathlibCompatibleJSONEncoder)


class PathlibCompatibleJSONEncoder(json.JSONEncoder):
    """JSON encoder that can handle pathlib objects."""

    def default(self, o: Any) -> Any:
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def set_egl_env_vars() -> None:
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
    os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
    os.environ["MUJOCO_GL"] = "glfw" if platform == "win32" else "egl"
    os.environ["PYOPENGL_PLATFORM"] = "glfw" if platform == "win32" else "egl"
    os.environ["EGL_PLATFORM"] = "device"


def make_vec_env_by_config(config: "Config"):
    make_env_kwargs = (
        dict(
            camera_config=config.reward.camera_config,
            textured=config.reward.textured,
            render_dim=config.render_dim,
        )
        if is_3d_env(config.env_name)
        else {}
    )
    if config.is_clip_rewarded:
        make_env_kwargs["episode_length"] = config.rl.episode_length
        env_name = get_clip_rewarded_env_name(config.env_name)
    else:
        make_env_kwargs["max_episode_steps"] = config.rl.episode_length
        env_name = config.env_name
    make_env_fn = get_make_env(env_name, **make_env_kwargs)
    vec_env = make_vec_env(
        make_env_fn,
        n_envs=config.rl.n_envs,
        seed=config.seed,
        vec_env_cls=SubprocVecEnvWrap,
        use_gpu_ids=config.rl.device_ids,
        vec_env_kwargs=dict(render_dim=config.render_dim),
    )
    vec_env = BaseAlgorithm._wrap_env(vec_env, True)
    return vec_env
