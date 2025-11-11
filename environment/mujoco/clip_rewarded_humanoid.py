from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv as GymHumanoidEnv
from gymnasium.envs.mujoco.mujoco_env import DEFAULT_SIZE
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as gymDict
from numpy.typing import NDArray
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pathlib


class CLIPRewardedHumanoidEnv(GymHumanoidEnv):
    def __init__(
        self,
        episode_length: int,
        render_mode: str = "rgb_array",
        forward_reward_weight: float = 1.25,
        ctrl_cost_weight: float = 0.1,
        healthy_reward: float = 5.0,
        healthy_z_range: Tuple[float] = (1.0, 2.0),
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        render_dim: Tuple[int, int, int] = (DEFAULT_SIZE, DEFAULT_SIZE, 3),
        camera_id: Optional[int] = -1,
        camera_config: Optional[Dict[str, Any]] = {
                                                    "trackbodyid": 1,
                                                    "distance": 3.5,
                                                    "lookat": [0.0, 0.0, 1.0],
                                                    "elevation": -10.0,
                                                    "azimuth": 180.0
                                                },
        textured: bool = True,
        **kwargs,
    ):
        terminate_when_unhealthy = False
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            render_mode=render_mode,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        humanoid_state_space = 376 if exclude_current_positions_from_observation else 378
        self.observation_space = gymDict({
            "vec": Box(low=-np.inf, high=np.inf, shape=(humanoid_state_space,), dtype=np.float64),
            "img": Box(low=0, high=255, shape=render_dim, dtype=np.uint8)
        })
        self.width, self.height, _ = render_dim

        env_file_name = None
        if textured:
            env_file_name = "humanoid_textured.xml"
        else:
            env_file_name = "humanoid.xml"
        model_path = str(pathlib.Path(__file__).parent / env_file_name)
        MujocoEnv.__init__(
            self,
            model_path,
            5,
            width=self.width,
            height=self.height,
            camera_id=camera_id,
            observation_space=self.observation_space,
            default_camera_config=camera_config,
            render_mode=render_mode,
            **kwargs,
        )
        self.episode_length = episode_length
        self.num_steps = 0

    def step(self, action) -> Tuple[NDArray, float, bool, bool, Dict]:
        humanoid_obs, reward, terminated, truncated, info = super().step(action)
        self.num_steps += 1
        terminated = self.num_steps >= self.episode_length
        image_obs = self.get_image_obs()
        return {"vec": humanoid_obs, "img": image_obs}, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.num_steps = 0
        humanoid_obs, info = super().reset(seed=seed, options=options)
        image_obs = self.get_image_obs()
        return {"vec": humanoid_obs, "img": image_obs}, info

    def get_image_obs(self):
        return self.render()
