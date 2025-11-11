import gymnasium

from .base import *  # noqa F401
from .base import get_clip_rewarded_env_name


gymnasium.register(
    get_clip_rewarded_env_name("MountainCarContinuous-v0"),
    "environment.classic_control.clip_rewarded_mountain_car_continuous:CLIPRewardedContinuousMountainCarEnv",  # noqa: E501
)

gymnasium.register(
    get_clip_rewarded_env_name("Humanoid-v4"),
    "environment.mujoco.clip_rewarded_humanoid:CLIPRewardedHumanoidEnv",
)
