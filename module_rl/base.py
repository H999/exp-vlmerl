from stable_baselines3.common.base_class import BaseAlgorithm

from .clip_rewarded_sac import CLIPRewardedSAC


def get_clip_rewarded_rl_algorithm_class(env_name: str) -> BaseAlgorithm:
    if env_name in ["Humanoid-v4", "MountainCarContinuous-v0"]:
        return CLIPRewardedSAC
    else:
        raise ValueError(f"Unknown environment: {env_name}")
