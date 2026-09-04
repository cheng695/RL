"""Gymnasium task registration for wheel-legged Isaac Lab environments."""

import gymnasium as gym

from .wheel_leg_env_cfg import WheelLegEnvCfg, WheelLegPPORunnerCfg


gym.register(
    id="WheelLeg-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.wheel_leg_env_cfg:WheelLegPPORunnerCfg",
    },
    disable_env_checker=True,
)

__all__ = [
    "WheelLegEnvCfg",
    "WheelLegPPORunnerCfg",
]
