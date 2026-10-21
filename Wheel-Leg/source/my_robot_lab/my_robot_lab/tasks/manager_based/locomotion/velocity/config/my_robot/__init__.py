import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="MyRobot-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyRobotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotFlatPPORunnerCfg",
    },
)

gym.register(
    id="MyRobot-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MyRobotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotFlatPPORunnerCfg",
    },
)

for suffix, cfg in (("", "MyRobotGentleEnvCfg"), ("-Play", "MyRobotGentleEnvCfg_PLAY")):
    gym.register(
        id=f"MyRobot-Velocity-Gentle{suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.gentle_env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotGentlePPORunnerCfg",
        },
    )
