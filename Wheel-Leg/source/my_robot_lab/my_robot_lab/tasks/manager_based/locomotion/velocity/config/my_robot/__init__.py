import gymnasium as gym

from . import agents

for task_suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"MyRobot-Velocity-Rough-Scan{task_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_scan_env_cfg:MyRobotRoughScanEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughScanPPORunnerCfg",
        },
    )

for task_suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"MyRobot-Velocity-Step-Course{task_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.step_course_env_cfg:MyRobotStepCourseEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotStepCoursePPORunnerCfg",
        },
    )

for task_suffix, cfg_suffix in (("", ""), ("-Play", "_PLAY")):
    gym.register(
        id=f"MyRobot-Velocity-Steps{task_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.steps_env_cfg:MyRobotStepsEnvCfg{cfg_suffix}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotStepsPPORunnerCfg",
        },
    )

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

gym.register(
    id="MyRobot-Velocity-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyRobotRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughPPORunnerCfg",
    },
)

gym.register(
    id="MyRobot-Velocity-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MyRobotRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyRobotRoughPPORunnerCfg",
    },
)
