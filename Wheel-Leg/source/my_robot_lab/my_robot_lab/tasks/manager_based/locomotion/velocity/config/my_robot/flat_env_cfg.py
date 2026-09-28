from isaaclab.utils import configclass

from .rough_env_cfg import MyRobotRoughEnvCfg


@configclass
class MyRobotFlatEnvCfg(MyRobotRoughEnvCfg):
    """Flat velocity environment configuration for the UZ-05 wheel-leg robot."""

    def __post_init__(self):
        super().__post_init__()

        # Inherit the shared reward tuning from RewardsCfg.

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class MyRobotFlatEnvCfg_PLAY(MyRobotFlatEnvCfg):
    """Play configuration for the UZ-05 wheel-leg flat velocity environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
