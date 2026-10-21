from isaaclab.utils import configclass

from .base_env_cfg import MyRobotEnvCfg


@configclass
class MyRobotFlatEnvCfg(MyRobotEnvCfg):
    """Flat velocity environment configuration for the UZ-05 wheel-leg robot."""

    def __post_init__(self):
        super().__post_init__()

        # Inherit the shared reward tuning from RewardsCfg.

        # Use only the plane collision surface, without generated obstacles.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

@configclass
class MyRobotFlatEnvCfg_PLAY(MyRobotFlatEnvCfg):
    """Play configuration for the UZ-05 wheel-leg flat velocity environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-12.0, 12.0)
        self.commands.base_height.height_range = (0.25, 0.4)
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
