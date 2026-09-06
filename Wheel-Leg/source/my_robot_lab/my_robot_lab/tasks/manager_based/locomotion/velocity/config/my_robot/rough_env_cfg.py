from isaaclab.utils import configclass

from my_robot_lab.assets.robots.my_robot_ import MY_ROBOT_CFG
from my_robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg


@configclass
class MyRobotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Rough velocity environment configuration for the UZ-05 wheel-leg robot."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/chassis/.*"
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/chassis/chassis"


@configclass
class MyRobotRoughEnvCfg_PLAY(MyRobotRoughEnvCfg):
    """Play configuration for the UZ-05 wheel-leg velocity environment."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
