from isaaclab.utils import configclass
import os

from my_robot_lab.assets.robots.my_robot_ import MY_ROBOT_CFG
from my_robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg


@configclass
class MyRobotEnvCfg(LocomotionVelocityEnvCfg):
    """Shared wheel-leg velocity configuration with the plane scene."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.prim_path = "{ENV_REGEX_NS}/Robot/chassis/.*"
        if os.environ.get("UZ05_DISABLE_BASE_COM", "0") == "1":
            self.events.base_com = None
