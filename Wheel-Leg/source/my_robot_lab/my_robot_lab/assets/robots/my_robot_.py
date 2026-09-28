from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .mjcf import spawn_closed_loop_mjcf


ROBOT_MJCF_DIR = Path(__file__).parent / "UZ05_MJCF_real_params"
ROBOT_MJCF_PATH = ROBOT_MJCF_DIR / "xmls" / "uz05_isaac.xml"

INITIAL_BASE_HEIGHT_RANGE = (0.25, 0.4)
INITIAL_BASE_HEIGHT = 0.30

LEG_JOINTS = [
    "left_J_chassis_link2",
    "left_J_chassis_link4",
    "right_J_chassis_link2",
    "right_J_chassis_link4",
]

WHEEL_JOINTS = [
    "left_J_link6_wheel",
    "right_J_link6_wheel",
]

PASSIVE_JOINTS = [
    "left_J_link2_link5",
    "left_J_link5_link3",
    "left_J_link5_link1",
    "left_J_link1_link6",
    "right_J_link2_link5",
    "right_J_link5_link3",
    "right_J_link5_link1",
    "right_J_link1_link6",
]

CONTROLLED_JOINTS = LEG_JOINTS + WHEEL_JOINTS
ALL_JOINTS = CONTROLLED_JOINTS + PASSIVE_JOINTS

BASE_BODY_NAME = "chassis"
BASE_CONTACT_BODY_NAMES = [
    "chassis",
]
WHEEL_BODY_NAMES = [
    "left_right_wheel",
    "right_right_wheel",
]

LEG_EXTENSION_BODY_NAMES = [
    "chassis",
    "left_right_wheel",
    "right_right_wheel",
]


MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        func=spawn_closed_loop_mjcf,
        asset_path=str(ROBOT_MJCF_PATH),
        force_usd_conversion=True,  # Converter also reads inertials from uz05.xml.
        import_sites=True,
        fix_base=False,
        make_instanceable=False,
        self_collision=False,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, INITIAL_BASE_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.97,
    actuators={
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=LEG_JOINTS,
            effort_limit_sim=39.0,
            velocity_limit_sim=17.0,
            stiffness=50.0,
            damping=10.0,
            armature=0.015795,
        ),
        "wheel_motors": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINTS,
            effort_limit_sim=4.5,
            velocity_limit_sim=60.0,
            stiffness=0.0,
            damping=0.2,
            armature=0.0001,
        ),
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINTS,
            effort_limit_sim=1.0e9,
            velocity_limit_sim=60.0,
            stiffness=0.0,
            damping=0.02,
            armature=0.0001,
        ),
    },
)

"""Configuration for the UZ-05 wheel-legged robot imported from MJCF."""
