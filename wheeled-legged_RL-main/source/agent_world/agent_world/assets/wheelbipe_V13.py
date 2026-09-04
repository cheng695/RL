# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg, DCMotorCfg

from agent_world import AssetPath

"""Configuration for the Wheelbipe V13 robot.

Includes both standard versions (with spring) and NS versions (No Spring):
- Wheelbipe_V13_CFG: Standard version with ImplicitActuator
- Wheelbipe_V13_DCMotor_CFG: Standard version with DCMotor wheel actuator
- Wheelbipe_V13_IdealPD_CFG: Standard version with IdealPDActuator
- Wheelbipe_V13_DelayPD_CFG: Standard version with DelayedPDActuator
- Wheelbipe_V13_NS_CFG: No Spring version with ImplicitActuator  
- Wheelbipe_V13_NS_IdealPD_CFG: No Spring version with IdealPDActuator
- Wheelbipe_V13_NS_DelayPD_CFG: No Spring version with DelayedPDActuator
"""


Wheelbipe_V13_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/wheelbipe_V13/wheelbipe_V13.usd",
        activate_contact_sensors=True,
        copy_from_source=True,  # Required for proper articulation loading in Isaac Lab 2.3.0
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
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            # sleep_threshold=0.005,
            # stabilization_threshold=0.001,
        ),
    ),
    # soft_joint_pos_limit_factor=0.95,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*_rear1_joint": 0.0,
            ".*_rear2_joint": 0.0,
            ".*_front1_joint": 0.0,
            ".*_front2_joint": 0.0,
            ".*_front3_joint": 0.0,
            ".*_front4_joint": 0.0,
            ".*_spring1_joint": 0.0,
            ".*_spring2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_act": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            # 30-40
            stiffness={
                ".*_rear1_joint": 25.,
                ".*_front1_joint": 25.,
            },
            # 0.25-2
            damping={
                ".*_rear1_joint": 2.5,
                ".*_front1_joint": 2.5,
            },
            effort_limit={
                ".*_rear1_joint": 40.,
                ".*_front1_joint": 40.,
            },
            velocity_limit={
                ".*_rear1_joint": 20.,
                ".*_front1_joint": 20.,
            },
            armature=0.001,
        ),
        "legs_inact": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
                ".*_guide_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
            effort_limit=50,
            velocity_limit=20,
        ),
        # "wheel": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_wheel_joint",
        #     ],
        #     stiffness=0.0,
        #     damping=0.01,
        #     effort_limit=5.,
        #     # velocity_limit=50.,
        #     armature=0.0001,
        # ),
        # "spring": ImplicitActuatorCfg(
        #     joint_names_expr=[
        #         ".*_spring2_joint",
        #     ],
        #     stiffness=0.,
        #     damping=0.01,
        #     # effort_limit=1000,
        #     # velocity_limit=100.,
        #     armature=0.0001,
        # ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_wheel_joint",
            ],
            stiffness=0.0,
            damping=0.2,
            effort_limit=6.,
            velocity_limit=60.,
            armature=0.001,
        ),
        # "wheel": DCMotorCfg(
        # joint_names_expr=[
        #     ".*_wheel_joint",
        # ],
        # # --- 控制层 (纯力矩直通，阻尼由物理层模型提供) ---
        # stiffness=0.0,
        # damping=0.2,
        # # --- 电机特性曲线参数 ---
        # saturation_effort=7.0,    # 堵转（峰值）力矩 [Nm]
        # effort_limit=5,          # 持续额定力矩 [Nm]
        # velocity_limit=50.0,       # 空载最高转速 [rad/s]
        # # # --- 物理摩擦参数（来自实测拟合，独立于控制阻尼作用于 PhysX 关节） ---
        # # friction=0.25,             # 静摩擦系数（死区阈值 ~0.2 Nm）
        # # dynamic_friction=0.0775,   # 动摩擦系数 [Nm，Coulomb常数项]
        # # viscous_friction=0.0988,   # 粘滞摩擦系数 [Nms/rad，速度线性项]
        # # --- 转子惯量 ---
        # armature=0.001,            # 待标定，当前为估算值 [kg·m²]
        # ),
        "spring": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_spring2_joint",
            ],
            stiffness=0.,
            damping=50.0,
            effort_limit=500,
            velocity_limit=50.,
            armature=0.0001,
        ),
    },
)

Wheelbipe_V13_IdealPD_CFG = Wheelbipe_V13_CFG.copy()
Wheelbipe_V13_IdealPD_CFG.actuators={
        "legs_act": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            # 30-40
            stiffness={
                ".*_rear1_joint": 40.,
                ".*_front1_joint": 40.,
            },
            # 0.25-2
            damping={
                ".*_rear1_joint": 1.0,
                ".*_front1_joint": 1.0,
            },
            # effort_limit={
            #     ".*_rear1_joint": 100.,
            #     ".*_front1_joint": 100.,
            # },
            # velocity_limit={
            #     ".*_rear1_joint": 20.,
            #     ".*_front1_joint": 20.,
            # },
            armature=0.001,
        ),
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
                ".*_guide_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
        ),
        "wheel": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_wheel_joint",
            ],
            stiffness=0.0,
            damping=0.1,
            effort_limit=5.,
            # velocity_limit=50.,
            armature=0.0001,
        ),
        "spring": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_spring2_joint",
            ],
            stiffness=0.,
            damping=0.01,
            # effort_limit=1000,
            # velocity_limit=100.,
            armature=0.0001,
        ),
    }

Wheelbipe_V13_DCMotor_CFG = Wheelbipe_V13_CFG.copy()
Wheelbipe_V13_DCMotor_CFG.actuators = {
        "legs_act": DCMotorCfg(
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            stiffness={
                ".*_rear1_joint": 25.,
                ".*_front1_joint": 25.,
            },
            damping={
                ".*_rear1_joint": 2.5,
                ".*_front1_joint": 2.5,
            },
            effort_limit={
                ".*_rear1_joint": 25.,
                ".*_front1_joint": 25.,
            },
            saturation_effort=40.0,
            velocity_limit={
                ".*_rear1_joint": 20.,
                ".*_front1_joint": 20.,
            },
            armature=0.001,
        ),
        "legs_inact": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
                ".*_guide_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
            effort_limit=100,
            velocity_limit=20,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=[
                ".*_wheel_joint",
            ],
            stiffness=0.0,
            damping=0.2,
            saturation_effort=7.0,
            effort_limit=5.0,
            velocity_limit=60.0,
            armature=0.001,
        ),
        "spring": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_spring2_joint",
            ],
            stiffness=0.,
            damping=200.0,
            effort_limit=500,
            velocity_limit=50.,
            armature=0.0001,
        ),
    }

Wheelbipe_V13_DelayPD_CFG = Wheelbipe_V13_CFG.copy()
Wheelbipe_V13_DelayPD_CFG.actuators={
        "legs_act": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            min_delay=0,
            max_delay=3,
            # 30-40
            stiffness={
                ".*_rear1_joint": 25.,
                ".*_front1_joint": 25.,
            },
            # 0.25-2
            damping={
                ".*_rear1_joint": 2.0,
                ".*_front1_joint": 2.0,
            },
            # effort_limit={
            #     ".*_rear1_joint": 100.,
            #     ".*_front1_joint": 100.,
            # },
            # velocity_limit={
            #     ".*_rear1_joint": 20.,
            #     ".*_front1_joint": 20.,
            # },
            armature=0.001,
        ),
        "legs_inact": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
                ".*_guide_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wheel_joint",
            ],
            min_delay=0,
            max_delay=3,
            stiffness=0.0,
            damping=0.2,
            effort_limit=5.,
            # velocity_limit=50.,
            armature=0.0001,
        ),
        "spring": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_spring2_joint",
            ],
            stiffness=0.,
            damping=0.01,
            # effort_limit=1000,
            # velocity_limit=100.,
            armature=0.0001,
        ),
    }

##############################################
# NS (No Spring) Versions
##############################################

Wheelbipe_V13_NS_CFG = Wheelbipe_V13_CFG.copy()
Wheelbipe_V13_NS_CFG.spawn.usd_path = f"{AssetPath}/usd_files/wheelbipe_V13_NS/wheelbipe_V13_NS.usd"
Wheelbipe_V13_NS_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.35),
    joint_pos={
        ".*_rear1_joint": 0.0,
        ".*_rear2_joint": 0.0,
        ".*_front1_joint": 0.0,
        ".*_front2_joint": 0.0,
        ".*_front3_joint": 0.0,
        ".*_front4_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)
Wheelbipe_V13_NS_CFG.actuators = {
    "legs_act": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_rear1_joint",
            ".*_front1_joint",
        ],
        stiffness={
            ".*_rear1_joint": 40.,
            ".*_front1_joint": 40.,
        },
        damping={
            ".*_rear1_joint": 1.0,
            ".*_front1_joint": 1.0,
        },
        armature=0.001,
    ),
    "legs_inact": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_rear2_joint",
            ".*_front2_joint",
            ".*_front3_joint",
            ".*_front4_joint",
            ".*_guide_joint",
        ],
        stiffness=0.0,
        damping=0.0,
        armature=0.0001,
    ),
    "wheel": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_wheel_joint",
        ],
        stiffness=0.0,
        damping=0.1,
        effort_limit=5.,
        armature=0.0001,
    ),
    # NS版本没有spring actuator
}

Wheelbipe_V13_NS_IdealPD_CFG = Wheelbipe_V13_NS_CFG.copy()
Wheelbipe_V13_NS_IdealPD_CFG.actuators = {
    "legs_act": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_rear1_joint",
            ".*_front1_joint",
        ],
        stiffness={
            ".*_rear1_joint": 40.,
            ".*_front1_joint": 40.,
        },
        damping={
            ".*_rear1_joint": 1.0,
            ".*_front1_joint": 1.0,
        },
        armature=0.001,
    ),
    "legs_inact": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_rear2_joint",
            ".*_front2_joint",
            ".*_front3_joint",
            ".*_front4_joint",
            ".*_guide_joint",
        ],
        stiffness=0.0,
        damping=0.0,
        armature=0.0001,
    ),
    "wheel": IdealPDActuatorCfg(
        joint_names_expr=[
            ".*_wheel_joint",
        ],
        stiffness=0.0,
        damping=0.1,
        effort_limit=5.,
        armature=0.0001,
    ),
}

Wheelbipe_V13_NS_DelayPD_CFG = Wheelbipe_V13_NS_CFG.copy()
Wheelbipe_V13_NS_DelayPD_CFG.actuators = {
    "legs_act": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_rear1_joint",
            ".*_front1_joint",
        ],
        min_delay=1,
        max_delay=5,
        stiffness={
            ".*_rear1_joint": 40.,
            ".*_front1_joint": 40.,
        },
        damping={
            ".*_rear1_joint": 1.0,
            ".*_front1_joint": 1.0,
        },
        armature=0.001,
    ),
    "legs_inact": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_rear2_joint",
            ".*_front2_joint",
            ".*_front3_joint",
            ".*_front4_joint",
            ".*_guide_joint",
        ],
        stiffness=0.0,
        damping=0.0,
        armature=0.0001,
    ),
    "wheel": DelayedPDActuatorCfg(
        joint_names_expr=[
            ".*_wheel_joint",
        ],
        min_delay=1,
        max_delay=5,
        stiffness=0.0,
        damping=0.1,
        effort_limit=5.,
        armature=0.0001,
    ),
}
