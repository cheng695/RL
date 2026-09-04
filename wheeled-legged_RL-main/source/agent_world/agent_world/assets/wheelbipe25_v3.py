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
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg

from agent_world import AssetPath

"""Configuration for the Wheelbipe25_v3 Wheelbipe robot."""

Wheelbipe25_v3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/wheelbipe25_v3_loop/wheelbipe25_v3_loop.usd",
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
                ".*_rear1_joint": 30.,
                ".*_front1_joint": 30.,
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
        "legs_inact": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
            ],
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
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
    },
)

Wheelbipe25_v3_IdealPD_CFG = Wheelbipe25_v3_CFG.copy()
Wheelbipe25_v3_IdealPD_CFG.actuators={
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

Wheelbipe25_v3_DelayPD_CFG = Wheelbipe25_v3_CFG.copy()
Wheelbipe25_v3_DelayPD_CFG.actuators={
        "legs_act": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_rear1_joint",
                ".*_front1_joint",
            ],
            min_delay=0,
            max_delay=4,
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
        "legs_inact": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
            ],
            min_delay=0,
            max_delay=4,
            stiffness=0.0,
            damping=0.0,
            armature=0.0001,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_wheel_joint",
            ],
            min_delay=0,
            max_delay=4,
            stiffness=0.0,
            damping=0.1,
            effort_limit=5.,
            # velocity_limit=50.,
            armature=0.0001,
        ),
        "spring": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_spring2_joint",
            ],
            min_delay=0,
            max_delay=4,
            stiffness=0.,
            damping=0.01,
            # effort_limit=1000,
            # velocity_limit=100.,
            armature=0.0001,
        ),
    }

Wheelbipe25_v3_guide_CFG = Wheelbipe25_v3_CFG.copy()
Wheelbipe25_v3_guide_CFG.spawn.usd_path=f"{AssetPath}/usd_files/wheelbipe25_v3_guide/wheelbipe25_v3_guide.usd"
Wheelbipe25_v3_guide_IdealPD_CFG = Wheelbipe25_v3_IdealPD_CFG.copy()
Wheelbipe25_v3_guide_IdealPD_CFG.spawn.usd_path=f"{AssetPath}/usd_files/wheelbipe25_v3_guide/wheelbipe25_v3_guide.usd"
Wheelbipe25_v3_guide_DelayPD_CFG = Wheelbipe25_v3_DelayPD_CFG.copy()
Wheelbipe25_v3_guide_DelayPD_CFG.spawn.usd_path=f"{AssetPath}/usd_files/wheelbipe25_v3_guide/wheelbipe25_v3_guide.usd"