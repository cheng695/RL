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

import agent_tasks.manager.mdp.isaaclab as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg, ConstantNoiseCfg, UniformNoiseCfg, GaussianNoiseCfg
from dataclasses import MISSING, field
from collections import OrderedDict
import copy
import os
import torch

from agent_world.assets.wheelbipe_V13 import (
    Wheelbipe_V13_CFG,
    Wheelbipe_V13_DCMotor_CFG,
    Wheelbipe_V13_IdealPD_CFG,
    Wheelbipe_V13_DelayPD_CFG,
    Wheelbipe_V13_NS_CFG,
    Wheelbipe_V13_NS_IdealPD_CFG,
    Wheelbipe_V13_NS_DelayPD_CFG,
)

from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import Wheelbipe25v3FlatEnvCfg
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import EventCfg
from agent_tasks.manager.mdp.terrain import TerrainCommandOverrideCfg
from torch.functional import F

V13_TASK_FLAG_DIM = 1
V13_BASE_POLICY_OBS_DIM = 28
V13_BASE_PRIVILEGED_OBS_DIM = 32
V13_BODY_HEIGHT_SCANNER_GRID_SIZE = (0.02, 0.02)
V13_BODY_HEIGHT_SCANNER_RESOLUTION = 0.01
V13_WHEEL_HEIGHT_SCANNER_GRID_SIZE = (0.015, 0.015)
V13_WHEEL_HEIGHT_SCANNER_RESOLUTION = 0.01



def _enable_v13_task_flag_obs(cfg) -> None:
    """Append a reserved task flag slot next to command observations."""
    obs_space_is_dict = isinstance(getattr(cfg, "observation_space", None), dict)
    cfg.task_flag_obs_enabled = True
    cfg.task_flag_obs_dim = V13_TASK_FLAG_DIM
    cfg.num_single_obs = V13_BASE_POLICY_OBS_DIM + V13_TASK_FLAG_DIM
    cfg.num_single_privileged_obs = getattr(cfg, "num_single_privileged_obs", V13_BASE_PRIVILEGED_OBS_DIM) + V13_TASK_FLAG_DIM

    cfg.state_space = cfg.num_privileged_obs_hist * cfg.num_single_privileged_obs

    obs_clip_cfg = dict(getattr(cfg, "obs_input_clip_cfg", {}))
    obs_clip_cfg["task_flag"] = [-10.0, 10.0]
    cfg.obs_input_clip_cfg = obs_clip_cfg

    if obs_space_is_dict:
        cfg.observation_space = dict(cfg.observation_space)
        cfg.observation_space["policy"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["policy_hist"] = cfg.num_single_obs * cfg.num_obs_hist
        cfg.observation_space["critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["prev_critic"] = cfg.num_single_privileged_obs
        cfg.observation_space["critic_hist"] = cfg.num_single_privileged_obs * cfg.num_privileged_obs_hist
    else:
        if bool(getattr(cfg, "use_frame_stack", False)):
            cfg.observation_space = cfg.num_obs_hist * cfg.num_single_obs
        else:
            cfg.observation_space = cfg.num_single_obs


def _enable_v13_body_height_scanner(cfg) -> None:
    """Apply V13-specific body height scanner settings."""
    cfg.height_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V13_BODY_HEIGHT_SCANNER_RESOLUTION,
        size=V13_BODY_HEIGHT_SCANNER_GRID_SIZE,
    )
    cfg.height_scanner.ray_alignment = "yaw"
    cfg.height_scanner.debug_vis = False


def _enable_v13_wheel_height_scanners(cfg) -> None:
    """Attach dedicated raycasters to the left/right wheel centers."""
    base_scanner = copy.deepcopy(cfg.height_scanner)
    base_scanner.pattern_cfg = patterns.GridPatternCfg(
        resolution=V13_WHEEL_HEIGHT_SCANNER_RESOLUTION,
        size=V13_WHEEL_HEIGHT_SCANNER_GRID_SIZE,
    )
    base_scanner.ray_alignment = "yaw"
    base_scanner.debug_vis = False

    right_scanner = copy.deepcopy(base_scanner)
    right_scanner.prim_path = "/World/envs/env_.*/Robot/right_wheel_link"

    left_scanner = copy.deepcopy(base_scanner)
    left_scanner.prim_path = "/World/envs/env_.*/Robot/left_wheel_link"

    cfg.right_wheel_height_scanner = right_scanner
    cfg.left_wheel_height_scanner = left_scanner


def _disable_v13_wheel_height_scanners(cfg) -> None:
    """Disable dedicated wheel raycasters when scan-based helpers are unused."""
    cfg.right_wheel_height_scanner = None
    cfg.left_wheel_height_scanner = None


@configclass
class EventCfgV13(EventCfg):
    """Configuration for Wheelbipe25_v3 specific events."""
    # on start up
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.8, 1.3),
            "operation": "scale",
        },
    )
    base_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_front1_link",".*_front2_link",".*_front3_link",".*_front4_link",
                                                             ".*_rear1_link",".*_rear2_link",".*_spring1_link",".*_spring2_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    # # legs_inertia = EventTerm(
    # #     func=mdp.randomize_rigid_body_inertia,
    # #     mode="startup",
    # #     params={
    # #         "asset_cfg": SceneEntityCfg("robot", body_names=[".*_front1_link",".*_front2_link",".*_front3_link",".*_front4_link",
    # #                                                          ".*_rear1_link",".*_rear2_link",".*_spring1_link",".*_spring2_link"]),
    # #         "inertia_distribution_params": (0.9, 1.1),
    # #         "operation": "scale",
    # #     },
    # # )
    wheels_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.04, 0.04), "z": (-0.04, 0.04)},
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_wheel_link"),
            "static_friction_range": (0.5, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.05, 0.5),
            "num_buckets": 1024,
            "make_consistent": True,
        },
    )
    # add more events if needed
    legs_act_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint", ".*_front1_joint"]),
            "static_friction_distribution_params": (0.5, 1.0),
            "viscous_friction_distribution_params": (0.1, 0.25),
            # "viscous_friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.001, 0.003),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    wheel_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"),
            "static_friction_distribution_params": (0.05, 0.25),
            "viscous_friction_distribution_params": (0.0, 0.01),
            # "viscous_friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.00, 0.003),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    legs_inact_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                ".*_rear2_joint",
                ".*_front2_joint",
                ".*_front3_joint",
                ".*_front4_joint",
                ".*_spring1_joint",
            ]),
            "static_friction_distribution_params": (0.05, 0.1),
            "viscous_friction_distribution_params": (0.01, 0.025),
            # "viscous_friction_distribution_params": (0.0, 0.0),
            # "armature_distribution_params": (0., 0.01),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    guide_joint_frictions = EventTerm(
        func=mdp.randomize_joint_parameters_v1,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_guide_joint"),
            "static_friction_distribution_params": (0.001, 0.003),
            "viscous_friction_distribution_params": (0.001, 0.003),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # spring_frictions = EventTerm(
    #     func=mdp.randomize_joint_parameters_v1,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*_spring2_joint"),
    #         "static_friction_distribution_params": (0.1, 3),
    #         "viscous_friction_distribution_params": (100, 200),
    #         # "armature_distribution_params": (0., 0.01),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.75, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {
                # "x": (-0.5, 0.5), 
                # "y": (-0.5, 0.5),
                # "z": (0.0, 0.1),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14)},
            "velocity_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.25, 0.5),
                # "roll": (-0.25, 0.25),
                # "pitch": (-0.25, 0.25),
                # "yaw": (-0.25, 0.25),
            },
        },
    )

    # robot_joint_stiffness_and_damping = None
    # leg_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_rear1_joint",".*_front1_joint"]),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )
    # wheel_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     min_step_count_between_reset=720,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wheel_joint"]),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #         "distribution": "uniform",
    #     },
    # )


    # # 保留质量随机化（域随机化的核心诉求：play 时不要把 add_* 关掉）
    # add_base_mass = EventCfgV13.add_base_mass
    # add_leg_mass = EventCfgV13.add_leg_mass
    # add_wheel_mass = EventCfgV13.add_wheel_mass

    # # 其他事件在 play 时禁用，避免引入 reset/interval 的额外不确定性
    # base_com = None
    # physics_material = None
    # legs_act_joint_frictions = None
    # wheel_joint_frictions = None
    # legs_inact_joint_frictions = None
    # guide_joint_frictions = None

    # reset_base = None
    # robot_joint_stiffness_and_damping = None
    # push_robot = None
    # base_external_force_torque_xyz = None







@configclass
class WheelbipeV13FlatEnvCfg(Wheelbipe25v3FlatEnvCfg):
    """Configuration for the Wheelbipe_V13 direct RL environment with flat terrain."""
    # events
    events = EventCfgV13()
    play_keep_done_reset = True
    reset_heading_axis_aligned_only = True
    # curriculum: CurriculumCfg = CurriculumCfg()
    # events = EventCfg()
    # robot
    robot_cfg: ArticulationCfg = Wheelbipe_V13_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # joint offsets

    # termination_roll_deg = 20.0
    # termination_pitch_deg = 20.0

    alpha_offset = [2.5/180.*torch.pi,
                    torch.pi,
                    33.44/180.*torch.pi,
                    (180.+2.5-2*33.44)/180.*torch.pi,
                    (20.+33.44)/180.*torch.pi,
                    33.44/180.*torch.pi]
    mute_wheel_pos_obs = True

    # random obs and act time lag
    obs_delay_cfg = {
        'root_ang_vel_b': [1, 5],      # 5-100ms
        'projected_gravity_b': [1, 5],
        'joint_pos': [1, 5],
        'joint_vel': [1, 5],
    }
    obs_history_len = 10
    obs_default_time_lag = 1  # 中位数
    use_obs_delay = True
    
    act_delay_cfg = {
        'leg_actions': [0, 5],
        'wheel_actions': [0, 5],
    }
    use_act_delay = True

    ''' noise '''
    self_obs_noise_cfg = {
        'root_ang_vel_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
        'projected_gravity_b': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        'joint_pos': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.02, n_max=0.02)),
        'leg_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1., n_max=1.)),
        'wheel_joint_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1., n_max=1.)),
        'joint_torque': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.25, n_max=0.25)),
        'lin_vel': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.1, n_max=0.1)),
        'height': NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01)),
    }

    ''' reset cfgs '''
    use_leg_length_as_height = False
    height_range = [0.20,0.45]
    terrain_command_overrides: dict[str, TerrainCommandOverrideCfg] = field(default_factory=dict)
    terrain_command_switch_hold_steps: int = 0
    use_absolute_height = False
    use_wheel_vel_control = True

    use_predefined_leg_random_start = True
    predefined_settings = dict(
        prob = 0.2,
        leg_height = [-0.05,0.15],
        leg_length = [0.16,0.35],
        start_reset_time=2.0,
        start_root_height=0.2,
    )
    reset_heading_target_terminate_enabled = False
    reset_heading_target_terminate_threshold_deg = 20.0
    airborne_state_machine_cfg = {
        "enabled": True,
        "enter": {
            "wheel_radius": 0.05,
            "body_height_threshold": 0.3,
            "wheel_clearance_threshold": 0.05,
        },
        "target_height": {
            "bias": 0.3,
            "max": 0.45,
        },
        "exit": {
            "wheel_contact_force_threshold": 20.0,
            "wheel_contact_duration_s": 0.75,
            "base_contact_force_threshold": 10.0,
            "base_contact_duration_s": 0.25,
            "max_duration_s": 2.0,
        },
        "reward_scales": {
            "undesired_contact": 100.0,
            "flat_orientation_y": 3.0,
            "flat_orientation_x": 3.0,
            "wheel_motor_z_axis_align_exp": 3.0,
            "wheel_motor_z_axis_align_exp_tight": 3.0,
        },
        "reward_additions": {},
    }
    wheel_forward_scan_cfg = {
        "enabled": True,
        "scan": {
            "forward_offset": 0.8,
        },
        "detect": {
            "step_height_min": 0.10,
            "step_height_max": 0.22,
            "wall_height": 0.22,
        },
        "height_cmd": {
            "bias": 0.15,
            "hold_s": 2.0,
            "max": 0.4,
        },
    }
    undesired_contact_force_threshold = 5.0

    ''' history obs to self-obs '''
    use_frame_stack = False
    num_obs_hist = 1
    num_privileged_obs_hist = 1
    # 概率帧遮蔽 [p_keep_1, p_keep_2, p_keep_3, p_keep_4, (p_keep_5=剩余)]
    #   40% 保留最新1帧（遮4帧）
    #   10% 保留最新2帧（遮3帧）
    #   10% 保留最新3帧（遮2帧）
    #   10% 保留最新4帧（遮1帧）
    #   30% 保留全部5帧（不遮）
    # frame_mask_probs: list | None = [0.3, 0.1, 0.1, 0.1]

    # 速度/角速度 tracking 奖励门控：只有接近水平时奖励才高（和 flat_orientation 关联）
    vel_upright_gate_enabled: bool = False
    vel_upright_gate_sigma: float = 0.1
    # 速度/角速度高度门控：直接使用 height tracker 形式 gate=exp(-height_err/sigma)
    # None 表示复用 height_sigma；参考 sigma：0.005(严格) / 0.01(平衡) / 0.02(宽松)
    vel_height_gate_enabled: bool = False
    vel_height_gate_tracker_sigma: float = 0.02
    # 高度 tracking 奖励门控：只有接近水平时奖励才高（和 flat_orientation 关联）
    height_upright_gate_enabled: bool = False
    height_upright_gate_sigma: float = 0.1
    # 轮子在机身坐标系下应尽量贴近 x=0；普通/tight 两档奖励都基于这个误差。
    wheel_motor_z_axis_align_ref_y_offset: float = 0.217
    wheel_motor_z_axis_align_tolerance: float = 0.0
    wheel_motor_z_axis_align_sigma: float = 0.01
    wheel_motor_z_axis_align_tight_sigma: float = 0.001

    # value 爆炸诊断：默认关闭，可在训练配置按需打开
    debug_value_diagnosis: bool = False
    debug_value_diagnosis_interval: int = 50
    debug_value_diagnosis_topk: int = 3
    # True: 只有超过阈值时才打印 ValueDebug
    debug_value_diagnosis_threshold_only: bool = False
    # 各项阈值（<=0 表示不启用该项阈值）
    debug_value_diagnosis_thresholds: dict = {
        "reward_total_abs": 1.0,
        "reward_term_abs": 0.5,
        "state_joint_vel_wheel_abs": 120.0,
        "state_root_lin_vel_b_abs": 20.0,
        "state_root_ang_vel_b_abs": 20.0,
        "state_applied_torque_abs": 350.0,
        "obs_policy_abs": 100.0,
        "obs_critic_abs": 100.0,
    }
    # 观测输入限幅（obs/priv 共用），按分量单独设置
    # 支持标量(abs 限幅)或[min,max]
    obs_input_clip_cfg: dict = {
        "command": 100.0,
        "height_cmd": [0.0, 1.0],
        "root_ang_vel_b": 100.0,
        "projected_gravity_b": 100.0,
        "joint_pos": 100.0,
        "joint_vel_leg": 200.0,
        "joint_vel_wheel": 200.0,
        "actions": 100.0,
        "root_lin_vel_b": 100.0,
        "obs_height": [-10.0, 10.0],
    }
    # 原始输入预警：超过阈值才开始打印，正常时不打印
    debug_obs_alert_threshold: float = 120.0
    debug_obs_alert_topk: int = 3
    debug_obs_alert_print_interval: int = 1

    ''' new reward '''
    rewards = OrderedDict(
        # ### alive
        # termination = -10.,
        # # epi_len = 0.,

        # ### regularization
        # # joint_acc = -2.5e-8,
        # leg_joint_acc = -5e-7,
        # # wheel_acc = -2.5e-9,
        # joint_vel = -5.0e-6,
        # # leg_joint_vel = -5.0e-7,
        # # wheel_vel = -5.0e-7,
        # joint_torque = -1e-4,
        # wheel_power = -1e-5,
        # lin_vel_z = -0.5,
        # # lin_vel_z_exp = 0.5,
        # ang_vel_xy = -0.05,
        # action_rate = -0.005,
        # action_smoothness = -0.05,

        # ### tasks
        # # flat_orientation = -1.,
        # flat_orientation_y = -3,
        # flat_orientation_y_v = -2.,
        # flat_orientation_x = -1.,
        # flat_orientation_x_v = -1.,

        # track_lin_vel_xy = 0.7,
        # # track_lin_vel_xy_soft = 0.4,
        # track_lin_vel_xy_tight = 1.0,
        # # track_lin_vel_xy_huge_gap = -1.0,
        # track_lin_vel_xy_square = -5.0,
        # # penalty_high_speed = -1.0,

        # track_ang_vel_z = 0.5,
        # # track_ang_vel_z_soft = 0.3,
        # # track_ang_vel_z_huge_gap = -1.0,
        # track_ang_vel_z_square = -1.0,

        # # track_height_square = -0.0,
        # track_height_exp = 0.4,
        # track_height_exp_tight = 1.2,
        ### alive
        # termination = 0.,
        # epi_len = 0.,

        ### regularization
        # joint_deviation = -1e-3,
        # joint_acc = -2.5e-8,
        leg_joint_acc = -2.5e-8,
        # wheel_acc = -2.5e-8,
        joint_vel = -5.0e-7,
        # leg_joint_vel = -5.0e-7,
        # wheel_vel = -5.0e-7,
        joint_torque = -1e-5,
        wheel_power = -1e-3,
        lin_vel_z = -1.0,
        # lin_vel_z_exp = 0.5,
        ang_vel_xy = -0.01,
        # action_rate = -0.02,
        action_smoothness = -0.01,

        ### tasks
        # flat_orientation = -1.,
        flat_orientation_y = -5.0,
        flat_orientation_y_v = -5.0,
        flat_orientation_x = -1.0,
        flat_orientation_x_v = -1.0,

        track_lin_vel_xy = 1.0,
        # track_lin_vel_xy_soft = 0.4,
        track_lin_vel_xy_tight = 1.0,
        # track_lin_vel_xy_huge_gap = -1.0,
        track_lin_vel_xy_square = -5.0,
        # penalty_high_speed = -1.0,
        # penalty_over_lin_vel_x_square = -1.0,
        # penalty_over_lin_vel_x_step = -1.0,

        track_ang_vel_z = 0.5,
        # track_ang_vel_z_soft = 0.3,
        # track_ang_vel_z_huge_gap = -1.0,
        track_ang_vel_z_square = -1.0,

        # track_height_square = -1.0,
        track_height_exp = 0.4,
        track_height_exp_tight = 1.0,
        # stand_nice = -0.0,
        # no_fork = -1.0,
        no_fork_exp = -5.0,
        no_fork_z_exp = -5.0,
        # no_fork_square = -1.0,
        undesired_contact = -2.0,
        wheel_motor_z_axis_align_exp = 1.0,
        wheel_motor_z_axis_align_exp_tight = 1.0,
        # desired_contact = -0.0,

        # actions_joint_limits = -0.0,
        # current_joint_limits = -1.0,
    )

    def __post_init__(self):
        super().__post_init__()
        _enable_v13_body_height_scanner(self)
        if bool(getattr(self, "use_leg_length_as_height", False)):
            # Flat + leg-length height tracking uses wheel-center geometry directly,
            # so wheel raycasters and their scan-driven helpers are unnecessary.
            self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
            self.airborne_state_machine_cfg["enabled"] = False
            self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
            self.wheel_forward_scan_cfg["enabled"] = False

        if bool(self.airborne_state_machine_cfg.get("enabled", False)) or bool(
            getattr(self, "wheel_forward_scan_cfg", {}).get("enabled", False)
        ):
            _enable_v13_wheel_height_scanners(self)
        else:
            _disable_v13_wheel_height_scanners(self)
        # reconfig obs space (Sim2Sim: policy = N 帧 [最旧...最新], 维度 N*D)
        if hasattr(self, 'use_frame_stack') and self.use_frame_stack:
            # self.num_obs_hist = self.obs_history_len  # 与 env 帧堆叠长度对齐
            self.observation_space = self.num_obs_hist * getattr(self, 'num_single_obs', 28)
        # critic 帧堆叠：state_space 更新为堆叠后的总维度
        # critic MLP in_features = num_privileged_obs_hist × num_single_privileged_obs
        self.state_space = self.num_privileged_obs_hist * getattr(self, 'num_single_privileged_obs', 32)

        # motion commands
        # self.commands = mdp.UniformVelocityCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(7.0, 15.0),
        #     rel_standing_envs=0.1,
        #     rel_heading_envs=0.5,
        #     heading_command=True,
        #     heading_control_stiffness=1.0,
        #     debug_vis=False,
        #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
        #         lin_vel_x=(-2.5, 2.5),
        #         lin_vel_y=(0., 0.),
        #         ang_vel_z=(-torch.pi, torch.pi),
        #         heading=(-torch.pi, torch.pi)
        #     ),
        # )
        self.commands = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(7.0, 15.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.5,
            heading_command=True,
            heading_control_stiffness=1.0,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-2.5, 2.5),
                lin_vel_y=(0., 0.),
                ang_vel_z=(-torch.pi, torch.pi),
                heading=(-torch.pi, torch.pi)
            ),
        )

        self.decimation = 4
        self.sim.dt = 1/200.
        self.max_wheel_torque = 20.





























        
##############################################
# NS (No Spring) Versions  
##############################################



        # self.rewards['leg_joint_acc'] = -2.5e-8
        # self.rewards['wheel_acc'] = -2.5e-9



# =============================================================================
# USD 地形环境：使用 RMUC2026 USD 文件作为整体地形
# terrain_type="usd" 模式下所有环境实例共享同一张地形，机器人按 env_spacing 网格排列
# 无子地形划分，所有环境实例共享同一整张 USD 地形
# =============================================================================



