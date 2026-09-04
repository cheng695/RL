from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .mdp import observations as wheel_leg_observations
from .mdp import rewards as wheel_leg_rewards
from .mdp import terminations as wheel_leg_terminations
from .mdp import commands as wheel_leg_commands
from .mdp import actions as wheel_leg_actions

USD_PATH = str(Path.home() / "桌面/usd/COD-2026RoboMaster-Balance-Simulation_File/USD/COD-2026RoboMaster-Balance.usd")

# 头顶可视化箭头：绿色表示期望速度，蓝色表示当前实际速度。
COMMAND_GOAL_VEL_MARKER_CFG = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
COMMAND_GOAL_VEL_MARKER_CFG.markers["arrow"].scale = (0.55, 0.16, 0.16)
COMMAND_CURRENT_VEL_MARKER_CFG = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_current")
COMMAND_CURRENT_VEL_MARKER_CFG.markers["arrow"].scale = (0.45, 0.14, 0.14)

# yaw rate 可视化圆弧箭头：绿色表示期望 yaw rate，蓝色表示实际 yaw rate。
COMMAND_GOAL_YAW_MARKER_CFG = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/yaw_goal")
COMMAND_GOAL_YAW_MARKER_CFG.markers["arrow"].scale = (0.18, 0.08, 0.08)
COMMAND_CURRENT_YAW_MARKER_CFG = BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/yaw_current")
COMMAND_CURRENT_YAW_MARKER_CFG.markers["arrow"].scale = (0.16, 0.07, 0.07)

# 三阶段 curriculum:
# 1 = standing pretrain, 2 = forward walking, 3 = velocity/yaw turning.
# 当前进入 Stage 2：80% 前进速度跟踪 + 20% 原地站立巩固。
CURRICULUM_STAGE = 2

# Stage 1 随机高度原地平衡范围：
# reset 时采样一个高度，episode 内保持不变。
STAGE1_BASE_HEIGHT_RANGE = (0.32, 0.42)

# base height command 采样范围，单位 m。
BASE_HEIGHT_COMMAND_RANGE = STAGE1_BASE_HEIGHT_RANGE

# reset 时 root/base_link 初始 z 高度，单位 m；实际 reset 还会叠加 RESET_ROOT_POS_Z_RANGE 扰动。
INITIAL_BASE_HEIGHT = 0.45

# base_link 低高度 termination 的阈值，单位 m；低于该高度且超过宽限步数会 reset。
MIN_BASE_HEIGHT = 0.23

# 腿部主动关节偏离 USD/default 站姿过大 termination 阈值，单位 rad。
MAX_LEG_JOINT_DEVIATION = 1.4

# 腿部主动关节偏离 termination 的启动宽限步数。
LEG_JOINT_DEVIATION_GRACE_STEPS = 60

# Fudan plane 风格：机身接近翻倒后持续约 1s 再 reset。
BAD_ORIENTATION_LIMIT_ANGLE = 1.47
BAD_ORIENTATION_GRACE_STEPS = 60

# reward 中 z 速度惩罚的裁剪上限。
LIN_VEL_Z_REWARD_CLIP = 25.0

# reward 中 roll/pitch 角速度惩罚的裁剪上限。
ANG_VEL_XY_REWARD_CLIP = 10.0

# reward 中关节速度惩罚的裁剪上限。
JOINT_VEL_REWARD_CLIP = 1.0e4

# reward 中关节加速度惩罚的裁剪上限。
JOINT_ACC_REWARD_CLIP = 1.0e5

# reward 中关节力矩惩罚的裁剪上限。
JOINT_TORQUE_REWARD_CLIP = 1.0e5

# reward 中 action rate 惩罚的裁剪上限。
ACTION_RATE_REWARD_CLIP = 1.0e3

# reward 中 action 二阶差分惩罚的裁剪上限。
ACTION_SMOOTH_REWARD_CLIP = 1.0e3

# vx-only 时鼓励左右轮 action 反号，辅助直线前后运动。
VX_WHEEL_ANTIPHASE_MIN_ABS_COMMAND = 0.1
VX_WHEEL_ANTIPHASE_YAW_DEADBAND = 0.02
VX_WHEEL_ANTIPHASE_REWARD_CLIP = 4.0

# 随机高度原地平衡阶段：允许小范围轮子修正，但限制长期漂移。
ROOT_XY_POSITION_DEADBAND = 0.10

# 整机 COM 与左右轮中点的 base x 方向对齐约束。
COM_WHEEL_ALIGNMENT_DEADBAND = 0.01
COM_WHEEL_ALIGNMENT_ERROR_SCALE = 0.02

# 基于 hip point -> wheel center 的弱腿部垂直支撑约束。
LEG_VERTICAL_ERROR_SCALE = 0.05
LEG_VERTICAL_DEADBAND = 0.02
WHEEL_RADIUS = 0.05
MIN_LEG_VERTICAL_TARGET = 0.07
MAX_LEG_VERTICAL_TARGET = 0.35

# Fudan soft dof position limit ratio.
SOFT_DOF_POS_LIMIT = 0.97

# Fudan plane 风格：关节 reset 回默认角，不额外随机扰动。
RESET_JOINT_POS_RANGE = (0.0, 0.0)

# Fudan plane 风格：关节 reset 速度清零。
RESET_JOINT_VEL_RANGE = (0.0, 0.0)

# reset 时 root x 位置随机扰动范围，单位 m。
RESET_ROOT_POS_X_RANGE = (-0.05, 0.05)

# reset 时 root y 位置随机扰动范围，单位 m。
RESET_ROOT_POS_Y_RANGE = (-0.05, 0.05)

# reset 时 root z 位置相对 INITIAL_BASE_HEIGHT 的随机扰动范围，单位 m。
RESET_ROOT_POS_Z_RANGE = (-0.02, 0.02)

# reset 时 root yaw 角随机扰动范围，单位 rad。
RESET_ROOT_YAW_RANGE = (-0.1, 0.1)

# velocity command 的重采样周期。
COMMAND_RESAMPLING_TIME = 5.0

# 当前随机高度平衡阶段：
# episode_length_s=20s，因此每 10s 重采样一次；一轮 episode 内经历两个随机高度。
BASE_HEIGHT_RESAMPLING_TIME = 10.0

# fallback vx 速度命令范围；三阶段 curriculum 会优先使用 stage2/stage3 的范围。
COMMAND_LIN_VEL_X_RANGE = (0.0, 0.0)

# vy 速度命令范围，两轮车通常固定为 0。
COMMAND_LIN_VEL_Y_RANGE = (0.0, 0.0)

# fallback yaw rate 命令范围；Stage 2 仍保持 yaw_rate=0，只训练直线 vx。
COMMAND_ANG_VEL_Z_RANGE = (0.0, 0.0)

# fallback 站立命令环境比例；Stage 2 第一版实际使用 stage2_standing_ratio=0.0，先避开停车样本。
COMMAND_STANDING_ENV_RATIO = 1.0

# 非 exclusive 模式下不额外制造 yaw-only 专项环境。
COMMAND_YAW_ENV_RATIO = 0.0

# 保留诊断字段；联合随机命令下正负 vx 由均匀采样自然接近 50/50。
COMMAND_FORWARD_VX_ENV_RATIO = 0.5

# vx/yaw tracking reward 的指数衰减系数：exp(-error^2 / sigma)。
LIN_VEL_TRACKING_SIGMA = 0.05
YAW_TRACKING_SIGMA = 0.05

# Stage 2 vx tracking:
# Gaussian 负责近目标精确跟踪，Huber 负责远离目标时仍有稳定梯度。
VX_TRACKING_SIGMA = 0.35
VX_TRACKING_HUBER_BETA = 0.3
VX_TRACKING_HUBER_CLIP = 2.0
WHEEL_VX_TRACKING_SIGMA = 0.35
WHEEL_VX_TRACKING_HUBER_BETA = 0.3
WHEEL_VX_TRACKING_HUBER_CLIP = 2.0
BASE_WHEEL_VX_CONSISTENCY_BETA = 0.2
BASE_WHEEL_VX_CONSISTENCY_CLIP = 2.0

# pitch 辅助平衡；权重低于 vx 主任务，避免 policy 只学会原地站直。
PITCH_TRACKING_SIGMA = 0.25
PITCH_RATE_REWARD_CLIP = 10.0

# Curriculum reward kernels. 改这些系数即可实现松/紧 kernel 切换：
# loose: exp(-2*x^2), tight: exp(-20*x^2) 或更高。
STAGE1_HEIGHT_KERNEL = 80.0
STAGE1_ORIENTATION_KERNEL = 20.0
STAGE1_ANG_VEL_REWARD_CLIP = 25.0

# base_link 接触地面 termination 的启动宽限步数。
BASE_CONTACT_GRACE_STEPS = 500

# base_link 低高度 termination 的启动宽限步数。
LOW_BASE_HEIGHT_GRACE_STEPS = 500

# Fudan collision reward 在当前 USD 中用 base_link 接触近似，单位 N。
BASE_CONTACT_REWARD_FORCE_THRESHOLD = 0.1

# base_link 接触 termination 的接触力阈值，单位 N。
BASE_CONTACT_TERMINATION_FORCE_THRESHOLD = 10.0

# Fudan nominal_state 在当前 USD 中用 hip point -> wheel center 的左右虚拟腿摆角差近似。
LEFT_HIP_OFFSET_B = (-0.0193914, -0.1717, -0.05)
RIGHT_HIP_OFFSET_B = (-0.0193914, 0.1707, -0.05)

# 左右轮在 base x 方向错开达到 12cm 时终止，作为极端劈叉的安全兜底，单位 m。
MAX_WHEEL_X_SEPARATION = 0.12

# 劈叉 termination 的启动宽限步数；训练初期先让 reward 塑形，hard reset 只兜底极端情况。
WHEEL_X_SEPARATION_GRACE_STEPS = 300

# 对齐 wheeled-legged_RL pretrained: leg_target = default + 0.5 * action。
LEG_POSITION_ACTION_SCALE = 0.5

# 腿部真实关节目标每个 policy step 最大变化量，单位 rad。
# 0 表示禁用额外限速，对齐 Fudan: pos_ref = action * scale。
LEG_POSITION_MAX_TARGET_DELTA = 0.0

# 对齐 wheeled-legged_RL pretrained: wheel_target_vel = 10.0 * action，单位 rad/s。
WHEEL_VELOCITY_ACTION_SCALE = 10.0

# 与移动阶段保持一致；该 pretrained 配置没有 standing/moving 分段轮速 scale。
WHEEL_STANDING_VELOCITY_ACTION_SCALE = 10.0

# 对齐 wheeled-legged_RL 的线性 action->target 映射，不额外做 cubic shaping。
WHEEL_ACTION_CUBIC_BETA = 0.0

# 主动控制的四个腿关节。
LEG_JOINTS = [
    "Left_front_joint",
    "Left_rear_joint",
    "Right_front_joint",
    "Right_rear_joint",
]

# 主动控制的两个轮子关节。
WHEEL_JOINTS = [
    "Left_Wheel_joint",
    "Right_Wheel_joint",
]

# 策略输出控制的全部 6 个关节：4 个腿位置 + 2 个轮速。
CONTROLLED_JOINTS = LEG_JOINTS + WHEEL_JOINTS
LEG_KINEMATICS_BODIES = [
    "base_link",
    "Left_Wheel_link",
    "Right_Wheel_link",
]

@configclass
class WheelLegSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.4,
                restitution=0.0,
            )
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot : ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",

        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, INITIAL_BASE_HEIGHT),
            joint_vel={".*": 0.0},
        ),

        actuators={
            "leg_motors": ImplicitActuatorCfg(
                joint_names_expr=LEG_JOINTS,
                effort_limit_sim=40.0,
                velocity_limit_sim=17.0,
                stiffness=60.0,
                damping=2.0,
                armature=0.015795,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=WHEEL_JOINTS,
                effort_limit_sim=5.0,
                velocity_limit_sim=60.0,
                stiffness=0.0,
                damping=0.2,
                armature=0.0,
            ),
        },
    )


    base_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=3,
        debug_vis=True,
    )

    wheel_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*Wheel_link",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
    )

@configclass
class ActionsCfg:
    leg_joint_positions = wheel_leg_actions.BoundedJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=LEG_POSITION_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        target_clip=LEG_POSITION_ACTION_SCALE,
        max_target_delta=LEG_POSITION_MAX_TARGET_DELTA,
    )

    wheel_joint_velocities = wheel_leg_actions.NonlinearJointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=WHEEL_VELOCITY_ACTION_SCALE,
        cubic_beta=WHEEL_ACTION_CUBIC_BETA,
        command_name="base_velocity",
        standing_scale=WHEEL_STANDING_VELOCITY_ACTION_SCALE,
        linear_command_scale_ref=1.0,
        yaw_command_scale_ref=1.0,
        use_default_offset=True,
        preserve_order=True,
    )

@configclass
class CommandsCfg:
    base_velocity = wheel_leg_commands.WheelLegVelocityCommandCfg(
        asset_name="robot",
        curriculum_stage=CURRICULUM_STAGE,
        stage2_standing_ratio=0.0,
        stage2_vx_range=(-1.0, 1.0),
        stage2_vx_abs_range=(0.3, 1.0),
        stage3_standing_ratio=0.1,
        stage3_walking_ratio=0.6,
        stage3_vx_range=(-1.0, 1.0),
        stage3_vy_range=(-0.3, 0.3),
        stage3_yaw_rate_range=(-1.0, 1.0),
        resampling_time_range=(COMMAND_RESAMPLING_TIME, COMMAND_RESAMPLING_TIME),
        rel_standing_envs=COMMAND_STANDING_ENV_RATIO,
        rel_yaw_envs=COMMAND_YAW_ENV_RATIO,
        rel_forward_vx_envs=COMMAND_FORWARD_VX_ENV_RATIO,
        exclusive_linear_yaw_commands=False,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        goal_arrow_height=0.75,
        current_arrow_height=0.75,
        goal_yaw_arrow_height=0.95,
        current_yaw_arrow_height=0.82,
        yaw_arc_radius=0.38,
        yaw_arc_span=4.2,
        yaw_arc_segments=5,
        yaw_arc_max_rate=3.0,
        goal_vel_visualizer_cfg=COMMAND_GOAL_VEL_MARKER_CFG,
        current_vel_visualizer_cfg=COMMAND_CURRENT_VEL_MARKER_CFG,
        goal_yaw_visualizer_cfg=COMMAND_GOAL_YAW_MARKER_CFG,
        current_yaw_visualizer_cfg=COMMAND_CURRENT_YAW_MARKER_CFG,
        ranges=wheel_leg_commands.WheelLegVelocityCommandCfg.Ranges(
            lin_vel_x=COMMAND_LIN_VEL_X_RANGE,
            lin_vel_y=COMMAND_LIN_VEL_Y_RANGE,
            ang_vel_z=COMMAND_ANG_VEL_Z_RANGE,
        ),
    )

    base_height = wheel_leg_commands.UniformRandomBaseHeightCommandCfg(
        asset_name="robot",
        # 当前 curriculum 阶段每个 episode 给两段高度：
        # reset 时随机一个高度，time-out 一半时再随机一次。
        resampling_time_range=(BASE_HEIGHT_RESAMPLING_TIME, BASE_HEIGHT_RESAMPLING_TIME),
        height_range=BASE_HEIGHT_COMMAND_RANGE,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Actor obs layout: 3 + 3 + 3 + 2 + 1 + 4 + 6 + 6 = 28 dims.
        # With history_length=5 and flatten_history_dim=True, actor input is 140 dims.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_commands = ObsTerm(
            func=wheel_leg_observations.velocity_command_x_yaw,
            params={
                "command_name": "base_velocity",
                "lin_vel_scale": 2.0,
                "yaw_rate_scale": 0.25,
            },
        )

        height_command = ObsTerm(
            func=wheel_leg_observations.base_height_command,
            params={
                "command_name": "base_height",
                "height_scale": 1.0,
            },
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True)},
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
        )

        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Stage 2 forward walking rewards.

    当前阶段把 vx tracking 作为主任务，同时要求 vx 主要由轮子公共速度产生。
    高度和 roll/pitch 姿态是完成速度任务的约束，但不能再次压过 vx 主任务。
    """

    # Stage 2 main vx tracking:
    # reward = exp(-((vx-vx_cmd)^2)/(sigma^2))。最大贡献 3.0，是当前主 reward。
    vx_tracking = RewTerm(
        func=wheel_leg_rewards.vx_tracking_gaussian,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sigma": VX_TRACKING_SIGMA,
        },
    )

    # Stage 2 vx Huber error:
    # 大误差时近似 L1，避免 Gaussian 在训练初期远离目标时梯度过小。
    vx_tracking_huber = RewTerm(
        func=wheel_leg_rewards.vx_tracking_huber,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "beta": VX_TRACKING_HUBER_BETA,
            "max_value": VX_TRACKING_HUBER_CLIP,
        },
    )

    # Stage 2 wheel-vx tracking:
    # 轮子公共速度换算出的 vx 也要跟随 command，避免 policy 用腿部摆动/蹭地制造 base vx。
    wheel_vx_tracking = RewTerm(
        func=wheel_leg_rewards.wheel_vx_tracking_gaussian,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "wheel_radius": WHEEL_RADIUS,
            "sigma": WHEEL_VX_TRACKING_SIGMA,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True),
        },
    )

    # Stage 2 wheel-vx Huber:
    # 大误差时仍推动轮子朝正确方向输出，不让 Gaussian 远处梯度消失。
    wheel_vx_tracking_huber = RewTerm(
        func=wheel_leg_rewards.wheel_vx_tracking_huber,
        weight=-0.3,
        params={
            "command_name": "base_velocity",
            "wheel_radius": WHEEL_RADIUS,
            "beta": WHEEL_VX_TRACKING_HUBER_BETA,
            "max_value": WHEEL_VX_TRACKING_HUBER_CLIP,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True),
        },
    )

    # Stage 2 base/wheel consistency:
    # 如果 base vx 和轮子滚动速度差很多，通常说明腿在蹭速度或轮子滑动，轻惩罚即可。
    base_wheel_vx_consistency = RewTerm(
        func=wheel_leg_rewards.base_wheel_vx_consistency_huber,
        weight=-0.2,
        params={
            "wheel_radius": WHEEL_RADIUS,
            "beta": BASE_WHEEL_VX_CONSISTENCY_BETA,
            "max_value": BASE_WHEEL_VX_CONSISTENCY_CLIP,
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True),
        },
    )

    # Stage 2 roll/pitch orientation:
    # pitch/roll 都要稳定；否则容易出现靠腿摆动前进、车体姿态很差的策略。
    orientation_tracking = RewTerm(
        func=wheel_leg_rewards.orientation_exp_kernel,
        weight=1.5,
        params={
            "kernel_coeff": STAGE1_ORIENTATION_KERNEL,
        },
    )

    # Stage 2 pitch auxiliary balance:
    # 保留一个较弱 pitch 单项，让倒立摆方向更稳，但仍低于 vx 主任务。
    pitch_tracking = RewTerm(
        func=wheel_leg_rewards.pitch_tracking_gaussian,
        weight=0.5,
        params={"sigma": PITCH_TRACKING_SIGMA},
    )

    # Stage 2 pitch-rate penalty:
    # 抑制大幅前后摆动，但权重较小，避免阻止通过身体倾斜加速。
    pitch_rate = RewTerm(
        func=wheel_leg_rewards.pitch_rate_l2,
        weight=-0.05,
        params={"max_value": PITCH_RATE_REWARD_CLIP},
    )

    # Stage 2 wheel action-rate penalty:
    # 轮子需要主动运动完成 vx，不要把该项设太大；这里只抑制高频反转。
    wheel_action_rate = RewTerm(
        func=wheel_leg_rewards.action_rate_l2,
        weight=-0.005,
        params={
            "action_slice": (4, 6),
            "max_value": ACTION_RATE_REWARD_CLIP,
        },
    )

    # Stage 2 height auxiliary:
    # 高度误差 5~7cm 时必须还有明显收益，否则 policy 容易只顾 vx 而牺牲构型。
    base_height_tracking = RewTerm(
        func=wheel_leg_rewards.base_height_command_exp_kernel,
        weight=2.0,
        params={
            "command_name": "base_height",
            "kernel_coeff": STAGE1_HEIGHT_KERNEL,
        },
    )

    # Stage 2 fine height tracking:
    # 接近目标高度后提供厘米级精细收敛；远离目标时主要交给上面的 coarse 项。
    base_height_fine = RewTerm(
        func=wheel_leg_rewards.base_height_command_tracking_fudan,
        weight=1.0,
        params={
            "command_name": "base_height",
        },
    )

    # Stage 2 yaw-rate auxiliary:
    # 直线 vx 阶段 yaw_cmd=0，弱惩罚转圈；不要强到压制速度跟踪。
    yaw_rate_penalty = RewTerm(
        func=wheel_leg_rewards.yaw_rate_l2_when_no_yaw_command,
        weight=-0.2,
        params={
            "command_name": "base_velocity",
            "command_deadband": 0.05,
            "yaw_rate_deadband": 0.02,
            "max_value": 4.0,
        },
    )

    # Stage 1 torque penalty:
    # penalty = sum(torque^2), weight = -0.0001.
    torques = RewTerm(
        func=wheel_leg_rewards.joint_torques_l2,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True),
            "max_value": JOINT_TORQUE_REWARD_CLIP,
        },
    )

    # Stage 1 base contact penalty:
    # orientation 水平无法区分"站着水平"和"趴地水平"，因此 base_link 碰地需要直接扣分。
    base_contact = RewTerm(
        func=wheel_leg_rewards.contact_sensor_contact,
        weight=-2.0,
        params={
            "threshold": BASE_CONTACT_REWARD_FORCE_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="base_link"),
        },
    )

    # Stage 1 standing-only reward，Stage 2 速度学习第一版先关闭，避免"原地不动"压过 vx tracking。
    root_position = RewTerm(
        func=wheel_leg_rewards.root_xy_position_l2_when_no_linear_command,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "vx_command_deadband": 0.05,
            "deadband": ROOT_XY_POSITION_DEADBAND,
        },
    )

    # Stage 1 standing-only reward，Stage 2 速度学习第一版先关闭。
    com_alignment = RewTerm(
        func=wheel_leg_rewards.com_wheel_alignment_l2_when_no_linear_command,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "vx_command_deadband": 0.05,
            "deadband": COM_WHEEL_ALIGNMENT_DEADBAND,
            "error_scale": COM_WHEEL_ALIGNMENT_ERROR_SCALE,
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_KINEMATICS_BODIES, preserve_order=True),
        },
    )

    # Stage 2 leg action smooth:
    # 轻微抑制腿部高频摆动，减少“靠腿蹭速度”；权重保持小，避免影响高度调节。
    leg_action_smooth = RewTerm(
        func=wheel_leg_rewards.action_second_order_l2,
        weight=-0.01,
        params={
            "action_slice": (0, 4),
            "max_value": ACTION_SMOOTH_REWARD_CLIP,
        },
    )

    # Stage 2 第一版用 termination 兜底极端劈叉，soft penalty 先关闭，避免限制加速探索。
    wheel_x_alignment = RewTerm(
        func=wheel_leg_rewards.wheel_x_separation_l2,
        weight=0.0,
        params={
            "deadband": 0.03,
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_KINEMATICS_BODIES, preserve_order=True),
        },
    )


@configclass
class TerminationsCfg:
    """Episode termination conditions for flat-ground wheel-leg training."""

    # 达到 episode 最大时长时正常结束；timeout 不应该算失败。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # base_link 碰地时终止。
    base_contact = DoneTerm(
        func=wheel_leg_terminations.illegal_contact_after_steps,
        params={
            "threshold": BASE_CONTACT_TERMINATION_FORCE_THRESHOLD,
            "min_steps": BASE_CONTACT_GRACE_STEPS,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="base_link"),
        },
    )

    # 机身高度长期没有撑起来时终止。
    low_base_height = DoneTerm(
        func=wheel_leg_terminations.root_height_below_minimum_after_steps,
        params={
            "minimum_height": MIN_BASE_HEIGHT,
            "min_steps": LOW_BASE_HEIGHT_GRACE_STEPS,
        },
    )

    leg_joint_deviation = DoneTerm(
        func=wheel_leg_terminations.joint_position_deviation_after_steps,
        params={
            "maximum_deviation": MAX_LEG_JOINT_DEVIATION,
            "min_steps": LEG_JOINT_DEVIATION_GRACE_STEPS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True),
        },
    )

    # 左右轮在 base x 方向错开太多时终止，堵死劈叉策略。
    wheel_x_separation = DoneTerm(
        func=wheel_leg_terminations.wheel_x_separation_after_steps,
        params={
            "maximum_separation": MAX_WHEEL_X_SEPARATION,
            "min_steps": WHEEL_X_SEPARATION_GRACE_STEPS,
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_KINEMATICS_BODIES, preserve_order=True),
        },
    )

    # Fudan plane 风格：真正接近翻倒并持续一段时间后再 reset。
    bad_orientation = DoneTerm(
        func=wheel_leg_terminations.bad_orientation_after_steps,
        params={
            "limit_angle": BAD_ORIENTATION_LIMIT_ANGLE,
            "min_steps": BAD_ORIENTATION_GRACE_STEPS,
        },
    )

    root_state_non_finite = DoneTerm(
        func=wheel_leg_terminations.root_state_non_finite,
    )


@configclass
class EventCfg:
    """Reset events for the wheel-legged robot.

    保持接近 Fudan plane 任务的 reset：关节回默认位，root 只做小扰动。
    """

    # 重置机身位置、姿态和速度。
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": RESET_ROOT_POS_X_RANGE,
                "y": RESET_ROOT_POS_Y_RANGE,
                "z": RESET_ROOT_POS_Z_RANGE,
                "yaw": RESET_ROOT_YAW_RANGE,
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.02, 0.02),
                "roll": (-0.02, 0.02),
                "pitch": (-0.02, 0.02),
                "yaw": (-0.02, 0.02),
            },
        },
    )

    # 关节回默认位，速度清零。
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True),
            "position_range": RESET_JOINT_POS_RANGE,
            "velocity_range": RESET_JOINT_VEL_RANGE,
        },
    )

@configclass
class WheelLegEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based RL environment configuration for the wheel-legged robot."""

    scene: WheelLegSceneCfg = WheelLegSceneCfg(
        num_envs=64,
        env_spacing=2.0,
        clone_in_fabric=False,
        replicate_physics=False,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        # 平地任务：20s episode，速度/高度命令每 5s 随机重采样。
        self.episode_length_s = 20.0

        self.viewer.eye = (2.0, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class WheelLegPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO configuration for the base wheel-legged environment."""

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "wheel_leg"
    run_name = ""
    clip_actions = 1.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
