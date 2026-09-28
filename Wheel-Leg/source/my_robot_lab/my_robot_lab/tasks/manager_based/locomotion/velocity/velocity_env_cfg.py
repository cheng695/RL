from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg as IsaacLabActionsCfg,
    CommandsCfg as IsaacLabCommandsCfg,
    CurriculumCfg as IsaacLabCurriculumCfg,
    EventCfg as IsaacLabEventCfg,
    LocomotionVelocityRoughEnvCfg as IsaacLabLocomotionVelocityRoughEnvCfg,
    ObservationsCfg as IsaacLabObservationsCfg,
    RewardsCfg as IsaacLabRewardsCfg,
    TerminationsCfg as IsaacLabTerminationsCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg as IsaacLabVelocitySceneCfg
from my_robot_lab.assets.robots.my_robot_ import (
    BASE_CONTACT_BODY_NAMES,
    BASE_BODY_NAME,
    CONTROLLED_JOINTS,
    INITIAL_BASE_HEIGHT_RANGE,
    LEG_EXTENSION_BODY_NAMES,
    LEG_JOINTS,
    WHEEL_JOINTS,
)
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import commands as my_commands
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as my_observations
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards as my_rewards
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import terminations as my_terminations

@configclass
class MySceneCfg(IsaacLabVelocitySceneCfg):
    """Scene configuration for the custom wheel-leg velocity task."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    height_scanner = RayCasterCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY_NAME}",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg(IsaacLabCommandsCfg):
    """Command specifications for the wheel-leg velocity task."""

    base_velocity = my_commands.PositiveBiasedVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(2.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-5.0, 5.0),
            heading=None,
        ),
        straight_command_prob=0.5,
        turn_command_prob=0.3,
        mixed_command_prob=0.2,
        # Mixed commands are projected by wheel speed, not component-clipped.
        mixed_yaw_limit=None,
        positive_lin_vel_x_prob=0.5,
        zero_lin_vel_x_prob=0.0,
        min_abs_lin_vel_x=0.08,
        min_abs_ang_vel_z=0.15,
        abrupt_flip_prob=0.0,
        abrupt_flip_min_abs_lin_vel_x=0.6,
        abrupt_flip_max_abs_lin_vel_x=1.0,
    )

    base_height = my_commands.UniformBaseHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        height_range=(0.25, 0.40),
        endpoint_switch_prob=0.7,
    )


@configclass
class ActionsCfg(IsaacLabActionsCfg):
    """Action specifications for the wheel-leg robot."""

    joint_pos = None

    leg_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.9,
        use_default_offset=True,
        preserve_order=True,
    )

    wheel_joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=20.0,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg(IsaacLabObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        base_height = ObsTerm(
            func=my_observations.base_height,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME)},
        )
        base_height_error = ObsTerm(
            func=my_observations.base_height_error,
            params={
                "command_name": "base_height",
                "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            },
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg(IsaacLabEventCfg):
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),  # Offset from the default root height.
                "yaw": (-0.1, 0.1),
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg(IsaacLabRewardsCfg):
    """Reward terms for the MDP."""

    # -- old wheel-leg velocity rewards
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None

    vx_tracking = RewTerm(
        func=my_rewards.vx_tracking_gaussian,
        # Prioritize translation after mass correction; retain the same precision kernel.
        weight=4.0,
        params={"command_name": "base_velocity", "sigma": 0.35},
    )

    vx_tracking_huber = RewTerm(
        func=my_rewards.vx_tracking_huber,
        # Retain a stronger error signal outside the Gaussian's useful range.
        weight=-1.2,
        params={"command_name": "base_velocity", "beta": 0.3, "max_value": 2.0},
    )

    yaw_tracking = RewTerm(
        func=my_rewards.yaw_tracking_gaussian,
        weight=4.0,
        params={"command_name": "base_velocity", "sigma": 0.45},
    )

    yaw_tracking_fine = None

    yaw_tracking_huber = RewTerm(
        func=my_rewards.yaw_tracking_huber,
        weight=-0.75,
        params={"command_name": "base_velocity", "beta": 0.35, "max_value": 6.0},
    )

    # Use measured base yaw for tracking until the imported wheel joint signs
    # have been verified in Isaac Lab. A wrong wheel constraint opposes turning.
    wheel_yaw_rate_consistency = None

    wheel_vx_tracking = None
    wheel_vx_tracking_huber = None
    base_wheel_vx_consistency = None

    orientation_tracking = RewTerm(
        func=my_rewards.orientation_exp_kernel,
        weight=3.0,
        params={"kernel_coeff": 20.0},
    )

    # redundant with orientation_tracking / ang_vel_xy_l2
    pitch_tracking = None
    pitch_rate = None

    straight_pitch = RewTerm(
        func=my_rewards.straight_pitch_deadband_l2,
        weight=-0.5,
        params={"command_name": "base_velocity", "height_command_name": "base_height",
                "settle_time": 1.0, "deadband_rad": 0.05236,
                "error_scale": 0.1, "max_value": 4.0},
    )

    base_height_tracking = RewTerm(
        func=my_rewards.base_height_command_exp_kernel,
        weight=3.0,
        params={
            "command_name": "base_height",
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "kernel_coeff": 1000.0,
        },
    )

    # redundant with base_height_tracking / base_height_l2
    base_height_fine = None

    base_height_l2 = RewTerm(
        func=my_rewards.base_height_command_l2,
        weight=-1.0,
        params={
            "command_name": "base_height",
            "error_scale": 0.05,
            "max_value": 9.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
        },
    )

    yaw_rate_penalty = RewTerm(
        func=my_rewards.yaw_rate_l2_when_no_yaw_command,
        weight=-0.05,
        params={
            "command_name": "base_velocity",
            "command_deadband": 0.05,
            "yaw_rate_deadband": 0.02,
            "max_value": 4.0,
        },
    )

    stand_still_lin_vel = RewTerm(
        func=my_rewards.stand_still_lin_vel_l2,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "command_deadband": 0.08,
            "max_value": 4.0,
        },
    )

    stand_still_drift = RewTerm(
        func=my_rewards.stand_still_drift_huber,
        weight=-0.2,
        params={"command_name": "base_velocity", "command_deadband": 1.0e-6,
                "velocity_scale": 0.05, "max_value": 10.0},
    )

    torques = RewTerm(
        func=my_rewards.joint_torques_l2,
        # Relax effort regularization while learning height and vx tracking.
        weight=-5.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True),
            "max_value": 1.0e5,
        },
    )

    wheel_power = RewTerm(
        func=my_rewards.wheel_power_l1_positive,
        weight=-1.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS, preserve_order=True),
            "max_value": 1.0e4,
        },
    )

    base_contact = RewTerm(
        func=my_rewards.contact_sensor_contact,
        weight=-2.0,
        params={
            "threshold": 0.1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_CONTACT_BODY_NAMES),
        },
    )

    wheel_action_rate = RewTerm(
        func=my_rewards.action_rate_l2,
        weight=-0.02,
        params={"action_slice": (4, 6), "max_value": 1.0e3},
    )

    leg_action_smooth = RewTerm(
        func=my_rewards.action_second_order_l2,
        weight=-0.03,
        params={"action_slice": (0, 4), "max_value": 1.0e3},
    )

    wheel_action_smooth = RewTerm(
        func=my_rewards.action_second_order_l2,
        weight=-0.01,
        params={"action_slice": (4, 6), "max_value": 1.0e3},
    )

    # -- anti-squat: penalize motion of the leg joints themselves
    # Keep this weak: leg motion also supports height changes and recovery.
    leg_joint_vel_l2 = RewTerm(
        func=my_rewards.leg_joint_vel_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINTS, preserve_order=True),
            "max_value": 1.0e3,
        },
    )

    # Optional height-dependent relaxation: disable the plain term first.
    # At equal weights this gate can only reduce the penalty, not strengthen it;
    # active base_height command transitions stay cheap.
    # leg_joint_vel_l2_height_gated = RewTerm(
    #     func=my_rewards.leg_joint_vel_l2_height_gated,
    #     weight=-0.01,
    #     params={
    #         "command_name": "base_height",
    #         "gate_sigma": 0.03,
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=LEG_JOINTS,
    #             body_names=BASE_BODY_NAME,
    #             preserve_order=True,
    #         ),
    #         "max_value": 1.0e3,
    #     },
    # )

    # -- inherited/default reward terms disabled or kept at zero
    track_base_height_exp = None
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = None
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg(IsaacLabTerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=my_terminations.illegal_contact_after_steps,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_CONTACT_BODY_NAMES),
            "threshold": 1.0,
            "min_steps": 100,
        },
    )
    bad_roll_pitch = DoneTerm(
        func=my_terminations.bad_roll_pitch,
        params={"asset_cfg": SceneEntityCfg("robot"), "limit_angle": 0.7},
    )
    leg_tendon_length = DoneTerm(
        func=my_terminations.leg_tendon_length_out_of_range,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEG_EXTENSION_BODY_NAMES, preserve_order=True),
            "min_length": 0.0,
            "max_length": 0.388,
            "tolerance": 0.005,
        },
    )


@configclass
class CurriculumCfg(IsaacLabCurriculumCfg):
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class LocomotionVelocityRoughEnvCfg(IsaacLabLocomotionVelocityRoughEnvCfg):
    """Velocity task config that keeps Isaac Lab defaults and overrides the scene."""

    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
