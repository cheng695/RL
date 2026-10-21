from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ImuCfg
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
    WHEEL_BODY_NAMES,
    WHEEL_JOINTS,
)
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import commands as my_commands
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp.actions import (
    RateLimitedJointPositionActionCfg,
    RateLimitedJointVelocityActionCfg,
)
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import observations as my_observations
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import rewards as my_rewards
from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp import terminations as my_terminations

@configclass
class MySceneCfg(IsaacLabVelocitySceneCfg):
    """Scene configuration for the custom wheel-leg velocity task."""

    # Flat collision plane required by the physics scene.
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
    imu = ImuCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{BASE_BODY_NAME}/chassis",
        gravity_bias=(0.0, 0.0, 0.0),
    )
    height_scanner = None

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
        # Joint translation/turning training: 20% standing, 40% straight,
        # 20% turn-in-place, 20% moving while turning. Mode probabilities are conditional
        # on the non-standing 80% of samples.
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        # Match wheel_joint_vel.scale=50 with policy actions clipped to +/-1.
        max_wheel_speed=50.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.5, 2.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-12.0, 12.0),
            heading=None,
        ),
        straight_command_prob=0.5,
        turn_command_prob=0.25,
        mixed_command_prob=0.25,
        positive_lin_vel_x_prob=0.5,
        zero_lin_vel_x_prob=0.0,
        min_abs_lin_vel_x=0.08,
        low_speed_max_abs_lin_vel_x=0.5,
        medium_speed_max_abs_lin_vel_x=1.0,
        # Low-speed emphasis for training the mass experiment from scratch.
        low_speed_command_prob=0.6,
        medium_speed_command_prob=0.25,
        high_speed_command_prob=0.15,
        min_abs_ang_vel_z=0.15,
        low_yaw_max_abs_ang_vel_z=1.0,
        medium_yaw_max_abs_ang_vel_z=2.5,
        high_yaw_max_abs_ang_vel_z=5.0,
        low_yaw_command_prob=0.6,
        medium_yaw_command_prob=0.25,
        high_yaw_command_prob=0.1,
        very_high_yaw_command_prob=0.05,
        abrupt_flip_prob=0.0,
        abrupt_flip_min_abs_lin_vel_x=0.6,
        abrupt_flip_max_abs_lin_vel_x=1.0,
    )

    base_height = my_commands.UniformBaseHeightCommandCfg(
        asset_name="robot",
        # Sample once on reset and hold the target for the entire episode.
        resampling_time_range=(1.0e6, 1.0e6),
        height_range=(0.25, 0.4),
        endpoint_switch_prob=0.0,
    )


@configclass
class ActionsCfg(IsaacLabActionsCfg):
    """Action specifications for the wheel-leg robot."""

    joint_pos = None

    leg_joint_pos = RateLimitedJointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
        max_position_rate=4.0,
    )

    wheel_joint_vel = RateLimitedJointVelocityActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINTS,
        scale=50.0,
        use_default_offset=True,
        preserve_order=True,
        max_velocity_rate=100.0,
    )


@configclass
class ObservationsCfg(IsaacLabObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=my_observations.imu_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        base_ang_vel = ObsTerm(
            func=my_observations.imu_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=my_observations.imu_projected_gravity,
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

    # Keep the MJCF nominal total mass (15.8 kg) fixed while isolating tracking.
    add_base_mass = None

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
class RewardsCfg:
    """Complete reward definition for the flat wheel-leg task."""

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

    yaw_tracking_huber = RewTerm(
        func=my_rewards.yaw_tracking_huber,
        weight=-1.2,
        params={"command_name": "base_velocity", "beta": 0.35, "max_value": 6.0},
    )

    lateral_velocity = RewTerm(
        func=my_rewards.lateral_velocity_l2,
        weight=-0.5,
        params={"max_value": 4.0},
    )

    stand_still_drift = RewTerm(
        func=my_rewards.stand_still_drift_huber,
        weight=-0.5,
        params={
            "command_name": "base_velocity",
            "command_deadband": 0.02,
            "velocity_scale": 0.10,
            "max_value": 5.0,
        },
    )

    orientation_tracking = RewTerm(
        func=my_rewards.orientation_exp_kernel,
        weight=2.0,
        params={"kernel_coeff": 20.0},
    )

    roll_pitch_penalty = RewTerm(
        func=my_rewards.roll_pitch_deadband_l2,
        weight=-2.0,
        params={
            "roll_deadband_rad": math.radians(3.0),
            "pitch_deadband_rad": math.radians(3.0),
            "error_scale": 0.15,
            "max_value": 5.0,
        },
    )

    straight_pitch_penalty = RewTerm(
        func=my_rewards.straight_pitch_deadband_l2,
        # Temporarily disable the extra straight-line lean cost during fine-tuning.
        # General orientation_tracking and roll_pitch_penalty remain active.
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "height_command_name": "base_height",
            "settle_time": 0.75,
            "deadband_rad": math.radians(5.0),
            "error_scale": 0.12,
            "max_value": 6.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    base_height_tracking = RewTerm(
        func=my_rewards.base_height_command_exp_kernel,
        weight=4.0,
        params={
            "command_name": "base_height",
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
            "kernel_coeff": 300.0,
        },
    )

    base_height_l2 = RewTerm(
        func=my_rewards.base_height_command_l2,
        weight=-2.0,
        params={
            "command_name": "base_height",
            "error_scale": 0.05,
            "max_value": 16.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY_NAME),
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

    leg_action_smooth = RewTerm(
        func=my_rewards.action_second_order_l2,
        weight=-0.02,
        params={"action_slice": (0, 4), "max_value": 1.0e3},
    )

    wheel_action_smooth = RewTerm(
        func=my_rewards.action_second_order_l2,
        weight=-0.01,
        params={"action_slice": (4, 6), "max_value": 1.0e3},
    )

    wheel_target_smooth = RewTerm(
        func=my_rewards.wheel_target_delta_l2,
        weight=-0.1,
        params={
            "action_name": "wheel_joint_vel",
            # At dt=0.02 s, 100 rad/s² permits 2 rad/s target change per step.
            "delta_scale": 2.0,
            "max_value": 4.0,
        },
    )

    wheel_split = RewTerm(
        func=my_rewards.wheel_fore_aft_split_l2,
        weight=-4.0,
        params={
            "deadband": 0.04,
            "error_scale": 0.10,
            "max_value": 4.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES, preserve_order=True),
        },
    )


@configclass
class TerminationsCfg(IsaacLabTerminationsCfg):
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Disable Isaac Lab's inherited termination, which looks for a body named "base".
    base_contact = None
    non_wheel_contact = DoneTerm(
        func=my_terminations.non_wheel_ground_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "threshold": 1.0,
            "min_steps": 10,
            "wheel_body_names": tuple(WHEEL_BODY_NAMES),
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
    """Empty curriculum for the flat velocity task."""

    terrain_levels = None


@configclass
class LocomotionVelocityEnvCfg(IsaacLabLocomotionVelocityRoughEnvCfg):
    """Velocity task config that keeps Isaac Lab defaults and overrides the scene."""

    scene: MySceneCfg = MySceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
