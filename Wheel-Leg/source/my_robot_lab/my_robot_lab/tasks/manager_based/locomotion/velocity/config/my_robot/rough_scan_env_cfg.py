"""Gentle roughness and waves for transferring the UZ05 flat policy."""
import isaaclab.terrains as terrains
from isaaclab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import terrain_levels_vel
from ...mdp import rough_posture, step_course
from ...mdp.gentle_terrain import progressive_random_rough
from .steps_env_cfg import MyRobotStepsEnvCfg


@configclass
class ProgressiveRoughTerrainCfg(terrains.HfRandomUniformTerrainCfg):
    # Unlike the upstream random terrain, this amplitude increases with difficulty.
    function = progressive_random_rough
    amplitude_range: tuple[float, float] = (0.003, 0.025)
    noise_range = (-0.025, 0.025)
    noise_step = 0.001
    downsampled_scale = 0.25


def initialized_terrain_levels(env, env_ids):
    """Evaluate distance only on straight moving episodes; circles are not failures."""
    ids = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(ids):
        command = env.command_manager.get_command("base_velocity")[ids]
        ids = ids[(command[:, 0].abs() > .05) & (command[:, 2].abs() < 1.e-6)]
        if len(ids):
            terrain_levels_vel(env, ids)
    return env.scene.terrain.terrain_levels.float().mean()


@configclass
class MyRobotRoughScanEnvCfg(MyRobotStepsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = terrains.TerrainGeneratorCfg(
            seed=42, size=(8., 8.), border_width=10., num_rows=10, num_cols=20,
            horizontal_scale=0.05, vertical_scale=0.001,
            curriculum=True, use_cache=False,
            sub_terrains={
                "flat": terrains.MeshPlaneTerrainCfg(proportion=0.30),
                "random_rough": ProgressiveRoughTerrainCfg(proportion=0.35),
                # Installed Isaac Lab uses half this amplitude for each sinusoid;
                # combined peak-to-peak height is approximately 2 * amplitude.
                "waves": terrains.HfWaveTerrainCfg(
                    proportion=0.35, amplitude_range=(0.005, 0.050), num_waves=2),
            },
        )
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 20.)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=.2, size=[1.6, .8], ordering="yx"),
            mesh_prim_paths=["/World/ground"], update_period=self.decimation * self.sim.dt,
        )
        # Preserve the flat 49D input prefix and append the same 45D scan as before.
        self.observations.policy.terrain_scan = ObservationTermCfg(func=step_course.terrain_scan)
        self.commands.base_velocity.ranges.lin_vel_x = (-1., 1.)
        self.commands.base_velocity.ranges.ang_vel_z = (-3., 3.)
        # Keep direction for the episode so displacement-based curriculum is meaningful.
        self.commands.base_velocity.resampling_time_range = (30., 30.)
        self.commands.base_velocity.straight_command_prob = .20
        self.commands.base_velocity.turn_command_prob = .20
        self.commands.base_velocity.mixed_command_prob = .60
        self.commands.base_velocity.rel_standing_envs = .20
        self.commands.base_velocity.min_abs_lin_vel_x = .15
        self.commands.base_height.height_range = (.30, .36)

        # Keep flat tracking rewards; allow suspension travel and natural vertical motion.
        self.rewards.base_height_tracking.params["kernel_coeff"] = 300.
        self.rewards.base_height_l2.weight = -.5
        self.rewards.orientation_tracking.func = rough_posture.gentle_orientation
        self.rewards.orientation_tracking.weight = 3.
        # Gentle waves do not need stair-climbing pitch freedom (formerly 5--15 deg).
        self.rewards.orientation_tracking.params.update(
            flat_tolerance=0.05236, obstacle_tolerance=0.13963)
        self.rewards.rough_pitch = RewardTermCfg(
            func=rough_posture.pitch_deadband, weight=-.5,
            params={"flat_tolerance": 0.05236, "obstacle_tolerance": 0.13963})
        self.rewards.ang_vel_xy_l2.weight = -.15
        self.rewards.leg_action_smooth.weight = -.04
        self.rewards.failure = RewardTermCfg(func=rough_posture.failure_event, weight=-5.)
        self.rewards.lin_vel_z_l2.weight = -.5
        self.rewards.leg_joint_vel_l2.weight = -.005
        self.rewards.wheel_split = RewardTermCfg(
            func=rough_posture.wheel_fore_aft_split, weight=-1.,
            params={"wheel_cfg": SceneEntityCfg("robot", body_names=[
                "left_right_wheel", "right_right_wheel"], preserve_order=True),
                    "deadband": .04, "scale": .10, "linear_tail": True, "max_value": None})
        # Normal reset/contact/tilt handling is inherited; no stair state or event rewards.
        self.curriculum.terrain_levels = CurriculumTermCfg(func=initialized_terrain_levels)


@configclass
class MyRobotRoughScanEnvCfg_PLAY(MyRobotRoughScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.curriculum.terrain_levels = None
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
