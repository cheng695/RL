"""ANYmal-style ray sensing and terrain levels, specialized for single steps."""
import isaaclab.terrains as terrains
from isaaclab.managers import ObservationTermCfg, CurriculumTermCfg, TerminationTermCfg, RewardTermCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from ...mdp import step_course
from .steps_env_cfg import MyRobotStepsEnvCfg


@configclass
class MyRobotStepCourseEnvCfg(MyRobotStepsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator = terrains.TerrainGeneratorCfg(
            seed=42, size=(8., 8.), border_width=10., num_rows=8, num_cols=20,
            curriculum=True, use_cache=False,
            sub_terrains={
                "flat": terrains.MeshPlaneTerrainCfg(proportion=.30),
                "step_up": terrains.MeshPitTerrainCfg(proportion=.35, function=step_course.course_step,
                    pit_depth_range=(.025, .20), platform_width=2.),
                "step_down": terrains.MeshBoxTerrainCfg(proportion=.35, function=step_course.course_step,
                    box_height_range=(.025, .20), platform_width=2.),
            },
        )
        self.scene.terrain_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 20.)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=.2, size=[1.6, .8], ordering="yx"),
            mesh_prim_paths=["/World/ground"], update_period=self.decimation * self.sim.dt,
        )
        # New term appended after the existing 49D terms; do not reactivate inherited height_scan.
        self.observations.policy.terrain_scan = ObservationTermCfg(func=step_course.terrain_scan)
        self.commands.base_velocity.class_type = step_course.StepCourseCommand
        # Flat columns retain the learned command range; obstacle attempts are overridden to slow straight motion.
        self.commands.base_velocity.ranges.lin_vel_x = (-1., 1.)
        self.commands.base_velocity.ranges.ang_vel_z = (-5., 5.)
        # Each obstacle attempt keeps its initial forward/reverse direction for the episode.
        self.commands.base_velocity.resampling_time_range = (30., 30.)
        self.curriculum.terrain_levels = CurriculumTermCfg(func=step_course.step_levels)
        self.terminations.step_success = TerminationTermCfg(func=step_course.crossed_and_settled)
        self.rewards.step_success = RewardTermCfg(func=step_course.crossing_bonus, weight=50.)
        self.rewards.base_height_tracking.func = step_course.adaptive_height_exp
        self.rewards.base_height_tracking.params["kernel_coeff"] = 1000.
        self.rewards.base_height_l2.func = step_course.adaptive_height_l2
        self.rewards.base_height_l2.weight = -1.
        self.rewards.orientation_tracking.func = step_course.adaptive_orientation
        self.rewards.orientation_tracking.weight = 3.


@configclass
class MyRobotStepCourseEnvCfg_PLAY(MyRobotStepCourseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.curriculum.terrain_levels = None
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
