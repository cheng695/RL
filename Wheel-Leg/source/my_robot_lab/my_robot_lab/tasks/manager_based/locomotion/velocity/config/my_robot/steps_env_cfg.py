"""Fixed 150/200 mm obstacles; separate from the flat baseline."""
import isaaclab.terrains as terrain_gen
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .flat_env_cfg import MyRobotFlatEnvCfg


@configclass
class MyRobotStepsEnvCfg(MyRobotFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
            seed=42, size=(8.0, 8.0), border_width=10.0,
            num_rows=4, num_cols=10, horizontal_scale=0.1, vertical_scale=0.005,
            # Deterministic columns for each type; all rows use the exact heights below.
            curriculum=True, use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "step_down_150mm": terrain_gen.MeshBoxTerrainCfg(
                    proportion=0.1, box_height_range=(0.15, 0.15), platform_width=2.0),
                "step_down_200mm": terrain_gen.MeshBoxTerrainCfg(
                    proportion=0.1, box_height_range=(0.20, 0.20), platform_width=2.0),
                "step_up_150mm": terrain_gen.MeshPitTerrainCfg(
                    proportion=0.1, pit_depth_range=(0.15, 0.15), platform_width=2.0),
                "step_up_200mm": terrain_gen.MeshPitTerrainCfg(
                    proportion=0.1, pit_depth_range=(0.20, 0.20), platform_width=2.0),
                "stairs_150mm": terrain_gen.MeshPyramidStairsTerrainCfg(
                    proportion=0.1, step_height_range=(0.15, 0.15), step_width=0.40,
                    platform_width=2.0, border_width=1.0),
                "stairs_200mm": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                    proportion=0.1, step_height_range=(0.20, 0.20), step_width=0.40,
                    platform_width=2.0, border_width=1.0),
                "waves": terrain_gen.HfWaveTerrainCfg(
                    proportion=0.2, amplitude_range=(0.01, 0.03), num_waves=3),
            },
        )
        # Small footprint estimates ground under the body, not the upcoming stair top.
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]),
            mesh_prim_paths=["/World/ground"], update_period=self.decimation * self.sim.dt,
        )
        self.commands.base_height.ground_sensor_name = "height_scanner"
        for name in ("base_height", "base_height_error"):
            getattr(self.observations.policy, name).params["height_reference_command"] = "base_height"
        for name in ("base_height_tracking", "base_height_l2"):
            getattr(self.rewards, name).params["relative_to_ground"] = True
        # A stair edge changes the local height reference abruptly; avoid opposing the climb.
        self.rewards.base_height_tracking.params["kernel_coeff"] = 200.0
        self.rewards.base_height_l2.weight = -0.25
        self.rewards.straight_pitch = None
        self.rewards.orientation_tracking.weight = 1.5
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)
        self.commands.base_height.height_range = (0.30, 0.40)
        self.events.push_robot = None


@configclass
class MyRobotStepsEnvCfg_PLAY(MyRobotStepsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        # Keep the complete training map, including flat and wave columns.
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
