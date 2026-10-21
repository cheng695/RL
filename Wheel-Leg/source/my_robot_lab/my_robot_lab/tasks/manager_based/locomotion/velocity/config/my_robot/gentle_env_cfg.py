"""Gentle terrain transfer with unchanged policy observation dimensions."""
import math
import os
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.terrains as terrains
from .base_env_cfg import MyRobotEnvCfg
from ...mdp import terrain_stage


@configclass
class MyRobotGentleEnvCfg(MyRobotEnvCfg):
    terrain_relative_height: bool = True
    terrain_stage: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.terrain_stage = int(os.environ.get("UZ05_TERRAIN_STAGE", "1"))
        if self.terrain_stage not in (1, 2, 3, 4):
            raise ValueError("UZ05_TERRAIN_STAGE must be 1, 2, 3 or 4")
        amplitude = {1: .005, 2: .01, 3: .02, 4: .03}[self.terrain_stage]
        sub = {
            "flat": terrains.MeshPlaneTerrainCfg(proportion=.25),
            "rough": terrains.HfRandomUniformTerrainCfg(
                proportion=.75 if self.terrain_stage == 1 else .25,
                noise_range=(-amplitude, amplitude), noise_step=.0025,
                downsampled_scale=.2, border_width=.25),
        }
        if self.terrain_stage >= 2:
            slope = math.tan(math.radians({2: 3., 3: 10., 4: 17.}[self.terrain_stage]))
            sub.update({
                "slope_up": terrains.HfPyramidSlopedTerrainCfg(
                    proportion=.25, slope_range=(slope / 2, slope), platform_width=.4),
                "slope_down": terrains.HfInvertedPyramidSlopedTerrainCfg(
                    proportion=.25, slope_range=(slope / 2, slope), platform_width=.4),
            })
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrains.TerrainGeneratorCfg(
            seed=int(os.environ.get("UZ05_TERRAIN_SEED", "42")),
            size=(12., 12.), border_width=5., num_rows=4, num_cols=8,
            horizontal_scale=.1, vertical_scale=.0025, slope_threshold=.75,
            curriculum=True, use_cache=False, sub_terrains=sub)
        self.scene.terrain.max_init_terrain_level = None
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/chassis/chassis",
            offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 2.)), ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(.4, .4)),
            mesh_prim_paths=["/World/ground"], update_period=self.sim.dt * self.decimation,
            debug_vis=False)
        # Freeze difficulty within each run. Rows provide geometry diversity only.
        self.curriculum.terrain_levels = None
        self.terminations.invalid_ground_scan = DoneTerm(func=terrain_stage.invalid_ground_scan)
        self.terminations.terrain_tile_exit = DoneTerm(func=terrain_stage.terrain_tile_exit)
        # Keep full command coverage and inherit the low-speed-biased sampling.
        cmd = self.commands.base_velocity
        cmd.ranges.lin_vel_x = (-2.5, 2.5)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (-12.0, 12.0)
        cmd.debug_vis = False
        self.commands.base_height.height_range = (.25, .4)
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.episode_length_s = 10.
        # Keep horizontal gravity posture rewards and non-wheel contact failures.


@configclass
class MyRobotGentleEnvCfg_PLAY(MyRobotGentleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        # Interactive validation uses the original full command workspace.
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.ranges.lin_vel_x = (-2.5, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-12.0, 12.0)
        self.commands.base_height.height_range = (0.25, 0.4)

    def select_terrain(self, name, amplitude=None, slope_deg=None, seed=None):
        """Build one repeatable terrain type for interactive inspection."""
        if name == "mixed":
            if amplitude is not None or slope_deg is not None:
                raise ValueError("Choose a single terrain before overriding its geometry")
            if seed is not None:
                self.scene.terrain.terrain_generator.seed = seed
            return
        if name in ("stairs_up", "stairs_down"):
            raise ValueError("Stair terrain is reserved and not enabled yet")
        default_amplitude = {1: .005, 2: .01, 3: .02, 4: .03}[self.terrain_stage]
        amplitude = default_amplitude if amplitude is None else amplitude
        slope_deg = {1: 3., 2: 3., 3: 10., 4: 17.}[self.terrain_stage] if slope_deg is None else slope_deg
        if name == "flat":
            sub = terrains.MeshPlaneTerrainCfg(proportion=1.)
        elif name in ("rough", "undulating"):
            sub = terrains.HfRandomUniformTerrainCfg(
                proportion=1., noise_range=(-amplitude, amplitude), noise_step=.0025,
                downsampled_scale=.2 if name == "rough" else .8, border_width=.25)
        elif name in ("slope_up", "slope_down"):
            cls = (terrains.HfPyramidSlopedTerrainCfg if name == "slope_up"
                   else terrains.HfInvertedPyramidSlopedTerrainCfg)
            slope = math.tan(math.radians(slope_deg))
            sub = cls(proportion=1., slope_range=(slope, slope), platform_width=.4)
        else:
            raise ValueError(f"Unknown terrain: {name}")
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {name: sub}
        # Repeat the same fixed-seed tile for multiple environments.
        gen.num_rows = gen.num_cols = 1
        if seed is not None:
            gen.seed = seed
        self.play_terrain = name
