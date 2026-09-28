"""Single-step curriculum with explicit crossing and settled two-wheel support."""
import torch
from isaaclab.utils import configclass
from isaaclab.terrains.trimesh.mesh_terrains import box_terrain, pit_terrain
from .commands import PositiveBiasedVelocityCommand, PositiveBiasedVelocityCommandCfg

HEIGHTS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)


def course_step(difficulty, cfg):
    # Generator difficulty is stratified into eight rows; guarantee exact endpoints.
    height = HEIGHTS[min(int(difficulty * len(HEIGHTS)), len(HEIGHTS) - 1)]
    if hasattr(cfg, "pit_depth_range"):
        return pit_terrain(difficulty, cfg.replace(pit_depth_range=(height, height)))
    return box_terrain(difficulty, cfg.replace(box_height_range=(height, height)))


def terrain_scan(env, sensor_name="terrain_scan", height_command="base_height"):
    """45 yaw-aligned points: x outer (-0.8..0.8), y inner (-0.4..0.4), meters.

    Heights relative to local ground; positive = raised terrain. Invalid rays = 0.
    """
    hits = env.scene[sensor_name].data.ray_hits_w[..., 2]
    ground = env.command_manager.get_term(height_command).reference_ground_height()
    return torch.where(torch.isfinite(hits), (hits - ground[:, None]).clamp(-0.5, 0.5), 0.0)


def posture_precision(env):
    """Smoothly widen posture tolerance when the scan straddles a step edge."""
    scan = terrain_scan(env)
    variation = scan.amax(-1) - scan.amin(-1)
    return 1.0 - 0.75 * (variation / .025).clamp(0., 1.)


def adaptive_height_exp(env, command_name, kernel_coeff, asset_cfg, relative_to_ground=True):
    from .rewards import base_height_command_exp_kernel
    return base_height_command_exp_kernel(env, command_name, kernel_coeff * posture_precision(env),
                                          asset_cfg, relative_to_ground)


def adaptive_height_l2(env, command_name, error_scale, max_value, asset_cfg, relative_to_ground=True):
    from .rewards import base_height_command_l2
    return posture_precision(env) * base_height_command_l2(
        env, command_name, error_scale, max_value, asset_cfg, relative_to_ground)


def adaptive_orientation(env, kernel_coeff):
    from .rewards import orientation_exp_kernel
    return orientation_exp_kernel(env, kernel_coeff * posture_precision(env))


class StepCourseCommand(PositiveBiasedVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.wheel_ids, _ = self.robot.find_bodies(
            ["left_right_wheel", "right_right_wheel"], preserve_order=True)
        self.contact_ids, _ = env.scene["contact_forces"].find_bodies(
            ["left_right_wheel", "right_right_wheel"], preserve_order=True)
        self.hold = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.success_streak = torch.zeros_like(self.hold)
        self.failure_streak = torch.zeros_like(self.hold)
        self.travel_sign = torch.ones_like(self.hold)
        self._last_check = -1

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        obstacle = self._env.scene.terrain.terrain_types[ids] >= 6
        ids = ids[obstacle]
        sign = torch.where(torch.rand(len(ids), device=self.device) < .5, -1., 1.)
        self.vel_command_b[ids] = 0
        self.vel_command_b[ids, 0] = sign * torch.empty(len(ids), device=self.device).uniform_(.15, .5)
        self.is_standing_env[ids] = False
        self.travel_sign[ids] = sign

    def check_success(self):
        if self._last_check == self._env.common_step_counter:
            return self.success
        self._last_check = self._env.common_step_counter
        env = self._env
        terrain = env.scene.terrain
        kind = terrain.terrain_types
        wheels = self.robot.data.body_pos_w[:, self.wheel_ids]
        relative = wheels - env.scene.env_origins[:, None, :]
        heights = torch.tensor(HEIGHTS, device=self.device)[terrain.terrain_levels]
        target_z = env.scene.env_origins[:, 2] + torch.where(kind < 13, heights, -heights) + .055
        crossed = ((relative[:, :, 0] * self.travel_sign[:, None] > 1.15)
                   & (relative[:, :, 1].abs() < .9)).all(-1)
        on_platform = ((wheels[:, :, 2] - target_z[:, None]).abs() < .04).all(-1)
        forces = env.scene["contact_forces"].data.net_forces_w[:, self.contact_ids]
        supported = (torch.linalg.vector_norm(forces, dim=-1) > 1.).all(-1)
        stable = ((-self.robot.data.projected_gravity_b[:, 2] > .94)
                  & (self.robot.data.root_lin_vel_b[:, 2].abs() < .15))
        good = (kind >= 6) & crossed & on_platform & supported & stable
        self.hold = torch.where(good, self.hold + env.step_dt, 0.)
        self.success |= self.hold >= .5
        return self.success

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        kinds = self._env.scene.terrain.terrain_types[ids]
        completed = self._env.episode_length_buf[ids] > 0
        logs = {}
        for name, mask in (("up", (kinds >= 6) & (kinds < 13)), ("down", kinds >= 13)):
            mask = mask & completed
            logs[f"step_{name}_episode_count"] = mask.sum().item()
            logs[f"step_{name}_success_rate"] = (
                (self.success[ids] & mask).sum() / mask.sum().clamp_min(1)).item()
            failed = self._env.termination_manager.terminated[ids] & ~self.success[ids]
            logs[f"step_{name}_failure_rate"] = ((failed & mask).sum() / mask.sum().clamp_min(1)).item()
            logs[f"step_{name}_timeout_rate"] = (
                (self._env.termination_manager.time_outs[ids] & ~self.success[ids] & mask).sum() / mask.sum().clamp_min(1)).item()
        result = super().reset(env_ids)
        result.update(logs)
        self.hold[ids] = 0
        self.success[ids] = False
        return result


@configclass
class StepCourseCommandCfg(PositiveBiasedVelocityCommandCfg):
    class_type: type = StepCourseCommand


def crossed_and_settled(env):
    term = env.command_manager.get_term("base_velocity")
    term.check_success()
    for name in ("base_contact", "bad_roll_pitch", "leg_tendon_length"):
        term.success &= ~env.termination_manager.get_term(name)
    return term.success


def crossing_bonus(env):
    return crossed_and_settled(env).float()


def step_levels(env, env_ids):
    term = env.command_manager.get_term("base_velocity")
    terrain = env.scene.terrain
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ids = ids[(terrain.terrain_types[ids] >= 6) & (env.episode_length_buf[ids] > 0)]
    won = term.success[ids] & ~env.termination_manager.get_term("base_contact")[ids] \
        & ~env.termination_manager.get_term("bad_roll_pitch")[ids] \
        & ~env.termination_manager.get_term("leg_tendon_length")[ids]
    term.success_streak[ids] = torch.where(won, term.success_streak[ids] + 1, 0.)
    term.failure_streak[ids] = torch.where(won, 0., term.failure_streak[ids] + 1)
    up = term.success_streak[ids] >= 3
    down = term.failure_streak[ids] >= 2
    terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + up.long() - down.long()).clamp(0, len(HEIGHTS) - 1)
    term.success_streak[ids[up]] = 0
    term.failure_streak[ids[down]] = 0
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return terrain.terrain_levels.float().mean()
