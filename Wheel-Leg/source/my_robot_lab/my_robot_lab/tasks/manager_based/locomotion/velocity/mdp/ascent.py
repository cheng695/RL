"""Forward-only ascent and permission for chassis support on ascent terrain."""
import torch
from .commands import PositiveBiasedVelocityCommand, UniformBaseHeightCommand
from .terminations import illegal_contact_after_steps
from .rewards import contact_sensor_contact


def ascent_mask(env):
    """Match the curriculum generator's column assignment, without hardcoded columns."""
    terrain = env.scene.terrain
    cfg = terrain.cfg.terrain_generator
    total = sum(t.proportion for t in cfg.sub_terrains.values())
    coordinate = terrain.terrain_types.float() / cfg.num_cols + .001
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    lower = 0.
    for name, sub in cfg.sub_terrains.items():
        upper = lower + sub.proportion / total
        if name == "pyramid_stairs_inv":
            result |= (coordinate >= lower) & (coordinate < upper)
        lower = upper
    return result


class ForwardAscentCommand(PositiveBiasedVelocityCommand):
    """Other terrain retains bidirectional commands; ascending stairs use slow forward motion."""
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[ascent_mask(self._env)[ids]]
        self.vel_command_b[ids] = 0.
        self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(.15, .30)
        self.is_standing_env[ids] = False
        self._project_to_wheel_speed_limit(ids)


def unsupported_base_contact(env, threshold, min_steps, sensor_cfg):
    """Allow support during an attempt; debounce other contact on ascent terrain."""
    term = env.command_manager.get_term("base_height")
    term.check_state()
    allowed = (term.phase > 0) | (term.cooldown > 0)
    raw = illegal_contact_after_steps(env, threshold, min_steps, sensor_cfg) & ~allowed
    if term._contact_check != env.common_step_counter:
        term.contact_time[:] = torch.where(raw, term.contact_time + env.step_dt, 0.)
        term._contact_check = env.common_step_counter
    return raw & (~ascent_mask(env) | (term.contact_time >= .2 - 1e-6))


def unsupported_contact_cost(env, threshold, sensor_cfg):
    # Do not simultaneously punish the support contact that is now permitted.
    term = env.command_manager.get_term("base_height")
    return contact_sensor_contact(env, threshold=threshold, sensor_cfg=sensor_cfg) * (term.phase == 0)


class AscentHeightCommand(UniformBaseHeightCommand):
    """0 cruise, 1 extend, 2 supported/retract. Uses existing height observation.

    All targets remain local-ground-relative. No joint actions are scripted.
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.settle = torch.zeros_like(self.elapsed)
        self.top = torch.zeros_like(self.elapsed)
        self.cooldown = torch.zeros_like(self.elapsed)
        self.success = torch.zeros_like(self.elapsed)
        self.course_success = torch.zeros_like(self.elapsed)
        self.completed_steps = torch.zeros_like(self.phase)
        self.highest_completed_top = torch.full_like(self.elapsed, -torch.inf)
        self.failed = torch.zeros_like(self.elapsed)
        self.nominal = torch.zeros_like(self.height_command)
        self.entry_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.direction = torch.zeros_like(self.entry_xy)
        self.crossing = torch.zeros_like(self.elapsed)
        self.best_progress = torch.zeros_like(self.elapsed)
        self.progress = torch.zeros_like(self.elapsed)
        self.height_tolerance = torch.zeros_like(self.elapsed)
        for name in ("near_time", "stall_time", "support_time", "support_event", "entry_ground",
                     "entry_chassis_z", "best_wheel_lift", "best_body_lift", "wheel_lift", "body_lift",
                     "contact_time", "tilt_time", "success_streak", "failure_streak", "attempts"):
            setattr(self, name, torch.zeros_like(self.elapsed))
        self._state_check = self._contact_check = self._tilt_check = -1
        robot = env.scene[cfg.asset_name]
        self.wheels, _ = robot.find_bodies(["left_right_wheel", "right_right_wheel"], preserve_order=True)
        sensor = env.scene["contact_forces"]
        self.wheel_contacts, _ = sensor.find_bodies(["left_right_wheel", "right_right_wheel"], preserve_order=True)
        self.chassis_contacts, _ = sensor.find_bodies("chassis")

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.nominal[env_ids] = self.height_command[env_ids]

    def apply_assist(self):
        """Also called after manual Play height overrides, before observation refresh."""
        self.height_command[:, 0] = torch.where(self.phase == 1, self.cfg.height_range[1],
            torch.where(self.phase == 2, self.cfg.height_range[0], self.height_command[:, 0]))

    def _update_command(self):
        super()._update_command()
        self.check_state()
        # Command resampling/manual overrides may have happened after reward evaluation.
        self.apply_assist()

    def check_state(self):
        """Evaluate once on current physics, before both terminations and rewards."""
        if self._state_check == self._env.common_step_counter:
            return
        self._state_check = self._env.common_step_counter
        self._advance_state()

    def _advance_state(self):
        env = self._env
        dt = env.step_dt
        self.success.zero_()
        self.course_success.zero_()
        self.failed.zero_()
        self.progress.zero_()
        self.wheel_lift.zero_()
        self.body_lift.zero_()
        self.support_event.zero_()
        self.cooldown.sub_(dt).clamp_min_(0)
        robot = env.scene[self.cfg.asset_name]
        scan = env.scene['terrain_scan'].data
        # Select forward points by world geometry, independent of ray array ordering.
        delta = scan.ray_hits_w - robot.data.root_pos_w[:, None, :]
        q = robot.data.root_quat_w
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        forward = delta[..., 0]*yaw.cos()[:, None] + delta[..., 1]*yaw.sin()[:, None]
        ground = self.reference_ground_height()
        rise = scan.ray_hits_w[..., 2] - ground[:, None]
        candidates = (forward > .15) & (forward < .65) & (rise > .02) & (rise <= .22) & torch.isfinite(rise)
        # Never pay for re-climbing a completed tread after rolling back.
        candidates &= scan.ray_hits_w[..., 2] > self.highest_completed_top[:, None] + .01
        # Nearest raised level, not the highest of several visible stair treads.
        top = torch.where(candidates, scan.ray_hits_w[..., 2], torch.inf).amin(-1)
        vx = env.command_manager.get_command('base_velocity')[:, 0]
        enter = (self.phase == 0) & (self.cooldown == 0) & ascent_mask(env) & (vx > .05) & torch.isfinite(top) & (self.completed_steps < 2)
        enter &= env.episode_length_buf > 0
        self.top[enter] = top[enter]
        self.entry_xy[enter] = robot.data.root_pos_w[enter, :2]
        self.direction[enter] = torch.stack((yaw.cos(), yaw.sin()), -1)[enter]
        # A scan point on the first tread is a conservative crossing plane.
        first_tread = candidates & ((scan.ray_hits_w[..., 2] - top[:, None]).abs() < .005)
        crossing = torch.where(first_tread, forward, torch.inf).amin(-1)
        self.crossing[enter] = crossing[enter]
        self.height_tolerance[enter] = ((top-ground)*.25).clamp(.005, .025)[enter]
        self.best_progress[enter] = 0
        self.entry_ground[enter] = ground[enter]
        self.entry_chassis_z[enter] = robot.data.body_pos_w[enter, self._height_body_ids[0], 2]
        for value in (self.near_time, self.stall_time, self.support_time, self.best_wheel_lift, self.best_body_lift):
            value[enter] = 0
        self.attempts[enter] += 1
        self.settle[enter] = 0
        self.phase[enter] = 1
        self.elapsed[enter] = 0
        active = self.phase > 0
        self.elapsed += active * dt
        forces = env.scene['contact_forces'].data.net_forces_w
        # Upward contact and chassis close to tread height approximate support;
        # a horizontal wall impact alone does not request retraction.
        support_force = forces[:, self.chassis_contacts, 2].sum(-1) > 5.
        chassis_z = robot.data.body_pos_w[:, self._height_body_ids[0], 2]
        supported = support_force & (chassis_z > self.top) & (chassis_z < self.top + .18)
        self.support_time[:] = torch.where(active & supported, self.support_time + dt, 0.)
        retract = (self.phase == 1) & (self.support_time >= .08 - 1e-6)
        self.support_event[retract] = 1.
        self.phase[retract] = 2
        bottoms = robot.data.body_pos_w[:, self.wheels, 2] - .055
        wheels_on_top = ((bottoms-self.top[:, None]).abs() < self.height_tolerance[:, None]).all(-1)
        wheel_delta = robot.data.body_pos_w[:, self.wheels, :2] - self.entry_xy[:, None, :]
        wheel_forward = (wheel_delta*self.direction[:, None, :]).sum(-1)
        crossed = (wheel_forward >= self.crossing[:, None]).all(-1)
        # Progress requires the lagging wheel to advance, not just pitching the chassis.
        distance = wheel_forward.amin(-1)
        distance = distance.clamp_min(0).minimum(self.crossing + .4)
        best = torch.maximum(self.best_progress, distance)
        self.progress[:] = torch.where(active, best-self.best_progress, 0.)
        self.best_progress[:] = torch.where(active, best, self.best_progress)
        near = active & ((self.crossing - wheel_forward.amin(-1)) < .25)
        self.near_time += near * dt
        riser = (self.top - self.entry_ground).clamp_min(.025)
        wheel_potential = ((bottoms-self.entry_ground[:, None]) / riser[:, None]).clamp(0, 1).amin(-1)
        body_potential = ((chassis_z-self.entry_chassis_z) / riser).clamp(0, 1)
        viable = near & (-robot.data.projected_gravity_b[:, 2] > .5)
        new_wheel = torch.maximum(self.best_wheel_lift, wheel_potential)
        new_body = torch.maximum(self.best_body_lift, body_potential)
        self.wheel_lift[:] = torch.where(viable, new_wheel-self.best_wheel_lift, 0.)
        self.body_lift[:] = torch.where(viable, new_body-self.best_body_lift, 0.)
        self.best_wheel_lift += self.wheel_lift
        self.best_body_lift += self.body_lift
        improving = (self.progress > .001) | (self.wheel_lift > .005) | (self.body_lift > .005)
        self.stall_time[:] = torch.where(near & ~improving, self.stall_time + dt, 0.)
        contact = (forces[:, self.wheel_contacts, 2] > 1.).all(-1)
        upright = -robot.data.projected_gravity_b[:, 2] > .90
        settled = active & crossed & wheels_on_top & contact & upright & (chassis_z > self.top + .22)
        self.settle = torch.where(settled, self.settle + dt, 0.)
        done = active & (self.settle >= .2 - 1e-6)
        # Approach time no longer consumes a short contact/retraction window.
        # A small recovery reversal is permitted; stalled attempts still end.
        failure = active & ((self.near_time >= 8.) | (self.elapsed >= 12.) |
                            ((self.near_time >= 2.) & (self.stall_time >= 3.))) & ~done
        self.success[done] = 1.
        self.completed_steps[done] += 1
        self.highest_completed_top[done] = self.top[done]
        self.course_success[done & (self.completed_steps == 2)] = 1.
        self.failed[failure] = 1.
        exit_mask = done | failure
        self.phase[exit_mask] = 0
        self.cooldown[exit_mask] = .5
        self.height_command[exit_mask] = self.nominal[exit_mask]
        self.apply_assist()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        mask = ascent_mask(self._env)[ids] & (self._env.episode_length_buf[ids] > 0)
        count = mask.sum().clamp_min(1)
        logs = {"ascent_episode_count": mask.sum().item()}
        for name, values in (("first_step_pass_rate", self.completed_steps >= 1),
                             ("two_step_pass_rate", self.completed_steps >= 2),
                             ("ascent_attempt_rate", self.attempts > 0),
                             ("ascent_stall_timeout_rate", self.failed > 0)):
            logs[name] = ((values[ids] & mask).sum() / count).item()
        extras = super().reset(env_ids)
        extras.update(logs)
        for value in (self.phase, self.elapsed, self.settle, self.top, self.cooldown, self.success, self.failed,
                      self.entry_xy, self.direction, self.crossing, self.best_progress, self.progress, self.height_tolerance,
                      self.course_success, self.completed_steps, self.near_time, self.stall_time,
                      self.support_time, self.support_event, self.entry_ground, self.entry_chassis_z,
                      self.best_wheel_lift, self.best_body_lift, self.wheel_lift, self.body_lift,
                      self.contact_time, self.tilt_time, self.attempts):
            value[ids] = 0
        self.highest_completed_top[ids] = -torch.inf
        return extras


def ascent_success(env):
    # RewardManager multiplies by dt: keep this a fixed bonus per crossing.
    return env.command_manager.get_term('base_height').success / env.step_dt


def ascent_failure(env):
    term = env.command_manager.get_term('base_height')
    term.check_state()
    return term.failed > 0


def ascent_course_success(env):
    """One additional bonus when both distinct ascending treads are completed."""
    return env.command_manager.get_term('base_height').course_success / env.step_dt


def ascent_completed(env):
    """Successful training episodes finish as truncations, not fall terminations."""
    term = env.command_manager.get_term('base_height')
    term.check_state()
    return term.course_success > 0


def ascent_progress(env):
    """Reward only a new farthest position in this attempt, never leg oscillation."""
    return env.command_manager.get_term('base_height').progress / env.step_dt


def ascent_wheel_lift(env):
    return env.command_manager.get_term('base_height').wheel_lift / env.step_dt


def ascent_body_lift(env):
    return env.command_manager.get_term('base_height').body_lift / env.step_dt


def ascent_support(env):
    return env.command_manager.get_term('base_height').support_event / env.step_dt


def ascent_time_cost(env):
    term = env.command_manager.get_term('base_height')
    return (((term.phase > 0) & (term.near_time > 0)) | (term.failed > 0)).float()


def ascent_failure_cost(env):
    return env.command_manager.get_term('base_height').failed / env.step_dt


def ascent_height_exp(env, command_name, kernel_coeff, asset_cfg, relative_to_ground=True):
    from .step_course import adaptive_height_exp
    active = env.command_manager.get_term('base_height').phase > 0
    return adaptive_height_exp(env, command_name, kernel_coeff, asset_cfg, relative_to_ground) * (~active).float()


def ascent_height_l2(env, command_name, error_scale, max_value, asset_cfg, relative_to_ground=True):
    from .step_course import adaptive_height_l2
    active = env.command_manager.get_term('base_height').phase > 0
    return adaptive_height_l2(env, command_name, error_scale, max_value, asset_cfg, relative_to_ground) * torch.where(active, .1, 1.)


def ascent_bad_roll_pitch(env, limit_angle, asset_cfg):
    """Permit transient climbing pitch, while retaining roll and severe-tip limits."""
    term = env.command_manager.get_term("base_height")
    term.check_state()
    g = env.scene[asset_cfg.name].data.projected_gravity_b
    roll = torch.atan2(g[:, 1], -g[:, 2]).abs()
    pitch = torch.asin(g[:, 0].clamp(-1, 1)).abs()
    climbing = term.phase > 0
    ordinary = torch.acos((-g[:, 2]).clamp(-1, 1)) > limit_angle
    raw = torch.where(climbing, (roll > limit_angle) | (pitch > 1.0472), ordinary)
    if term._tilt_check != env.common_step_counter:
        term.tilt_time[:] = torch.where(raw, term.tilt_time + env.step_dt, 0.)
        term._tilt_check = env.common_step_counter
    immediate = (roll > 1.3) | (pitch > 1.3) | ~torch.isfinite(g).all(-1)
    return immediate | (raw & (~climbing | (term.tilt_time >= .2 - 1e-6)))


def ascent_orientation(env, kernel_coeff):
    from .step_course import adaptive_orientation
    g = env.scene["robot"].data.projected_gravity_b
    roll = torch.atan2(g[:, 1], -g[:, 2])
    pitch = torch.asin(g[:, 0].clamp(-1, 1)).abs()
    # A modest pitch is useful for support; roll remains constrained.
    climb_reward = .1 * torch.exp(-10*roll.square() - 3*(pitch-.5236).clamp_min(0).square())
    return torch.where(env.command_manager.get_term("base_height").phase > 0,
                       climb_reward, adaptive_orientation(env, kernel_coeff))


def ascent_yaw_tracking(env, command_name, sigma):
    from .rewards import yaw_tracking_gaussian
    active = env.command_manager.get_term("base_height").phase > 0
    return yaw_tracking_gaussian(env, command_name, sigma) * torch.where(active, .1, 1.)


def ascent_vx_tracking(env, command_name, sigma):
    from .rewards import vx_tracking_gaussian
    active = env.command_manager.get_term("base_height").phase > 0
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    target = env.command_manager.get_command(command_name)[:, 0]
    moving = (vx / target.clamp_min(.05)).clamp(0, 1)
    return vx_tracking_gaussian(env, command_name, sigma) * torch.where(active, moving, 1.)


def ascent_pitch_cost(env):
    from .rough_posture import pitch_deadband
    g = env.scene["robot"].data.projected_gravity_b
    pitch = torch.asin(g[:, 0].clamp(-1, 1)).abs()
    active = env.command_manager.get_term("base_height").phase > 0
    cost = ((pitch-.5236).clamp_min(0) / .15).square().clamp_max(4.)
    return torch.where(active, cost, pitch_deadband(env))


def ascent_vertical_velocity_cost(env):
    active = env.command_manager.get_term("base_height").phase > 0
    return env.scene["robot"].data.root_lin_vel_b[:, 2].square() * torch.where(active, .1, 1.)


def reset_ascent_root(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    """Reset upright on the lower platform, progressively further from the riser."""
    from isaaclab.envs.mdp import reset_root_state_uniform
    from isaaclab.managers import SceneEntityCfg
    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    ascending = ascent_mask(env)[ids]
    ordinary = ids[~ascending]
    if len(ordinary):
        reset_root_state_uniform(env, ordinary, pose_range, velocity_range, asset_cfg)
    # Group by level to keep the standard, tested root reset implementation.
    terrain = env.scene.terrain
    rows = terrain.cfg.terrain_generator.num_rows
    for level in range(rows):
        group = ids[ascending & (terrain.terrain_levels[ids] == level)]
        if not len(group):
            continue
        center_x = .90 * (1. - level / max(rows-1, 1))
        pose = dict(pose_range)
        pose.update(x=(center_x-.05, center_x+.05), y=(-.05, .05), yaw=(-.05, .05))
        reset_root_state_uniform(env, group, pose, velocity_range, asset_cfg)


def update_ascent_curriculum(env, env_ids):
    """Three full successes move the start back; two failures move it closer."""
    term = env.command_manager.get_term("base_height")
    terrain = env.scene.terrain
    ids = env_ids[ascent_mask(env)[env_ids]]
    won = (term.completed_steps[ids] == 2) & ~env.termination_manager.terminated[ids]
    term.success_streak[ids] = torch.where(won, term.success_streak[ids]+1, 0.)
    term.failure_streak[ids] = torch.where(won, 0., term.failure_streak[ids]+1)
    up, down = term.success_streak[ids] >= 3, term.failure_streak[ids] >= 2
    terrain.terrain_levels[ids] = (terrain.terrain_levels[ids] + up.long() - down.long()).clamp(
        0, terrain.cfg.terrain_generator.num_rows-1)
    term.success_streak[ids[up]] = 0
    term.failure_streak[ids[down]] = 0
    terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
