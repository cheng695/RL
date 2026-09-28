"""Soft geometry constraints: preserve climbing freedom without rewarding a split stance."""
import torch
from isaaclab.managers import SceneEntityCfg


def wheel_fore_aft_split(
    env, wheel_cfg: SceneEntityCfg, deadband=0.08, scale=0.15,
    linear_tail=False, max_value=4.0,
):
    """Penalize excessive fore/aft separation, independent of mirrored joint signs.

    Use yaw-only forward direction so wheel height differences on a stair do not
    masquerade as longitudinal separation. Defaults allow 8 cm of stepping.
    The gentle task uses a narrower band and an uncapped linear tail so a
    large split still costs more, without an explosive quadratic penalty.
    """
    robot = env.scene[wheel_cfg.name]
    wheels = robot.data.body_pos_w[:, wheel_cfg.body_ids]
    q = robot.data.root_quat_w
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    delta = wheels[:, 0] - wheels[:, 1]
    separation = (delta[:, 0] * yaw.cos() + delta[:, 1] * yaw.sin()).abs()
    error = (separation - deadband).clamp_min(0) / scale
    cost = torch.where(error < 1., error.square(), 2. * error - 1.) if linear_tail else error.square()
    return cost if max_value is None else cost.clamp_max(max_value)


def pitch_deadband(env, flat_tolerance=0.08727, obstacle_tolerance=0.2618, scale=0.15):
    """5 deg tolerance on level ground, up to 15 deg at obstacles/slopes.

    A larger pitch is possible but no longer free. This does not force a climbing
    robot to remain exactly horizontal.
    """
    from .step_course import terrain_scan
    scan = terrain_scan(env)
    variation = scan.amax(-1) - scan.amin(-1)
    tolerance = flat_tolerance + (obstacle_tolerance-flat_tolerance) * (variation / .10).clamp(0, 1)
    gravity = env.scene['robot'].data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=-1)).abs()
    return ((pitch-tolerance).clamp_min(0) / scale).square().clamp_max(4)


def gentle_orientation(env, kernel_coeff=20.0, flat_tolerance=0.08727, obstacle_tolerance=0.2618):
    """Keep roll upright with configurable terrain-dependent pitch tolerance."""
    gravity = env.scene['robot'].data.projected_gravity_b
    pitch_cost = pitch_deadband(env, flat_tolerance, obstacle_tolerance) * 0.15**2
    return torch.exp(-kernel_coeff * (gravity[:, 1].square() + pitch_cost))


def failure_event(env):
    """One unit per physical failure, independent of RewardManager dt scaling.

    TerminationManager.terminated excludes timeout-only resets; simultaneous
    physical failure and timeout remains a failure.
    """
    return env.termination_manager.terminated.float() / env.step_dt
