from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def _body_height_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if body_height.ndim > 1:
        body_height = torch.mean(body_height, dim=1)
    from .observations import local_ground_height
    return body_height - local_ground_height(env)


def _finite_or_penalty(value: torch.Tensor, penalty: float) -> torch.Tensor:
    """Keep reward terms finite on the same step that a bad state is terminated."""
    return torch.where(torch.isfinite(value), value, torch.full_like(value, penalty))


def vx_tracking_gaussian(
    env,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded forward velocity with a Gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(error) / sigma_sq), penalty=0.0)


def vx_tracking_huber(
    env,
    command_name: str,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Forward velocity Huber penalty; distinguishes large errors until the cap."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def yaw_tracking_gaussian(
    env,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded yaw rate with a Gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_ang_vel_b[:, 2] - command[:, 2]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(error) / sigma_sq), penalty=0.0)


def yaw_tracking_huber(
    env,
    command_name: str,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw error outside the useful range of the Gaussian reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_ang_vel_b[:, 2] - command[:, 2]
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def lateral_velocity_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sideways base motion for a no-sideways-command task."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.square(asset.data.root_lin_vel_b[:, 1])
    return _finite_or_penalty(value.clamp(max=max_value), penalty=max_value)


def orientation_exp_kernel(
    env,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward small roll/pitch error."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_error_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return _finite_or_penalty(torch.exp(-kernel_coeff * roll_pitch_error_sq), penalty=0.0)


def roll_pitch_deadband_l2(
    env,
    roll_deadband_rad: float,
    pitch_deadband_rad: float,
    error_scale: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll and pitch only after they exceed their allowed deadbands."""
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=-1))
    roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
    roll_excess = (roll.abs() - roll_deadband_rad).clamp_min(0.0)
    pitch_excess = (pitch.abs() - pitch_deadband_rad).clamp_min(0.0)
    value = (torch.square(roll_excess / error_scale) + torch.square(pitch_excess / error_scale))
    return _finite_or_penalty(value.clamp(max=max_value), penalty=max_value)


def wheel_fore_aft_split_l2(
    env,
    deadband: float,
    error_scale: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize fore-aft misalignment between the left and right wheel bodies."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if wheel_pos.shape[1] != 2:
        raise ValueError(f"Expected exactly two wheel bodies, got {wheel_pos.shape[1]}")

    # Project the wheel-to-wheel vector onto the robot's horizontal forward axis.
    delta = wheel_pos[:, 0, :2] - wheel_pos[:, 1, :2]
    quat = asset.data.root_quat_w
    yaw_sin = 2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2])
    yaw_cos = 1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square())
    fore_aft_offset = torch.abs(delta[:, 0] * yaw_cos + delta[:, 1] * yaw_sin)
    excess = (fore_aft_offset - deadband).clamp_min(0.0)
    value = torch.square(excess / error_scale)
    return _finite_or_penalty(value.clamp(max=max_value), penalty=max_value)


def straight_pitch_deadband_l2(
    env, command_name: str, height_command_name: str, settle_time: float,
    deadband_rad: float, error_scale: float, max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Discourage persistent straight-line lean, with grace after either command changes.

    Command age is a transition guard, not proof that physical acceleration ended.
    """
    velocity_term = env.command_manager.get_term(command_name)
    height_term = env.command_manager.get_term(height_command_name)
    command = velocity_term.command
    active = ((command[:, 0].abs() > 1e-6) & (command[:, 2].abs() <= 1e-6)
              & (velocity_term.command_age >= settle_time)
              & (height_term.command_age >= settle_time))
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    pitch = torch.atan2(gravity[:, 0], torch.linalg.vector_norm(gravity[:, 1:], dim=-1))
    excess = (pitch.abs() - deadband_rad).clamp_min(0.0)
    cost = _finite_or_penalty((excess / error_scale).square().clamp(max=max_value), penalty=max_value)
    return torch.where(active, cost, torch.zeros_like(cost))


def base_height_command_exp_kernel(
    env,
    command_name: str,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base height above the flat ground plane."""
    body_height = _body_height_w(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    error_sq = torch.square(body_height - command[:, 0])
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def base_height_command_l2(
    env,
    command_name: str,
    error_scale: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized commanded base-height tracking error on flat ground."""
    body_height = _body_height_w(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    scaled_error = (body_height - command[:, 0]) / error_scale
    value = torch.square(scaled_error)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def yaw_rate_l2_when_no_yaw_command(
    env,
    command_name: str,
    command_deadband: float,
    yaw_rate_deadband: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize yaw drift when the commanded yaw rate is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_yaw_command = torch.abs(command[:, 2]) <= command_deadband
    yaw_rate_excess = torch.clamp(torch.abs(asset.data.root_ang_vel_b[:, 2]) - yaw_rate_deadband, min=0.0)
    value = torch.where(no_yaw_command, torch.square(yaw_rate_excess), torch.zeros_like(yaw_rate_excess))
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def stand_still_lin_vel_l2(
    env,
    command_name: str,
    command_deadband: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal drift when the commanded forward speed is near zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_lin_command = torch.abs(command[:, 0]) <= command_deadband
    lin_vel_xy = asset.data.root_lin_vel_b[:, :2]
    value = torch.where(no_lin_command, torch.sum(torch.square(lin_vel_xy), dim=1), torch.zeros_like(command[:, 0]))
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def stand_still_drift_huber(
    env,
    command_name: str,
    command_deadband: float,
    velocity_scale: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Scale-aware horizontal drift cost only for zero translation AND yaw commands.

    Smooth near zero and linear for larger speeds, allowing braking and balance
    recovery without imposing a joint-motion or vertical-velocity penalty.
    """
    if velocity_scale <= 0.0:
        raise ValueError("velocity_scale must be positive")
    command = env.command_manager.get_command(command_name)
    standing = (command[:, :3].abs() <= command_deadband).all(dim=-1)
    speed = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2].abs() / velocity_scale
    cost = torch.where(speed < 1.0, 0.5 * speed.square(), speed - 0.5).sum(dim=-1)
    cost = _finite_or_penalty(cost.clamp(max=max_value), penalty=max_value)
    return torch.where(standing, cost, torch.zeros_like(cost))


def joint_torques_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected motor torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def wheel_power_l1_positive(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize positive wheel mechanical power, matching the old wheel-leg reward style."""
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = torque * joint_vel
    # Invalid power must not disappear as zero consumption (including -Inf).
    power = _finite_or_penalty(power, penalty=max_value)
    value = torch.sum(torch.clamp(power, min=0.0), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def leg_joint_vel_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize squared velocities of the leg joints only (anti-squat term).

    Targets unnecessary periodic leg extension/retraction directly, without
    touching the wheel joints that need to roll continuously for balancing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def action_rate_l2(
    env,
    max_value: float,
    action_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalize action rate, optionally on a slice of the action vector."""
    action_delta = env.action_manager.action - env.action_manager.prev_action
    if action_slice is not None:
        start, stop = action_slice
        action_delta = action_delta[:, start:stop]
    value = torch.sum(torch.square(action_delta), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def wheel_target_delta_l2(
    env,
    action_name: str,
    delta_scale: float,
    max_value: float,
) -> torch.Tensor:
    """Mean squared final wheel-target change; history is owned/reset by the action term.

    delta_scale is in rad/s per control step, not rad/s².
    Reading this term does not advance history, so repeated reward reads are safe.
    """
    if delta_scale <= 0.0 or max_value <= 0.0:
        raise ValueError("delta_scale and max_value must be positive.")
    delta = env.action_manager.get_term(action_name).target_delta
    value = torch.square(delta / delta_scale).mean(dim=-1)
    return _finite_or_penalty(value.clamp(max=max_value), penalty=max_value)


def action_second_order_l2(
    env,
    max_value: float,
    action_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Penalize second-order action differences."""
    action = env.action_manager.action
    if not hasattr(env, "_my_robot_prev_prev_action"):
        env._my_robot_prev_prev_action = env.action_manager.prev_action.clone()

    step = int(getattr(env, "common_step_counter", -1))
    if (
        not hasattr(env, "_my_robot_action_second_order_cache")
        or getattr(env, "_my_robot_action_second_order_cache_step", None) != step
    ):
        prev_action = env.action_manager.prev_action
        prev_prev_action = env._my_robot_prev_prev_action
        # ManagerBasedRLEnv increments episode lengths before reward evaluation.
        # On the first step after reset, action_manager has zeroed its history;
        # clear our extra history only for those environments as well.
        prev_prev_action[env.episode_length_buf <= 1] = 0.0
        env._my_robot_action_second_order_cache = action - 2.0 * prev_action + prev_prev_action
        env._my_robot_action_second_order_cache_step = step
        env._my_robot_prev_prev_action = prev_action.clone()

    second_order = env._my_robot_action_second_order_cache
    if action_slice is not None:
        start, stop = action_slice
        second_order = second_order[:, start:stop]
    value = torch.sum(torch.square(second_order), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def contact_sensor_contact(
    env,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Return 1.0 when any selected contact sensor body exceeds the force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    has_contact = has_contact | torch.any(~torch.isfinite(contact_force.flatten(start_dim=1)), dim=1)
    return has_contact.float()
