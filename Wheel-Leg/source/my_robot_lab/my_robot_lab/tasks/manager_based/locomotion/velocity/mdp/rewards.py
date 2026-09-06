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
    return body_height


def base_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize squared error between the selected body height and the target height."""

    body_height = _body_height_w(env, asset_cfg)
    return torch.square(body_height - target_height)


def _sampled_base_height_target(env, target_height_range: tuple[float, float]) -> torch.Tensor:
    lower, upper = target_height_range
    device = env.device
    num_envs = env.num_envs
    target = getattr(env, "_my_robot_base_height_target", None)
    if target is None or target.shape[0] != num_envs:
        target = torch.empty(num_envs, device=device)
        setattr(env, "_my_robot_base_height_target", target)
        reset_ids = torch.arange(num_envs, device=device)
    else:
        reset_ids = torch.nonzero(env.episode_length_buf == 0, as_tuple=False).squeeze(-1)

    if reset_ids.numel() > 0:
        target[reset_ids] = torch.empty(reset_ids.numel(), device=device).uniform_(lower, upper)
    return target


def track_base_height_exp(
    env,
    std: float,
    target_height: float | None = None,
    target_height_range: tuple[float, float] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward selected body height tracking with an exponential kernel."""

    body_height = _body_height_w(env, asset_cfg)
    if target_height_range is not None:
        target = _sampled_base_height_target(env, target_height_range)
    elif target_height is not None:
        target = target_height
    else:
        raise ValueError("Either target_height or target_height_range must be provided.")
    height_error = torch.square(body_height - target)
    return torch.exp(-height_error / std**2)


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
    """Forward velocity Huber penalty; keeps useful gradient when the error is large."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.nan_to_num(asset.data.root_lin_vel_b[:, 0] - command[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def wheel_vx_tracking_gaussian(
    env,
    command_name: str,
    wheel_radius: float,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Track commanded vx using the common wheel rolling speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.joint_ids) != 2:
        raise ValueError("wheel_vx_tracking_gaussian expects two wheel joints ordered [left, right].")
    command = env.command_manager.get_command(command_name)
    left_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    right_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[1]]
    wheel_forward_vel = 0.5 * (left_wheel_vel - right_wheel_vel)
    vx_from_wheels = float(wheel_radius) * wheel_forward_vel
    error = vx_from_wheels - command[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(error) / sigma_sq), penalty=0.0)


def wheel_vx_tracking_huber(
    env,
    command_name: str,
    wheel_radius: float,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Huber penalty between commanded vx and wheel-implied vx."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.joint_ids) != 2:
        raise ValueError("wheel_vx_tracking_huber expects two wheel joints ordered [left, right].")
    command = env.command_manager.get_command(command_name)
    left_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    right_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[1]]
    wheel_forward_vel = 0.5 * (left_wheel_vel - right_wheel_vel)
    vx_from_wheels = float(wheel_radius) * wheel_forward_vel
    error = torch.nan_to_num(vx_from_wheels - command[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def base_wheel_vx_consistency_huber(
    env,
    wheel_radius: float,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize mismatch between base vx and wheel rolling velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.joint_ids) != 2:
        raise ValueError("base_wheel_vx_consistency_huber expects two wheel joints ordered [left, right].")
    left_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    right_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[1]]
    wheel_forward_vel = 0.5 * (left_wheel_vel - right_wheel_vel)
    vx_from_wheels = float(wheel_radius) * wheel_forward_vel
    error = torch.nan_to_num(asset.data.root_lin_vel_b[:, 0] - vx_from_wheels, nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def orientation_exp_kernel(
    env,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward small roll/pitch error."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_error_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return _finite_or_penalty(torch.exp(-kernel_coeff * roll_pitch_error_sq), penalty=0.0)


def pitch_tracking_gaussian(
    env,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Auxiliary pitch balance reward using projected gravity as a pitch proxy."""
    asset: Articulation = env.scene[asset_cfg.name]
    pitch_proxy = asset.data.projected_gravity_b[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(pitch_proxy) / sigma_sq), penalty=0.0)


def pitch_rate_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitch angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.square(asset.data.root_ang_vel_b[:, 1])
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def base_height_range_exp_kernel(
    env,
    target_height_range: tuple[float, float],
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track an episode-sampled base-height target with exp(-k * error^2)."""
    body_height = _body_height_w(env, asset_cfg)
    target = _sampled_base_height_target(env, target_height_range)
    error_sq = torch.square(body_height - target)
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def base_height_range_tracking_fudan(
    env,
    target_height_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fudan-style tight base-height tracking: exp(-error^2 / 0.001)."""
    body_height = _body_height_w(env, asset_cfg)
    target = _sampled_base_height_target(env, target_height_range)
    error = torch.square(body_height - target)
    return _finite_or_penalty(torch.exp(-error / 0.001), penalty=0.0)


def base_height_command_exp_kernel(
    env,
    command_name: str,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track commanded base height with exp(-k * error^2)."""
    body_height = _body_height_w(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    error_sq = torch.square(body_height - command[:, 0])
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def base_height_command_tracking_fudan(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fudan-style tight commanded height tracking: exp(-error^2 / 0.001)."""
    body_height = _body_height_w(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    error = torch.square(body_height - command[:, 0])
    return _finite_or_penalty(torch.exp(-error / 0.001), penalty=0.0)


def base_height_command_l2(
    env,
    command_name: str,
    error_scale: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized commanded base-height tracking error."""
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


def joint_torques_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize selected motor torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
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


def leg_joint_vel_l2_height_gated(
    env,
    command_name: str,
    gate_sigma: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Leg-joint velocity penalty gated by the base-height tracking error.

    .. math::
        \\text{gate} = \\exp\\left(-\\left(\\frac{h_\\text{base} - h_\\text{cmd}}{\\sigma}\\right)^2\\right)

    Full penalty when the robot settles near the commanded height, relaxed
    while the height error is large (command transitions, push recovery), so
    active raising/lowering of the base is still possible.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = _body_height_w(env, asset_cfg)
    command = env.command_manager.get_command(command_name)
    height_error = body_height - command[:, 0]
    gate = torch.exp(-torch.square(height_error / gate_sigma))
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    value = gate * joint_vel_sq
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def leg_joint_acc_l2(
    env,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize squared accelerations of the leg joints only (escalation lever)."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)
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
    action_delta = torch.nan_to_num(action_delta, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.sum(torch.square(action_delta), dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


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
        env._my_robot_action_second_order_cache = action - 2.0 * prev_action + prev_prev_action
        env._my_robot_action_second_order_cache_step = step
        env._my_robot_prev_prev_action = prev_action.clone()

    second_order = env._my_robot_action_second_order_cache
    if action_slice is not None:
        start, stop = action_slice
        second_order = second_order[:, start:stop]
    second_order = torch.nan_to_num(second_order, nan=0.0, posinf=0.0, neginf=0.0)
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
