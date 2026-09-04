from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from .observations import usd_leg_kinematics


def _whole_robot_com_w(asset: Articulation) -> torch.Tensor:
    """Return mass-weighted whole-robot COM in world frame."""
    masses = asset.data.default_mass
    body_com_pos_w = asset.data.body_com_pos_w
    if masses is None:
        raise RuntimeError("asset.data.default_mass is not available; cannot compute whole-robot COM.")
    masses = masses.to(device=body_com_pos_w.device, dtype=body_com_pos_w.dtype)
    weights = masses.unsqueeze(-1)
    return torch.sum(body_com_pos_w * weights, dim=1) / torch.sum(weights, dim=1)


def _com_minus_wheel_x_b(asset: Articulation, wheel_body_ids: Sequence[int]) -> torch.Tensor:
    """Return COM x offset from wheel midpoint in base frame."""
    if len(wheel_body_ids) == 2:
        left_wheel_id, right_wheel_id = wheel_body_ids
    elif len(wheel_body_ids) == 3:
        _, left_wheel_id, right_wheel_id = wheel_body_ids
    else:
        raise ValueError(
            "COM-wheel alignment expects wheel body ids ordered as "
            "[Left_Wheel_link, Right_Wheel_link] or [base_link, Left_Wheel_link, Right_Wheel_link]. "
            f"Resolved body ids: {wheel_body_ids}"
        )
    com_pos_w = _whole_robot_com_w(asset)
    wheel_center_w = 0.5 * (asset.data.body_pos_w[:, left_wheel_id] + asset.data.body_pos_w[:, right_wheel_id])
    com_minus_wheel_b = quat_apply_inverse(asset.data.root_quat_w, com_pos_w - wheel_center_w)
    return com_minus_wheel_b[:, 0]


def tracking_lin_vel_x(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跟踪前进速度 vx。这里约定 command[:, 0] 是目标前进速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-error / tracking_sigma)


def vx_tracking_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """vx 主跟踪奖励：exp(-((vx-vx_cmd)^2)/(sigma^2))。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(error) / sigma_sq), penalty=0.0)


def vx_tracking_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """vx Huber 误差项：大误差时仍保留稳定梯度，避免 Gaussian 远处梯度消失。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.nan_to_num(asset.data.root_lin_vel_b[:, 0] - command[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    return _finite_or_penalty(torch.clamp(loss, max=max_value), penalty=max_value)


def tracking_lin_vel_x_enhance(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fudan-style relaxed vx tracking: exp(-error / (10*sigma)) - 1."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return _finite_or_penalty(torch.exp(-error / tracking_sigma / 10.0) - 1.0, penalty=-1.0)


def lin_vel_x_tracking_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Direct vx tracking error penalty: (vx_cmd - vx_actual)^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    return _finite_or_penalty(torch.clamp(error, max=max_value), penalty=max_value)


def lin_vel_x_tracking_error_l2_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_command_name: str,
    height_gate_margin: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """高度接近目标后再惩罚 vx 跟踪误差，避免未站稳时速度目标和高度目标互相抢梯度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    gate = _height_command_gate(env, height_command_name, height_gate_margin, asset_cfg)
    value = torch.clamp(error, max=max_value) * gate
    return _finite_or_penalty(value, penalty=max_value)


def tracking_lin_vel_x_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_command_name: str,
    tracking_sigma: float,
    height_gate_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 base link 接近目标高度时奖励 vx 跟踪。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-error / tracking_sigma)
    gate = _height_command_gate(env, height_command_name, height_gate_margin, asset_cfg)
    return _finite_or_penalty(reward * gate, penalty=0.0)


def tracking_lin_vel_xy(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跟踪 base link 的 x/y 平面线速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return _finite_or_penalty(torch.exp(-error / tracking_sigma), penalty=0.0)


def tracking_yaw_rate(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跟踪偏航角速度。Isaac Lab 速度命令中 command[:, 2] 是目标 yaw rate。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return _finite_or_penalty(torch.exp(-error / tracking_sigma), penalty=0.0)


def tracking_yaw_rate_enhance(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fudan-style relaxed yaw tracking: exp(-error / (10*sigma)) - 1."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return _finite_or_penalty(torch.exp(-error / tracking_sigma / 10.0) - 1.0, penalty=-1.0)


def base_height_command_exp_kernel(
    env: ManagerBasedRLEnv,
    command_name: str,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum height tracking reward: exp(-k * (base_height - target_height)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error_sq = torch.square(asset.data.root_pos_w[:, 2] - command[:, 0])
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def orientation_exp_kernel(
    env: ManagerBasedRLEnv,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum orientation reward: exp(-k * (roll_error^2 + pitch_error^2))."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_error_sq = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return _finite_or_penalty(torch.exp(-kernel_coeff * roll_pitch_error_sq), penalty=0.0)


def base_ang_vel_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum angular velocity penalty: wx^2 + wy^2 + wz^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def velocity_tracking_x_exp_kernel(
    env: ManagerBasedRLEnv,
    command_name: str,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum vx tracking reward: exp(-k * (vx - vx_target)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error_sq = torch.square(asset.data.root_lin_vel_b[:, 0] - command[:, 0])
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def pitch_tracking_gaussian(
    env: ManagerBasedRLEnv,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """pitch 辅助平衡奖励：用 projected_gravity_b[:,0] 近似 sin(pitch)。"""
    asset: Articulation = env.scene[asset_cfg.name]
    pitch_proxy = asset.data.projected_gravity_b[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(pitch_proxy) / sigma_sq), penalty=0.0)


def pitch_rate_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """pitch 角速度惩罚，抑制大幅前后摆动，但不完全禁止加速时的自然俯仰。"""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.square(asset.data.root_ang_vel_b[:, 1])
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def yaw_rate_tracking_exp_kernel(
    env: ManagerBasedRLEnv,
    command_name: str,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum yaw tracking reward: exp(-k * (wz - yaw_rate_target)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error_sq = torch.square(asset.data.root_ang_vel_b[:, 2] - command[:, 2])
    return _finite_or_penalty(torch.exp(-kernel_coeff * error_sq), penalty=0.0)


def lateral_lin_vel_y_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum lateral velocity penalty: base_vy^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.square(asset.data.root_lin_vel_b[:, 1])
    return _finite_or_penalty(value, penalty=max_value)


def foot_clearance_exp_kernel(
    env: ManagerBasedRLEnv,
    target_clearance: float,
    kernel_coeff: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Stage 2 clearance reward for foot/wheel bodies: mean exp(-k * (z - target)^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(1)
    error_sq = torch.square(body_height - target_clearance)
    value = torch.mean(torch.exp(-kernel_coeff * error_sq), dim=1)
    return _finite_or_penalty(value, penalty=0.0)


def contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Stage 2 contact reward: fraction of selected foot/wheel bodies in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    in_contact = torch.max(contact_force, dim=1)[0] > threshold
    value = torch.mean(in_contact.float(), dim=1)
    return _finite_or_penalty(value, penalty=0.0)


def foot_slip_l2(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    max_value: float,
) -> torch.Tensor:
    """Stage 3 slip penalty: sum(contact * foot_velocity_xy^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    in_contact = torch.max(contact_force, dim=1)[0] > threshold
    body_lin_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    slip = torch.sum(torch.square(body_lin_vel_xy), dim=-1)
    value = torch.sum(torch.where(in_contact, slip, torch.zeros_like(slip)), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def yaw_rate_l2_when_no_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_deadband: float,
    yaw_rate_deadband: float,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 yaw 命令接近 0 时惩罚实际 yaw rate，避免 vx-only 阶段自转。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    no_yaw_command = torch.abs(command[:, 2]) <= command_deadband
    yaw_rate_excess = torch.clamp(torch.abs(asset.data.root_ang_vel_b[:, 2]) - yaw_rate_deadband, min=0.0)
    value = torch.where(no_yaw_command, torch.square(yaw_rate_excess), torch.zeros_like(yaw_rate_excess))
    return _finite_or_penalty(value, penalty=max_value)


def base_height_command_tracking_fudan(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fudan base-height tracking: exp(-(base_h - cmd_h)^2 / 0.001)."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(asset.data.root_pos_w[:, 2] - command[:, 0])
    return _finite_or_penalty(torch.exp(-error / 0.001), penalty=0.0)


def tracking_yaw_rate_height_gated(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_command_name: str,
    tracking_sigma: float,
    height_gate_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 base link 接近目标高度时奖励 yaw rate 跟踪。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-error / tracking_sigma)
    gate = _height_command_gate(env, height_command_name, height_gate_margin, asset_cfg)
    return _finite_or_penalty(reward * gate, penalty=0.0)


def base_height_tracking(
    env: ManagerBasedRLEnv,
    target_height: float,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """跟踪固定机身高度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-error / tracking_sigma)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机身高度偏离目标值。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite_or_penalty(torch.square(asset.data.root_pos_w[:, 2] - target_height), penalty=1.0)


def base_height_command_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机身高度偏离目标高度命令。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return _finite_or_penalty(torch.square(asset.data.root_pos_w[:, 2] - command[:, 0]), penalty=1.0)


def base_height_command_scaled_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用厘米级尺度归一化高度误差，避免米²惩罚数值太小。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    scaled_error = (asset.data.root_pos_w[:, 2] - command[:, 0]) / error_scale
    return _finite_or_penalty(torch.square(scaled_error), penalty=1.0)


def base_height_command_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励机身高度接近目标高度命令。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error = torch.square(asset.data.root_pos_w[:, 2] - command[:, 0])
    return _finite_or_penalty(torch.exp(-error / tracking_sigma), penalty=0.0)


def base_height_too_high_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在机身高度高于命令 + margin 时单边惩罚；拉力不随误差衰减，专治"上得去下不来"。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    height_excess = torch.clamp(asset.data.root_pos_w[:, 2] - command[:, 0] - margin, min=0.0)
    return _finite_or_penalty(torch.square(height_excess), penalty=1.0)


def base_height_too_high_scaled_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用厘米级尺度归一化高于命令的单边高度误差。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    scaled_excess = torch.clamp(asset.data.root_pos_w[:, 2] - command[:, 0] - margin, min=0.0) / error_scale
    return _finite_or_penalty(torch.square(scaled_excess), penalty=1.0)


def base_height_too_low_scaled_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """用厘米级尺度归一化低于命令的单边高度误差，堵住"赖在底部"的免费安全区。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    scaled_deficit = torch.clamp(command[:, 0] - asset.data.root_pos_w[:, 2] - margin, min=0.0) / error_scale
    return _finite_or_penalty(torch.square(scaled_deficit), penalty=1.0)


def base_height_velocity_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    tracking_sigma: float,
    height_gate_margin: float,
    kp: float,
    max_desired_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """奖励跟随"前馈命令速率 + 高度误差反馈"的合成目标速度。

    desired_vz = cmd_rate + kp * (cmd - height)，限幅 ±max_desired_velocity：
    高于目标时 desired_vz 为负，明确鼓励主动向下追，而不是停在原高度。
    使用世界坐标系 z 速度，与 root_pos_w 的高度定义保持一致。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    height = asset.data.root_pos_w[:, 2]
    height_error = command[:, 0] - height
    desired_vz = torch.clamp(
        command[:, 1] + kp * height_error,
        min=-max_desired_velocity,
        max=max_desired_velocity,
    )
    velocity_error = torch.square(asset.data.root_lin_vel_w[:, 2] - desired_vz)
    gate = torch.clamp(1.0 - torch.abs(height_error) / height_gate_margin, min=0.0, max=1.0)
    descending_from_above = (command[:, 1] < 0.0) & (height > command[:, 0])
    gate = torch.where(descending_from_above, torch.ones_like(gate), gate)
    reward = torch.exp(-velocity_error / tracking_sigma) * gate
    return _finite_or_penalty(reward, penalty=0.0)


def base_height_velocity_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    kp: float,
    max_desired_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚 base z 速度偏离高度误差反馈合成的目标速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    height_error = command[:, 0] - asset.data.root_pos_w[:, 2]
    desired_vz = torch.clamp(
        command[:, 1] + kp * height_error,
        min=-max_desired_velocity,
        max=max_desired_velocity,
    )
    value = torch.square(asset.data.root_lin_vel_w[:, 2] - desired_vz)
    return _finite_or_penalty(value, penalty=1.0)


def upward_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只惩罚 base link 过快向上弹起，允许低速升高。"""
    asset: Articulation = env.scene[asset_cfg.name]
    upward_vel = torch.clamp(asset.data.root_lin_vel_b[:, 2] - deadband, min=0.0)
    return _finite_or_penalty(torch.square(upward_vel), penalty=1.0)


def base_height_below_minimum_l2(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 base link 高度低于阈值时惩罚。"""
    asset: Articulation = env.scene[asset_cfg.name]
    height_deficit = torch.clamp(minimum_height - asset.data.root_pos_w[:, 2], min=0.0)
    return _finite_or_penalty(torch.square(height_deficit), penalty=1.0)


def base_lin_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚 base link 水平线速度，不惩罚 z —— 升降阶段需要主动产生 z 速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return _finite_or_penalty(
        torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1),
        penalty=100.0,
    )


def base_lin_vel_xy_l2_when_no_linear_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    vx_command_deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 vx command 接近 0 时惩罚 base 水平速度。"""
    command = env.command_manager.get_command(command_name)
    active = torch.abs(command[:, 0]) < vx_command_deadband
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    gated_value = torch.where(active, value, torch.zeros_like(value))
    return _finite_or_penalty(gated_value, penalty=100.0)


def root_xy_position_l2(
    env: ManagerBasedRLEnv,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚 base link 在各自 env 原点附近的水平漂移。"""
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    distance = torch.linalg.norm(root_xy, dim=1)
    excess = torch.clamp(distance - deadband, min=0.0)
    return _finite_or_penalty(torch.square(excess), penalty=10.0)


def root_xy_position_l2_when_no_linear_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    vx_command_deadband: float,
    deadband: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """只在 vx command 接近 0 时惩罚 base 偏离 env 原点。"""
    command = env.command_manager.get_command(command_name)
    active = torch.abs(command[:, 0]) < vx_command_deadband
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    distance = torch.linalg.norm(root_xy, dim=1)
    excess = torch.clamp(distance - deadband, min=0.0)
    value = torch.square(excess)
    gated_value = torch.where(active, value, torch.zeros_like(value))
    return _finite_or_penalty(gated_value, penalty=10.0)


def com_wheel_alignment_l2(
    env: ManagerBasedRLEnv,
    deadband: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚整机 COM 在 base x 方向偏离左右轮中点。

    定义与 diagnose_com_height.py 的 COMx-wheelx_b 一致：
    先计算整机质量加权 COM 和左右轮中心点，再将二者差值转到 base frame 并取 x。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_error_x = _com_minus_wheel_x_b(asset, asset_cfg.body_ids)
    error = torch.clamp(torch.abs(com_error_x) - deadband, min=0.0)
    scaled_error = error / error_scale
    return _finite_or_penalty(torch.square(scaled_error), penalty=1.0)


def com_wheel_alignment_l2_when_no_linear_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    vx_command_deadband: float,
    deadband: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """只在 vx command 接近 0 时约束 COM 与左右轮中点的 base x 对齐。"""
    command = env.command_manager.get_command(command_name)
    active = torch.abs(command[:, 0]) < vx_command_deadband
    asset: Articulation = env.scene[asset_cfg.name]
    com_error_x = _com_minus_wheel_x_b(asset, asset_cfg.body_ids)
    error = torch.clamp(torch.abs(com_error_x) - deadband, min=0.0)
    scaled_error = error / error_scale
    value = torch.square(scaled_error)
    gated_value = torch.where(active, value, torch.zeros_like(value))
    return _finite_or_penalty(gated_value, penalty=1.0)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机身 z 方向速度，避免上下跳动。"""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.square(asset.data.root_lin_vel_b[:, 2])
    return _finite_or_penalty(value, penalty=max_value)


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚 roll/pitch 角速度，避免机身快速翻滚或点头。"""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """惩罚机身倾斜，使用机体系下重力方向的 x/y 分量。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def yaw_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """临时惩罚 base yaw 角，抑制固定命令下原地转圈。"""
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat.unbind(dim=1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return _finite_or_penalty(torch.square(yaw), penalty=10.0)


def leg_length_l2(
    env: ManagerBasedRLEnv,
    target_length: float,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """惩罚左右腿当前腿长偏离目标腿长。"""
    leg_state = usd_leg_kinematics(env, asset_cfg, left_hip_offset_b, right_hip_offset_b)
    length_error = leg_state[:, 0:2] - target_length
    return torch.sum(torch.square(length_error), dim=1)


def base_height_commanded_leg_vertical_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    error_scale: float,
    deadband: float,
    wheel_radius: float,
    min_target_length: float,
    max_target_length: float,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """Penalize vertical hip-to-wheel support height implied by base-height command.

    This uses direct point geometry: hip point -> wheel center. A small deadband
    absorbs millimeter-level left/right USD or closed-chain solver asymmetry.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if len(body_ids) != 3:
        raise ValueError(
            "base_height_commanded_leg_vertical_l2 expects exactly three bodies: "
            "[base_link, Left_Wheel_link, Right_Wheel_link]. "
            f"Resolved body ids: {body_ids}"
        )

    command = env.command_manager.get_command(command_name)
    base_id, left_wheel_id, right_wheel_id = body_ids
    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]
    left_wheel_pos_b = quat_apply_inverse(base_quat_w, asset.data.body_pos_w[:, left_wheel_id] - base_pos_w)
    right_wheel_pos_b = quat_apply_inverse(base_quat_w, asset.data.body_pos_w[:, right_wheel_id] - base_pos_w)

    left_vertical = left_hip_offset_b[2] - left_wheel_pos_b[:, 2]
    right_vertical = right_hip_offset_b[2] - right_wheel_pos_b[:, 2]
    vertical_support = torch.stack((left_vertical, right_vertical), dim=1)

    hip_z = 0.5 * (left_hip_offset_b[2] + right_hip_offset_b[2])
    target_vertical = command[:, 0] + hip_z - wheel_radius
    target_vertical = torch.clamp(target_vertical, min=min_target_length, max=max_target_length)
    error = vertical_support - target_vertical.unsqueeze(1)
    error = torch.sign(error) * torch.clamp(torch.abs(error) - deadband, min=0.0)
    scaled_error = error / error_scale
    return _finite_or_penalty(torch.sum(torch.square(scaled_error), dim=1), penalty=1.0)


def leg_length_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """惩罚左右腿长不一致。"""
    leg_state = usd_leg_kinematics(env, asset_cfg, left_hip_offset_b, right_hip_offset_b)
    return torch.square(leg_state[:, 0] - leg_state[:, 1])


def leg_angle_symmetry_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """Fudan nominal_state analogue using point-geometry leg swing angles."""
    leg_state = usd_leg_kinematics(env, asset_cfg, left_hip_offset_b, right_hip_offset_b)
    return _finite_or_penalty(torch.square(leg_state[:, 4] - leg_state[:, 5]), penalty=1.0)


def leg_length_below_minimum_l2(
    env: ManagerBasedRLEnv,
    minimum_length: float,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """惩罚任意一条腿短于最小腿长，避免单腿撑起、另一条腿缩下去。"""
    leg_state = usd_leg_kinematics(env, asset_cfg, left_hip_offset_b, right_hip_offset_b)
    length_deficit = torch.clamp(minimum_length - leg_state[:, 0:2], min=0.0)
    return torch.sum(torch.square(length_deficit), dim=1)


def mirrored_leg_joint_pattern_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from the mirrored standing pattern: left same, right opposite."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    if joint_pos.shape[1] != 4:
        raise ValueError(
            "mirrored_leg_joint_pattern_l2 expects four joints ordered as "
            "[Left_front, Left_rear, Right_front, Right_rear]."
        )
    left_front, left_rear, right_front, right_rear = joint_pos.unbind(dim=1)
    left_sync = torch.square(left_front - left_rear)
    right_sync = torch.square(right_front - right_rear)
    front_mirror = torch.square(left_front + right_front)
    rear_mirror = torch.square(left_rear + right_rear)
    return _finite_or_penalty(left_sync + right_sync + front_mirror + rear_mirror, penalty=1.0)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚关节速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def wheel_x_separation_l2(
    env: ManagerBasedRLEnv,
    deadband: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚左右轮在 base 坐标系 x 方向错开太多：deadband 内为 0，超出后平方增长。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if len(body_ids) != 3:
        raise ValueError(
            "wheel_x_separation_l2 expects exactly three bodies: "
            "[base_link, Left_Wheel_link, Right_Wheel_link]. "
            f"Resolved body ids: {body_ids}"
        )

    base_id = body_ids[0]
    left_wheel_id = body_ids[1]
    right_wheel_id = body_ids[2]

    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]
    left_wheel_pos_b = quat_apply_inverse(base_quat_w, asset.data.body_pos_w[:, left_wheel_id] - base_pos_w)
    right_wheel_pos_b = quat_apply_inverse(base_quat_w, asset.data.body_pos_w[:, right_wheel_id] - base_pos_w)

    x_separation = torch.abs(left_wheel_pos_b[:, 0] - right_wheel_pos_b[:, 0])
    excess_separation = torch.clamp(x_separation - deadband, min=0.0)
    return _finite_or_penalty(torch.square(excess_separation), penalty=1.0)


def joint_acc_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚关节加速度。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_acc = torch.nan_to_num(asset.data.joint_acc[:, asset_cfg.joint_ids], nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.sum(torch.square(joint_acc), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def joint_pos_limits_soft(
    env: ManagerBasedRLEnv,
    soft_ratio: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Fudan-style soft joint position limit penalty."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
    finite_limits = torch.isfinite(limits[..., 0]) & torch.isfinite(limits[..., 1])
    safe_lower = torch.where(finite_limits, limits[..., 0], joint_pos)
    safe_upper = torch.where(finite_limits, limits[..., 1], joint_pos)
    joint_pos_mean = 0.5 * (safe_lower + safe_upper)
    joint_pos_range = safe_upper - safe_lower
    soft_lower = joint_pos_mean - 0.5 * joint_pos_range * soft_ratio
    soft_upper = joint_pos_mean + 0.5 * joint_pos_range * soft_ratio
    out_of_limits = -(joint_pos - soft_lower).clip(max=0.0)
    out_of_limits += (joint_pos - soft_upper).clip(min=0.0)
    out_of_limits = torch.where(finite_limits, out_of_limits, torch.zeros_like(out_of_limits))
    return _finite_or_penalty(torch.sum(out_of_limits, dim=1), penalty=1.0)


def joint_torques_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚电机力矩。"""
    asset: Articulation = env.scene[asset_cfg.name]
    value = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def action_rate_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    action_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """惩罚 action 高频变化。"""
    action_delta = env.action_manager.action - env.action_manager.prev_action
    if action_slice is not None:
        start, stop = action_slice
        action_delta = action_delta[:, start:stop]
    action_delta = torch.nan_to_num(action_delta, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.sum(torch.square(action_delta), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def action_second_order_l2(
    env: ManagerBasedRLEnv,
    max_value: float,
    action_slice: tuple[int, int] | None = None,
) -> torch.Tensor:
    """惩罚二阶 action 差分，抑制目标角来回折返造成的高频抖动。"""
    action = env.action_manager.action
    if not hasattr(env, "_wheel_leg_prev_prev_action"):
        env._wheel_leg_prev_prev_action = env.action_manager.prev_action.clone()

    step = int(getattr(env, "common_step_counter", -1))
    if (
        not hasattr(env, "_wheel_leg_action_second_order_cache")
        or getattr(env, "_wheel_leg_action_second_order_cache_step", None) != step
    ):
        prev_action = env.action_manager.prev_action
        prev_prev_action = env._wheel_leg_prev_prev_action
        env._wheel_leg_action_second_order_cache = action - 2.0 * prev_action + prev_prev_action
        env._wheel_leg_action_second_order_cache_step = step
        env._wheel_leg_prev_prev_action = prev_action.clone()

    second_order = env._wheel_leg_action_second_order_cache

    if action_slice is not None:
        start, stop = action_slice
        second_order = second_order[:, start:stop]
    second_order = torch.nan_to_num(second_order, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.sum(torch.square(second_order), dim=1)
    return _finite_or_penalty(value, penalty=max_value)


def vx_wheel_antiphase_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    min_abs_vx_command: float,
    yaw_command_deadband: float,
    max_value: float,
) -> torch.Tensor:
    """For vx-only commands, encourage left/right wheel actions to be opposite-signed."""
    command = env.command_manager.get_command(command_name)
    active = (torch.abs(command[:, 0]) >= min_abs_vx_command) & (torch.abs(command[:, 2]) <= yaw_command_deadband)
    wheel_actions = env.action_manager.action[:, 4:6]
    wheel_sum = torch.nan_to_num(wheel_actions[:, 0] + wheel_actions[:, 1], nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.where(active, torch.square(wheel_sum), torch.zeros_like(wheel_sum))
    return _finite_or_penalty(value, penalty=max_value)


def straight_wheel_diff_l2_when_no_yaw_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    yaw_command_deadband: float,
    max_value: float,
) -> torch.Tensor:
    """Only when yaw_cmd is near zero, discourage wheel differential action that causes turning."""
    command = env.command_manager.get_command(command_name)
    active = torch.abs(command[:, 2]) <= yaw_command_deadband
    wheel_actions = env.action_manager.action[:, 4:6]
    # Forward-normalized convention: left_forward=raw_left, right_forward=-raw_right.
    # Differential action is what produces yaw; keep it small only for straight commands.
    wheel_diff = 0.5 * ((-wheel_actions[:, 1]) - wheel_actions[:, 0])
    wheel_diff = torch.nan_to_num(wheel_diff, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.where(active, torch.square(wheel_diff), torch.zeros_like(wheel_diff))
    return _finite_or_penalty(value, penalty=max_value)


def stand_wheel_common_action_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_command_deadband: float,
    yaw_command_deadband: float,
    max_value: float,
) -> torch.Tensor:
    """站立命令下惩罚轮子 common action，减少主动输出前后速度意图。"""
    command = env.command_manager.get_command(command_name)
    active = (torch.linalg.norm(command[:, :2], dim=1) <= linear_command_deadband) & (
        torch.abs(command[:, 2]) <= yaw_command_deadband
    )
    wheel_actions = env.action_manager.action[:, 4:6]
    # Forward-normalized convention: left_forward=raw_left, right_forward=-raw_right.
    # Common action produces forward/backward velocity intent.
    wheel_common = 0.5 * (wheel_actions[:, 0] - wheel_actions[:, 1])
    wheel_common = torch.nan_to_num(wheel_common, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.where(active, torch.square(wheel_common), torch.zeros_like(wheel_common))
    return _finite_or_penalty(value, penalty=max_value)


def stand_wheel_diff_action_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    linear_command_deadband: float,
    yaw_command_deadband: float,
    max_value: float,
) -> torch.Tensor:
    """站立命令下惩罚轮子 differential action，减少主动输出 yaw/差速意图。"""
    command = env.command_manager.get_command(command_name)
    active = (torch.linalg.norm(command[:, :2], dim=1) <= linear_command_deadband) & (
        torch.abs(command[:, 2]) <= yaw_command_deadband
    )
    wheel_actions = env.action_manager.action[:, 4:6]
    # Forward-normalized convention: left_forward=raw_left, right_forward=-raw_right.
    # Differential action produces yaw intent.
    wheel_diff = 0.5 * ((-wheel_actions[:, 1]) - wheel_actions[:, 0])
    wheel_diff = torch.nan_to_num(wheel_diff, nan=0.0, posinf=0.0, neginf=0.0)
    value = torch.where(active, torch.square(wheel_diff), torch.zeros_like(wheel_diff))
    return _finite_or_penalty(value, penalty=max_value)


def wheel_velocity_target_smooth_l1(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """辅助惩罚轮速目标与实际轮速误差，避免 policy 输出执行器长期跟不上的轮速 target。

    Smooth-L1 在小误差时近似 L2，保留细梯度；大误差时近似 L1，避免该项统治高度/姿态任务。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_term_name)
    target_vel = action_term.processed_actions
    actual_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    error = torch.nan_to_num(target_vel - actual_vel, nan=0.0, posinf=0.0, neginf=0.0)
    abs_error = torch.abs(error)
    beta = max(float(beta), 1.0e-6)
    loss = torch.where(abs_error < beta, 0.5 * torch.square(error) / beta, abs_error - 0.5 * beta)
    value = torch.sum(loss, dim=1)
    return _finite_or_penalty(torch.clamp(value, max=max_value), penalty=max_value)


def wheel_vx_tracking_gaussian(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    sigma: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """轮子前向速度跟踪奖励：鼓励 vx 主要由左右轮公共速度产生，而不是靠腿部摆动蹭出来。"""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.joint_ids) != 2:
        raise ValueError("wheel_vx_tracking_gaussian expects two wheel joints ordered [left, right].")
    command = env.command_manager.get_command(command_name)
    left_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[0]]
    right_wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids[1]]
    # 右轮关节正方向和左轮相反；统一成“正值代表车体前进”的 wheel-frame 速度。
    wheel_forward_vel = 0.5 * (left_wheel_vel - right_wheel_vel)
    vx_from_wheels = float(wheel_radius) * wheel_forward_vel
    error = vx_from_wheels - command[:, 0]
    sigma_sq = max(float(sigma) ** 2, 1.0e-6)
    return _finite_or_penalty(torch.exp(-torch.square(error) / sigma_sq), penalty=0.0)


def wheel_vx_tracking_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    wheel_radius: float,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """轮速-vx Huber 误差：远离目标时也给出明确梯度，推动 policy 使用轮子完成速度命令。"""
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
    env: ManagerBasedRLEnv,
    wheel_radius: float,
    beta: float,
    max_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """惩罚 base vx 与轮子滚动速度不一致，减少腿部摆动/滑移制造出的假前进速度。"""
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


def contact_sensor_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """检测指定 contact sensor 是否超过接触力阈值，作为持续惩罚项。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    has_contact = has_contact | torch.any(~torch.isfinite(contact_force.flatten(start_dim=1)), dim=1)
    return has_contact.float()


def wheel_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """任意轮子离地时返回 1，用作惩罚项。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    max_contact_force = torch.max(contact_force, dim=1)[0]
    wheel_in_contact = max_contact_force > threshold
    wheel_air = ~torch.all(wheel_in_contact, dim=1)
    invalid_contact = torch.any(~torch.isfinite(contact_force.flatten(start_dim=1)), dim=1)
    return (wheel_air | invalid_contact).float()


def _finite_or_penalty(value: torch.Tensor, penalty: float) -> torch.Tensor:
    """Keep reward terms finite on the same step that a bad state is terminated."""
    return torch.where(torch.isfinite(value), value, torch.full_like(value, penalty))


def _height_command_gate(
    env: ManagerBasedRLEnv,
    command_name: str,
    margin: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Smoothly opens from 0 to 1 when the base height is close to the commanded height."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - command[:, 0])
    return torch.clamp(1.0 - height_error / margin, min=0.0, max=1.0)
