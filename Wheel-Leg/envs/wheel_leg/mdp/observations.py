import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse


def velocity_command_x_yaw(
    env: ManagerBasedRLEnv,
    command_name: str,
    lin_vel_scale: float,
    yaw_rate_scale: float,
) -> torch.Tensor:
    """Return scaled forward velocity and yaw-rate commands."""
    command = env.command_manager.get_command(command_name)
    return torch.stack(
        (
            command[:, 0] * lin_vel_scale,
            command[:, 2] * yaw_rate_scale,
        ),
        dim=1,
    )


def base_height_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    height_scale: float,
) -> torch.Tensor:
    """Return the scaled base-height command without the internal sweep rate."""
    command = env.command_manager.get_command(command_name)
    return command[:, 0:1] * height_scale


def virtual_leg_kinematics(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    offset: float,
    l1: float,
    l2: float,
    left_front_sign: float = 1.0,
    left_rear_sign: float = 1.0,
    right_front_sign: float = -1.0,
    right_rear_sign: float = -1.0,
    rear_joint_bias: float = torch.pi / 2,
) -> torch.Tensor:
    """Compute deployable virtual leg state from the four active leg joint encoders.

    Returns [L_l, L_r, dL_l, dL_r, theta_l, theta_r, dtheta_l, dtheta_r].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if joint_pos.shape[1] != 4:
        raise ValueError(
            "virtual_leg_kinematics expects four joints ordered as "
            "[Left_front, Left_rear, Right_front, Right_rear]."
        )

    lf_q, lr_q, rf_q, rr_q = joint_pos.unbind(dim=1)
    lf_dq, lr_dq, rf_dq, rr_dq = joint_vel.unbind(dim=1)

    theta1 = torch.stack((left_front_sign * lf_q, right_front_sign * rf_q), dim=1)
    theta2 = torch.stack(
        (
            left_rear_sign * lr_q + rear_joint_bias,
            right_rear_sign * rr_q + rear_joint_bias,
        ),
        dim=1,
    )
    theta1_dot = torch.stack((left_front_sign * lf_dq, right_front_sign * rf_dq), dim=1)
    theta2_dot = torch.stack((left_rear_sign * lr_dq, right_rear_sign * rr_dq), dim=1)

    theta12 = theta1 + theta2
    theta12_dot = theta1_dot + theta2_dot
    end_x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta12)
    end_y = l1 * torch.sin(theta1) + l2 * torch.sin(theta12)
    end_x_dot = -l1 * torch.sin(theta1) * theta1_dot - l2 * torch.sin(theta12) * theta12_dot
    end_y_dot = l1 * torch.cos(theta1) * theta1_dot + l2 * torch.cos(theta12) * theta12_dot

    leg_length = torch.sqrt(end_x * end_x + end_y * end_y).clamp_min(1.0e-6)
    leg_length_rate = (end_x * end_x_dot + end_y * end_y_dot) / leg_length
    leg_angle = torch.atan2(end_y, end_x) - torch.pi / 2
    leg_angle_rate = (end_x * end_y_dot - end_y * end_x_dot) / (leg_length * leg_length)

    return torch.stack(
        (
            leg_length[:, 0],
            leg_length[:, 1],
            leg_length_rate[:, 0],
            leg_length_rate[:, 1],
            leg_angle[:, 0],
            leg_angle[:, 1],
            leg_angle_rate[:, 0],
            leg_angle_rate[:, 1],
        ),
        dim=1,
    )


def usd_leg_kinematics(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """从 USD body 状态计算左右腿的腿长、腿长速度、腿角、腿角速度。

    返回顺序:
      [L_L, L_R, dL_L, dL_R, theta_L, theta_R, dtheta_L, dtheta_R]

    asset_cfg 的 body 顺序必须是:
      [base_link, Left_Wheel_link, Right_Wheel_link]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if len(body_ids) != 3:
        raise ValueError(
            "usd_leg_kinematics expects exactly three bodies: "
            "[base_link, Left_Wheel_link, Right_Wheel_link]. "
            f"Resolved body ids: {body_ids}"
        )

    base_id = body_ids[0]
    left_wheel_id = body_ids[1]
    right_wheel_id = body_ids[2]

    base_pos_w = asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]
    base_lin_vel_w = asset.data.body_lin_vel_w[:, base_id]
    base_ang_vel_w = asset.data.body_ang_vel_w[:, base_id]
    base_ang_vel_b = quat_apply_inverse(base_quat_w, base_ang_vel_w)

    left_hip_offset = torch.tensor(left_hip_offset_b, dtype=base_pos_w.dtype, device=base_pos_w.device).unsqueeze(0)
    right_hip_offset = torch.tensor(right_hip_offset_b, dtype=base_pos_w.dtype, device=base_pos_w.device).unsqueeze(0)

    left_hip_pos_w, left_hip_vel_w = _offset_point_state(
        base_pos_w,
        base_quat_w,
        base_lin_vel_w,
        base_ang_vel_w,
        left_hip_offset,
    )
    right_hip_pos_w, right_hip_vel_w = _offset_point_state(
        base_pos_w,
        base_quat_w,
        base_lin_vel_w,
        base_ang_vel_w,
        right_hip_offset,
    )

    left_state = _leg_state_from_points(
        base_quat_w,
        base_ang_vel_b,
        left_hip_pos_w,
        left_hip_vel_w,
        asset.data.body_pos_w[:, left_wheel_id],
        asset.data.body_lin_vel_w[:, left_wheel_id],
    )
    right_state = _leg_state_from_points(
        base_quat_w,
        base_ang_vel_b,
        right_hip_pos_w,
        right_hip_vel_w,
        asset.data.body_pos_w[:, right_wheel_id],
        asset.data.body_lin_vel_w[:, right_wheel_id],
    )

    leg_state = torch.stack(
        (
            left_state[:, 0],
            right_state[:, 0],
            left_state[:, 1],
            right_state[:, 1],
            left_state[:, 2],
            right_state[:, 2],
            left_state[:, 3],
            right_state[:, 3],
        ),
        dim=1,
    )
    return leg_state


def _offset_point_state(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    offset_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset_w = quat_apply(body_quat_w, offset_b.expand_as(body_pos_w))
    point_pos_w = body_pos_w + offset_w
    point_vel_w = body_lin_vel_w + torch.cross(body_ang_vel_w, offset_w, dim=1)
    return point_pos_w, point_vel_w


def _leg_state_from_points(
    base_quat_w: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    hip_pos_w: torch.Tensor,
    hip_vel_w: torch.Tensor,
    wheel_pos_w: torch.Tensor,
    wheel_vel_w: torch.Tensor,
) -> torch.Tensor:
    leg_vec_w = wheel_pos_w - hip_pos_w
    leg_vel_w = wheel_vel_w - hip_vel_w
    leg_vec_b = quat_apply_inverse(base_quat_w, leg_vec_w)
    leg_vel_b = quat_apply_inverse(base_quat_w, leg_vel_w) - torch.cross(base_ang_vel_b, leg_vec_b, dim=1)

    x = leg_vec_b[:, 0]
    z_down = -leg_vec_b[:, 2]
    vx = leg_vel_b[:, 0]
    vz_down = -leg_vel_b[:, 2]
    leg_length = torch.sqrt(x * x + z_down * z_down).clamp_min(1.0e-6)
    leg_length_rate = (x * vx + z_down * vz_down) / leg_length
    theta = torch.atan2(x, z_down)
    theta_rate = (z_down * vx - x * vz_down) / (leg_length * leg_length)

    return torch.stack(
        (
            leg_length,
            leg_length_rate,
            theta,
            theta_rate,
        ),
        dim=1,
    )
