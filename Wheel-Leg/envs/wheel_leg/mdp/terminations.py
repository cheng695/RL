import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as torch_utils

from .observations import usd_leg_kinematics


def root_height_below_minimum_after_steps(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    min_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身高度低于阈值时终止；前几步给机器人一个落地/恢复缓冲时间。"""
    asset: Articulation = env.scene[asset_cfg.name]
    too_low = asset.data.root_pos_w[:, 2] < minimum_height
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(too_low, after_grace)


def root_too_high(
    env: ManagerBasedRLEnv,
    maximum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身高度超过阈值时立即终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] > maximum_height


def root_linear_velocity_too_high(
    env: ManagerBasedRLEnv,
    maximum_linear_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """root 线速度过大时立即终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    root_lin_vel = torch.linalg.norm(asset.data.root_lin_vel_b, dim=1)
    return root_lin_vel > maximum_linear_velocity


def root_angular_velocity_too_high(
    env: ManagerBasedRLEnv,
    maximum_angular_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """root 角速度过大时立即终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    root_ang_vel = torch.linalg.norm(asset.data.root_ang_vel_b, dim=1)
    return root_ang_vel > maximum_angular_velocity


def joint_velocity_too_high(
    env: ManagerBasedRLEnv,
    maximum_joint_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """任意关节速度过大时立即终止。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.max(torch.abs(asset.data.joint_vel), dim=1)[0]
    return joint_vel > maximum_joint_velocity


def root_state_non_finite(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """任意 root/body/关节状态出现非有限值时立即终止，防止 NaN 污染 rollout。"""
    asset: Articulation = env.scene[asset_cfg.name]
    non_finite_root = torch.any(~torch.isfinite(asset.data.root_pos_w), dim=1)
    non_finite_root_vel = torch.any(~torch.isfinite(asset.data.root_lin_vel_b), dim=1)
    non_finite_root_ang_vel = torch.any(~torch.isfinite(asset.data.root_ang_vel_b), dim=1)
    non_finite_projected_gravity = torch.any(~torch.isfinite(asset.data.projected_gravity_b), dim=1)
    non_finite_body_pos = torch.any(~torch.isfinite(asset.data.body_pos_w.flatten(start_dim=1)), dim=1)
    non_finite_body_vel = torch.any(~torch.isfinite(asset.data.body_lin_vel_w.flatten(start_dim=1)), dim=1)
    non_finite_joint_pos = torch.any(~torch.isfinite(asset.data.joint_pos), dim=1)
    non_finite_joint_vel = torch.any(~torch.isfinite(asset.data.joint_vel), dim=1)
    return (
        non_finite_root
        | non_finite_root_vel
        | non_finite_root_ang_vel
        | non_finite_projected_gravity
        | non_finite_body_pos
        | non_finite_body_vel
        | non_finite_joint_pos
        | non_finite_joint_vel
    )


def bad_orientation_after_steps(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    min_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """机身倾角过大时终止，对应 Wheel-Legged-Gym 里的 projected_gravity 翻倒判断。"""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0))
    bad_orientation = tilt_angle > limit_angle
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(bad_orientation, after_grace)


def joint_position_deviation_after_steps(
    env: ManagerBasedRLEnv,
    maximum_deviation: float,
    min_steps: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """关节偏离默认姿态太远时终止，避免闭链被策略推到奇异姿态。"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_deviation = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids])
    too_far = torch.any(joint_deviation > maximum_deviation, dim=1)
    non_finite_joint_pos = torch.any(~torch.isfinite(asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(too_far | non_finite_joint_pos, after_grace)


def leg_length_below_minimum_after_steps(
    env: ManagerBasedRLEnv,
    minimum_length: float,
    min_steps: int,
    asset_cfg: SceneEntityCfg,
    left_hip_offset_b: tuple[float, float, float],
    right_hip_offset_b: tuple[float, float, float],
) -> torch.Tensor:
    """任意一条腿过短时终止，防止闭链机构进入危险/奇异姿态。"""
    leg_state = usd_leg_kinematics(env, asset_cfg, left_hip_offset_b, right_hip_offset_b)
    too_short = torch.any(leg_state[:, 0:2] < minimum_length, dim=1)
    non_finite_leg_state = torch.any(~torch.isfinite(leg_state), dim=1)
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(too_short | non_finite_leg_state, after_grace)


def wheel_x_separation_after_steps(
    env: ManagerBasedRLEnv,
    maximum_separation: float,
    min_steps: int,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """左右轮在 base 坐标系 x 方向错开过大时终止，避免劈叉导致 base 触地。"""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    if len(body_ids) != 3:
        raise ValueError(
            "wheel_x_separation_after_steps expects exactly three bodies: "
            "[base_link, Left_Wheel_link, Right_Wheel_link]. "
            f"Resolved body ids: {body_ids}"
        )

    base_id, left_wheel_id, right_wheel_id = body_ids
    left_wheel_pos_b = asset.data.body_pos_w[:, left_wheel_id] - asset.data.body_pos_w[:, base_id]
    right_wheel_pos_b = asset.data.body_pos_w[:, right_wheel_id] - asset.data.body_pos_w[:, base_id]
    base_quat_w = asset.data.body_quat_w[:, base_id]
    left_wheel_pos_b = torch_utils.quat_apply_inverse(base_quat_w, left_wheel_pos_b)
    right_wheel_pos_b = torch_utils.quat_apply_inverse(base_quat_w, right_wheel_pos_b)
    x_separation = torch.abs(left_wheel_pos_b[:, 0] - right_wheel_pos_b[:, 0])
    too_far = x_separation >= maximum_separation
    non_finite = ~torch.isfinite(x_separation)
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(too_far | non_finite, after_grace)


def illegal_contact_after_steps(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_steps: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """指定 body 接触力超过阈值时终止，通常用于 base_link 触地。"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_force = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    has_contact = torch.any(torch.max(contact_force, dim=1)[0] > threshold, dim=1)
    after_grace = env.episode_length_buf > min_steps
    return torch.logical_and(has_contact, after_grace)
