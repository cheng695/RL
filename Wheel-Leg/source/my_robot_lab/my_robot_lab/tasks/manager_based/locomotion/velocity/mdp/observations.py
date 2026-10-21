from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Imu


def imu_lin_vel(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Return linear velocity measured in the configured IMU frame."""
    imu: Imu = env.scene[sensor_cfg.name]
    return imu.data.lin_vel_b


def imu_ang_vel(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Return angular velocity measured by the configured IMU frame."""
    imu: Imu = env.scene[sensor_cfg.name]
    return imu.data.ang_vel_b


def imu_projected_gravity(env, sensor_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Return gravity direction measured in the configured IMU frame."""
    imu: Imu = env.scene[sensor_cfg.name]
    return imu.data.projected_gravity_b


def base_height_command(env, command_name: str) -> torch.Tensor:
    """Return the commanded base height."""
    return env.command_manager.get_command(command_name)


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return body height relative to the task ground reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if body_height.ndim > 1:
        body_height = torch.mean(body_height, dim=1)
    return (body_height - local_ground_height(env)).unsqueeze(-1)


def base_height_error(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return actual base height minus commanded base height."""
    return base_height(env, asset_cfg) - env.command_manager.get_command(command_name)


def local_ground_height(env) -> torch.Tensor:
    """Local vertical ground reference; zero for the original flat task.

    A small downward grid estimates ground under the chassis, not exact wheel
    contact clearance. Invalid scans use the tile origin until termination.
    """
    if not getattr(env.cfg, "terrain_relative_height", False):
        return torch.zeros(env.num_envs, device=env.device)
    z = env.scene["height_scanner"].data.ray_hits_w[..., 2]
    valid = torch.isfinite(z)
    ground = torch.where(valid, z, torch.zeros_like(z)).sum(-1) / valid.sum(-1).clamp_min(1)
    return torch.where(valid.any(-1), ground, env.scene.env_origins[:, 2])
