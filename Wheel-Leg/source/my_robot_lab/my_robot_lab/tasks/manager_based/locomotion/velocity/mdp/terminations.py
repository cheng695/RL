from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def bad_roll_pitch(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root body tilts beyond the allowed roll/pitch angle."""
    asset: Articulation = env.scene[asset_cfg.name]
    tilt_angle = torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0))
    return tilt_angle > limit_angle


def leg_tendon_length_out_of_range(
    env: ManagerBasedRLEnv,
    min_length: float,
    max_length: float,
    tolerance: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when either leg extension tendon is outside its allowed length range."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    left_length = torch.linalg.norm(body_pos_w[:, 1] - body_pos_w[:, 0], dim=1)
    right_length = torch.linalg.norm(body_pos_w[:, 3] - body_pos_w[:, 2], dim=1)
    lengths = torch.stack((left_length, right_length), dim=1)
    return torch.any((lengths < min_length - tolerance) | (lengths > max_length + tolerance), dim=1)


def illegal_contact_after_steps(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_steps: int,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate on illegal contact only after the episode has passed a minimum number of steps."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    has_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )
    return has_contact & (env.episode_length_buf >= min_steps)
