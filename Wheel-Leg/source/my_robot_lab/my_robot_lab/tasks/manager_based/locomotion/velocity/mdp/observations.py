from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def base_height_command(env, command_name: str) -> torch.Tensor:
    """Return the commanded base height."""
    return env.command_manager.get_command(command_name)


def base_height(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                height_reference_command: str | None = None) -> torch.Tensor:
    """Return the selected body height in world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if body_height.ndim > 1:
        body_height = torch.mean(body_height, dim=1)
    if height_reference_command is not None:
        body_height = body_height - env.command_manager.get_term(height_reference_command).reference_ground_height()
    return body_height.unsqueeze(-1)


def base_height_error(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_reference_command: str | None = None,
) -> torch.Tensor:
    """Return actual base height minus commanded base height."""
    return base_height(env, asset_cfg, height_reference_command) - env.command_manager.get_command(command_name)
