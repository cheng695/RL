from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class UniformBaseHeightCommand(CommandTerm):
    """Uniformly sampled base-height command."""

    cfg: "UniformBaseHeightCommandCfg"

    def __init__(self, cfg: "UniformBaseHeightCommandCfg", env):
        super().__init__(cfg, env)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.height_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.height_range
        )

    def _update_command(self):
        pass


@configclass
class UniformBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for a uniformly sampled base-height command."""

    class_type: type = UniformBaseHeightCommand
    asset_name: str = "robot"
    height_range: tuple[float, float] = (0.28, 0.32)
