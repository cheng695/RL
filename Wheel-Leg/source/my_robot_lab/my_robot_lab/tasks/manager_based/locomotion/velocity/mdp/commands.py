from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass


class PositiveBiasedVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with mode-based vx/yaw sampling."""

    cfg: "PositiveBiasedVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        for name in ("vx_mae_mps", "yaw_mae_radps", "action_at_limit_fraction"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        self._metric_steps += 1
        values = {
            "vx_mae_mps": torch.abs(self.command[:, 0] - self.robot.data.root_lin_vel_b[:, 0]),
            "yaw_mae_radps": torch.abs(self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2]),
            "action_at_limit_fraction": (
                self._env.action_manager.action.abs() >= self.cfg.action_limit - 1.0e-6
            ).float().mean(dim=-1),
        }
        # Per-episode running means do not depend on command resampling duration.
        for name, value in values.items():
            self.metrics[name] += (value - self.metrics[name]) / self._metric_steps

    def reset(self, env_ids=None):
        extras = super().reset(env_ids)
        self._metric_steps[slice(None) if env_ids is None else env_ids] = 0
        return extras

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer = VisualizationMarkers(self.cfg.goal_yaw_visualizer_cfg)
                self.current_yaw_visualizer = VisualizationMarkers(self.cfg.current_yaw_visualizer_cfg)
            self.goal_yaw_visualizer.set_visibility(True)
            self.current_yaw_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_yaw_visualizer"):
                self.goal_yaw_visualizer.set_visibility(False)
                self.current_yaw_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized or not hasattr(self, "goal_yaw_visualizer"):
            return

        goal_pos_w = self._yaw_marker_position(self.cfg.goal_yaw_marker_offset)
        current_pos_w = self._yaw_marker_position(self.cfg.current_yaw_marker_offset)
        goal_scale, goal_quat = self._resolve_yaw_rate_to_arrow(self.command[:, 2])
        current_scale, current_quat = self._resolve_yaw_rate_to_arrow(self.robot.data.root_ang_vel_b[:, 2])

        self.goal_yaw_visualizer.visualize(goal_pos_w, goal_quat, goal_scale)
        self.current_yaw_visualizer.visualize(current_pos_w, current_quat, current_scale)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        num_resampled = len(env_ids)
        mode_values = torch.rand(num_resampled, device=self.device)

        command_prob_sum = (
            self.cfg.straight_command_prob + self.cfg.turn_command_prob + self.cfg.mixed_command_prob
        )
        if command_prob_sum <= 0.0:
            raise ValueError("At least one velocity command mode probability must be positive.")
        straight_cutoff = self.cfg.straight_command_prob / command_prob_sum
        turn_cutoff = (self.cfg.straight_command_prob + self.cfg.turn_command_prob) / command_prob_sum

        straight_envs = mode_values < straight_cutoff
        turn_envs = (mode_values >= straight_cutoff) & (mode_values < turn_cutoff)
        mixed_envs = mode_values >= turn_cutoff
        linear_envs = straight_envs | mixed_envs
        yaw_envs = turn_envs | mixed_envs

        self.vel_command_b[env_ids, :] = 0.0

        linear_env_ids = env_ids[linear_envs]
        yaw_env_ids = env_ids[yaw_envs]
        if len(linear_env_ids) > 0:
            self.vel_command_b[linear_env_ids, 0] = self._sample_lin_vel_x(len(linear_env_ids))
            self._apply_abrupt_lin_vel_x_flips(linear_env_ids)
        if len(yaw_env_ids) > 0:
            self.vel_command_b[yaw_env_ids, 2] = self._sample_ang_vel_z(len(yaw_env_ids))

        old_vx = getattr(self, "_previous_lin_vel_x_command", None)
        if old_vx is None or old_vx.shape[0] != self.num_envs:
            old_vx = torch.zeros(self.num_envs, device=self.device)
            setattr(self, "_previous_lin_vel_x_command", old_vx)
        old_vx[env_ids] = self.vel_command_b[env_ids, 0]

    def _sample_lin_vel_x(self, num_samples: int) -> torch.Tensor:
        x_min, x_max = self.cfg.ranges.lin_vel_x
        min_abs = min(abs(x_min), abs(x_max), self.cfg.min_abs_lin_vel_x)
        min_abs = max(min_abs, 0.0)

        random_values = torch.rand(num_samples, device=self.device)
        positive_envs = random_values < self.cfg.positive_lin_vel_x_prob

        pos_low = min(min_abs, max(x_max, 0.0))
        pos_high = max(x_max, pos_low)
        neg_low = min(x_min, -min_abs)
        neg_high = max(min(-min_abs, 0.0), neg_low)

        pos_samples = torch.empty(num_samples, device=self.device).uniform_(pos_low, pos_high)
        neg_samples = torch.empty(num_samples, device=self.device).uniform_(neg_low, neg_high)
        return torch.where(positive_envs, pos_samples, neg_samples)

    def _apply_abrupt_lin_vel_x_flips(self, env_ids: torch.Tensor):
        old_vx = getattr(self, "_previous_lin_vel_x_command", None)
        if old_vx is not None and old_vx.shape[0] == self.num_envs:
            x_min, x_max = self.cfg.ranges.lin_vel_x
            prev_vx = old_vx[env_ids]
            can_flip = torch.abs(prev_vx) >= self.cfg.min_abs_lin_vel_x
            flip_envs = (torch.rand(len(env_ids), device=self.device) < self.cfg.abrupt_flip_prob) & can_flip
            target_sign = -torch.sign(prev_vx)
            flip_high = min(max(abs(x_min), abs(x_max)), self.cfg.abrupt_flip_max_abs_lin_vel_x)
            flip_low = min(self.cfg.abrupt_flip_min_abs_lin_vel_x, flip_high)
            flip_abs = torch.empty(len(env_ids), device=self.device).uniform_(
                flip_low,
                flip_high,
            )
            flip_vx = torch.clamp(target_sign * flip_abs, min=x_min, max=x_max)
            self.vel_command_b[env_ids, 0] = torch.where(flip_envs, flip_vx, self.vel_command_b[env_ids, 0])

    def _sample_ang_vel_z(self, num_samples: int) -> torch.Tensor:
        z_min, z_max = self.cfg.ranges.ang_vel_z
        min_abs = min(abs(z_min), abs(z_max), self.cfg.min_abs_ang_vel_z)
        min_abs = max(min_abs, 0.0)

        pos_low = min(min_abs, max(z_max, 0.0))
        pos_high = max(z_max, pos_low)
        neg_low = min(z_min, -min_abs)
        neg_high = max(min(-min_abs, 0.0), neg_low)

        positive_turn = torch.rand(num_samples, device=self.device) < 0.5
        pos_samples = torch.empty(num_samples, device=self.device).uniform_(pos_low, pos_high)
        neg_samples = torch.empty(num_samples, device=self.device).uniform_(neg_low, neg_high)
        return torch.where(positive_turn, pos_samples, neg_samples)

    def _yaw_marker_position(self, offset_b: tuple[float, float, float]) -> torch.Tensor:
        offset = torch.tensor(offset_b, device=self.device).repeat(self.num_envs, 1)
        return self.robot.data.root_pos_w + math_utils.quat_apply_yaw(self.robot.data.root_quat_w, offset)

    def _resolve_yaw_rate_to_arrow(self, yaw_rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.cfg.goal_yaw_visualizer_cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.num_envs, 1)
        arrow_scale[:, 0] *= torch.clamp(torch.abs(yaw_rate) * self.cfg.yaw_visualizer_scale, min=0.05)

        turn_direction = torch.where(yaw_rate >= 0.0, 1.0, -1.0)
        zeros = torch.zeros_like(yaw_rate)
        local_yaw = turn_direction * torch.full_like(yaw_rate, 0.5 * torch.pi)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, local_yaw)
        arrow_quat = math_utils.quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class PositiveBiasedVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for biased forward-vx and abrupt velocity-transition training."""

    class_type: type = PositiveBiasedVelocityCommand
    action_limit: float = 1.0
    straight_command_prob: float = 0.3
    turn_command_prob: float = 0.5
    mixed_command_prob: float = 0.2
    positive_lin_vel_x_prob: float = 0.7
    zero_lin_vel_x_prob: float = 0.05
    min_abs_lin_vel_x: float = 0.08
    min_abs_ang_vel_z: float = 0.2
    abrupt_flip_prob: float = 0.25
    abrupt_flip_min_abs_lin_vel_x: float = 0.6
    abrupt_flip_max_abs_lin_vel_x: float = 1.0
    yaw_visualizer_scale: float = 3.0
    goal_yaw_marker_offset: tuple[float, float, float] = (0.35, 0.18, 0.7)
    current_yaw_marker_offset: tuple[float, float, float] = (0.35, -0.18, 0.7)
    goal_yaw_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/yaw_goal"
    )
    current_yaw_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/yaw_current"
    )

    # Half the default velocity marker dimensions at equal numeric speed.
    goal_yaw_visualizer_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)
    current_yaw_visualizer_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)


class UniformBaseHeightCommand(CommandTerm):
    """Uniformly sampled base-height command."""

    cfg: "UniformBaseHeightCommandCfg"

    def __init__(self, cfg: "UniformBaseHeightCommandCfg", env):
        super().__init__(cfg, env)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self._height_body_ids, _ = env.scene[cfg.asset_name].find_bodies(cfg.body_name)
        if len(self._height_body_ids) != 1:
            raise ValueError(f"Expected one height-tracking body: {cfg.body_name}")
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        self.metrics["height_mae_cm"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        height = self._env.scene[self.cfg.asset_name].data.body_pos_w[:, self._height_body_ids[0], 2]
        error_cm = (height - self.command[:, 0]).abs() * 100.0
        self._metric_steps += 1
        self.metrics["height_mae_cm"] += (error_cm - self.metrics["height_mae_cm"]) / self._metric_steps

    def reset(self, env_ids=None):
        extras = super().reset(env_ids)
        self._metric_steps[slice(None) if env_ids is None else env_ids] = 0
        return extras

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
    body_name: str = "chassis"
    height_range: tuple[float, float] = (0.28, 0.32)
