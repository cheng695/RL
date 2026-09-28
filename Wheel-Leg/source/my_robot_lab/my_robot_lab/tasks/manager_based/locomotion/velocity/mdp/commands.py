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
        self.command_age = torch.zeros(self.num_envs, device=self.device)
        for name in (
            "vx_mae_mps", "yaw_mae_radps", "action_at_limit_fraction",
            "leg_action_at_limit_fraction", "wheel_action_at_limit_fraction",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._mode_error_sums = {
            mode: torch.zeros(self.num_envs, device=self.device)
            for mode in (
                "straight", "turn", "mixed", "straight_forward", "straight_reverse",
                "straight_low_speed", "straight_high_speed",
            )
        }
        self._mode_steps = {mode: torch.zeros_like(value) for mode, value in self._mode_error_sums.items()}
        # Standing-only count, signed body vx sum, horizontal speed sum, path integral.
        self._standing_sums = torch.zeros(self.num_envs, 4, device=self.device)

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
        # ActionsCfg orders four leg position actions before two wheel velocities.
        at_limit = (self._env.action_manager.action.abs() >= self.cfg.action_limit - 1.0e-6).float()
        for name, value in (
            ("leg_action_at_limit_fraction", at_limit[:, :4].mean(dim=-1)),
            ("wheel_action_at_limit_fraction", at_limit[:, 4:6].mean(dim=-1)),
        ):
            self.metrics[name] += (value - self.metrics[name]) / self._metric_steps

        moving = self.command[:, 0].abs() > 1.0e-6
        turning = self.command[:, 2].abs() > 1.0e-6
        standing = (self.command[:, :3].abs() <= 1.0e-6).all(dim=-1)
        velocity = self.robot.data.root_lin_vel_b[:, :2]
        speed = torch.linalg.vector_norm(velocity, dim=-1)
        samples = torch.stack((torch.ones_like(speed), velocity[:, 0], speed,
                               speed * self._env.step_dt), dim=-1)
        self._standing_sums += torch.where(standing[:, None], samples, 0.0)
        for mode, mask in (
            ("straight", moving & ~turning),
            ("turn", ~moving & turning),
            ("mixed", moving & turning),
            ("straight_forward", (self.command[:, 0] > 1.0e-6) & ~turning),
            ("straight_reverse", (self.command[:, 0] < -1.0e-6) & ~turning),
            ("straight_low_speed", moving & ~turning & (self.command[:, 0].abs() <= 0.5)),
            ("straight_high_speed", moving & ~turning & (self.command[:, 0].abs() > 0.5)),
        ):
            self._mode_error_sums[mode] += torch.where(mask, values["vx_mae_mps"], 0.0)
            self._mode_steps[mode] += mask.float()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        # Pool only samples of this mode; episodes without it must not dilute MAE.
        mode_metrics = {}
        standing = self._standing_sums[ids].sum(dim=0)
        mode_metrics["stand_sample_count"] = standing[0].item()
        mode_metrics["stand_vx_bias_mps"] = (standing[1] / standing[0].clamp_min(1)).item()
        mode_metrics["stand_speed_xy_mps"] = (standing[2] / standing[0].clamp_min(1)).item()
        # Mean per episode with standing samples, not net displacement from a fixed anchor.
        episodes = (self._standing_sums[ids, 0] > 0).sum().clamp_min(1)
        mode_metrics["stand_path_length_m"] = (standing[3] / episodes).item()
        for mode in self._mode_steps:
            count = self._mode_steps[mode][ids].sum()
            mode_metrics[f"vx_mae_{mode}_mps"] = (
                self._mode_error_sums[mode][ids].sum() / count.clamp_min(1.0)
            ).item()
            # A zero count means the corresponding MAE is unavailable, not perfect.
            mode_metrics[f"{mode}_sample_count"] = count.item()
        extras = super().reset(env_ids)
        extras.update(mode_metrics)
        self._metric_steps[ids] = 0
        self._standing_sums[ids] = 0
        for mode in self._mode_steps:
            self._mode_steps[mode][ids] = 0
            self._mode_error_sums[mode][ids] = 0
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
        self.command_age[env_ids] = 0.0
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
        mixed_ids = env_ids[mixed_envs]
        if self.cfg.mixed_yaw_limit is not None and len(mixed_ids) > 0:
            self.vel_command_b[mixed_ids, 2] = self._sample_ang_vel_z(
                len(mixed_ids), self.cfg.mixed_yaw_limit)

        # Project the requested body twist onto the differential-drive wheel-speed
        # workspace.  Both vx and yaw use the same factor, so the requested motion
        # direction is preserved instead of clipping the two components separately.
        self._project_to_wheel_speed_limit(env_ids)

        old_vx = getattr(self, "_previous_lin_vel_x_command", None)
        if old_vx is None or old_vx.shape[0] != self.num_envs:
            old_vx = torch.zeros(self.num_envs, device=self.device)
            setattr(self, "_previous_lin_vel_x_command", old_vx)
        old_vx[env_ids] = self.vel_command_b[env_ids, 0]

    def _project_to_wheel_speed_limit(self, env_ids: torch.Tensor):
        wheel_half_track = float(self.cfg.wheel_half_track)
        wheel_radius = float(self.cfg.wheel_radius)
        max_wheel_speed = float(self.cfg.max_wheel_speed)
        command = self.vel_command_b[env_ids]
        left = (command[:, 0] - wheel_half_track * command[:, 2]) / wheel_radius
        right = (command[:, 0] + wheel_half_track * command[:, 2]) / wheel_radius
        peak_speed = torch.maximum(left.abs(), right.abs())
        scale = torch.clamp(max_wheel_speed / peak_speed.clamp_min(1.0e-6), max=1.0)
        self.vel_command_b[env_ids] = command * scale[:, None]

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

    def _update_command(self):
        super()._update_command()
        self.command_age += self._env.step_dt

    def _sample_ang_vel_z(self, num_samples: int, limit: float | None = None) -> torch.Tensor:
        z_min, z_max = self.cfg.ranges.ang_vel_z
        if limit is not None:
            z_min, z_max = max(z_min, -limit), min(z_max, limit)
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
    mixed_yaw_limit: float | None = None
    wheel_radius: float = 0.055
    wheel_half_track: float = 0.210335
    max_wheel_speed: float = 20.0
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
    """Mix uniform heights with opposite-endpoint transitions."""

    cfg: "UniformBaseHeightCommandCfg"

    def __init__(self, cfg: "UniformBaseHeightCommandCfg", env):
        super().__init__(cfg, env)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.command_age = torch.zeros(self.num_envs, device=self.device)
        self._height_body_ids, _ = env.scene[cfg.asset_name].find_bodies(cfg.body_name)
        if len(self._height_body_ids) != 1:
            raise ValueError(f"Expected one height-tracking body: {cfg.body_name}")
        self._metric_steps = torch.zeros(self.num_envs, device=self.device)
        for name in ("height_mae_cm", "height_bias_cm", "height_actual_cm", "height_target_cm"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)
        self._height_bins = {
            name: torch.zeros(self.num_envs, 4, device=self.device)
            for name in ("low", "mid", "high")
        }  # columns: count, actual, target, absolute error (cm)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def _update_metrics(self):
        height = self._env.scene[self.cfg.asset_name].data.body_pos_w[:, self._height_body_ids[0], 2]
        height = height - self.reference_ground_height()
        error_cm = (height - self.command[:, 0]) * 100.0
        self._metric_steps += 1
        values = {
            "height_mae_cm": error_cm.abs(),
            "height_bias_cm": error_cm,
            "height_actual_cm": height * 100.0,
            "height_target_cm": self.command[:, 0] * 100.0,
        }
        for name, value in values.items():
            self.metrics[name] += (value - self.metrics[name]) / self._metric_steps

        low, high = self.cfg.height_range
        target = self.command[:, 0]
        lower, upper = low + (high - low) / 3, high - (high - low) / 3
        samples = torch.stack((torch.ones_like(height), height * 100, target * 100, error_cm.abs()), -1)
        for name, mask in (("low", target < lower), ("mid", (target >= lower) & (target <= upper)),
                           ("high", target > upper)):
            self._height_bins[name] += torch.where(mask[:, None], samples, 0.0)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        grouped = {}
        for name, samples in self._height_bins.items():
            total = samples[ids].sum(dim=0)
            grouped[f"height_{name}_sample_count"] = total[0].item()
            for index, metric in enumerate(("actual", "target", "mae"), 1):
                grouped[f"height_{name}_{metric}_cm"] = (total[index] / total[0].clamp_min(1)).item()
            samples[ids] = 0
        extras = super().reset(env_ids)
        extras.update(grouped)
        self._metric_steps[ids] = 0
        return extras

    def reference_ground_height(self):
        """Local ground median shared by height metrics, rewards and observations."""
        sensor_name = getattr(self.cfg, "ground_sensor_name", None)
        if sensor_name is None:
            return torch.zeros(self.num_envs, device=self.device)
        hits = self._env.scene[sensor_name].data.ray_hits_w[..., 2]
        valid_hits = torch.where(torch.isfinite(hits), hits, torch.nan)
        ground = torch.nanmedian(valid_hits, dim=-1).values
        return torch.where(torch.isfinite(ground), ground, self._env.scene.env_origins[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        low, high = self.cfg.height_range
        self.command_age[env_ids] = 0.0
        previous = self.height_command[env_ids, 0]
        uniform = torch.empty(len(env_ids), device=self.device).uniform_(low, high)
        # On episode reset choose either endpoint; otherwise switch across the midpoint.
        initial = self._env.episode_length_buf[env_ids] == 0
        choose_high = torch.where(initial, torch.rand(len(env_ids), device=self.device) < 0.5,
                                  previous <= (low + high) / 2)
        endpoint = torch.where(choose_high, high, low)
        use_endpoint = torch.rand(len(env_ids), device=self.device) < self.cfg.endpoint_switch_prob
        self.height_command[env_ids, 0] = torch.where(use_endpoint, endpoint, uniform)

    def _update_command(self):
        self.command_age += self._env.step_dt


@configclass
class UniformBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for a uniformly sampled base-height command."""

    class_type: type = UniformBaseHeightCommand
    asset_name: str = "robot"
    body_name: str = "chassis"
    height_range: tuple[float, float] = (0.28, 0.32)
    endpoint_switch_prob: float = 0.0
    ground_sensor_name: str | None = None
