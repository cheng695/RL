from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils


class WheelLegVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with clearer over-head goal/current velocity arrows."""

    cfg: "WheelLegVelocityCommandCfg"

    def __init__(self, cfg: "WheelLegVelocityCommandCfg", env):
        super().__init__(cfg, env)
        # 训练日志保持精简：只保留 UniformVelocityCommand 自带的 error_vel_xy/error_vel_yaw。
        # 细分 command mode / vx-only / mixed 诊断请用专项 verify/play 脚本查看，避免每轮 PPO 打印过长。

    def __str__(self) -> str:
        msg = "WheelLegVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tExclusive vx/yaw commands: {self.cfg.exclusive_linear_yaw_commands}\n"
        msg += f"\tYaw-only probability among moving envs: {self.cfg.rel_yaw_envs}\n"
        msg += f"\tForward-vx probability among vx envs: {self.cfg.rel_forward_vx_envs}"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        if self.cfg.curriculum_stage in (1, 2, 3):
            self._resample_curriculum_command(env_ids)
            return

        if not self.cfg.exclusive_linear_yaw_commands:
            super()._resample_command(env_ids)
            return

        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device)
        if len(env_ids_tensor) == 0:
            return

        num_envs = len(env_ids_tensor)
        random_values = torch.rand(num_envs, device=self.device)
        stand_mask = random_values < self.cfg.rel_standing_envs
        yaw_mask = (random_values >= self.cfg.rel_standing_envs) & (
            random_values < self.cfg.rel_standing_envs + (1.0 - self.cfg.rel_standing_envs) * self.cfg.rel_yaw_envs
        )
        vx_mask = ~(stand_mask | yaw_mask)

        self.vel_command_b[env_ids_tensor, :] = 0.0
        if torch.any(vx_mask):
            vx_env_ids = env_ids_tensor[vx_mask]
            num_vx_envs = len(vx_env_ids)
            forward_mask = torch.rand(num_vx_envs, device=self.device) < self.cfg.rel_forward_vx_envs
            backward_mask = ~forward_mask
            lin_vel_x_min, lin_vel_x_max = self.cfg.ranges.lin_vel_x
            if torch.any(forward_mask):
                forward_env_ids = vx_env_ids[forward_mask]
                forward_low = max(0.0, lin_vel_x_min)
                forward_high = max(0.0, lin_vel_x_max)
                self.vel_command_b[forward_env_ids, 0] = torch.empty(
                    len(forward_env_ids), device=self.device
                ).uniform_(forward_low, forward_high)
            if torch.any(backward_mask):
                backward_env_ids = vx_env_ids[backward_mask]
                backward_low = min(0.0, lin_vel_x_min)
                backward_high = min(0.0, lin_vel_x_max)
                self.vel_command_b[backward_env_ids, 0] = torch.empty(
                    len(backward_env_ids), device=self.device
                ).uniform_(backward_low, backward_high)
        if torch.any(yaw_mask):
            yaw_env_ids = env_ids_tensor[yaw_mask]
            self.vel_command_b[yaw_env_ids, 2] = torch.empty(len(yaw_env_ids), device=self.device).uniform_(
                *self.cfg.ranges.ang_vel_z
            )

        self.is_standing_env[env_ids_tensor] = stand_mask
        if self.cfg.heading_command:
            self.is_heading_env[env_ids_tensor] = False

    def _resample_curriculum_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device)
        if len(env_ids_tensor) == 0:
            return

        num_envs = len(env_ids_tensor)
        self.vel_command_b[env_ids_tensor, :] = 0.0
        self.is_standing_env[env_ids_tensor] = False

        if self.cfg.curriculum_stage == 1:
            self.is_standing_env[env_ids_tensor] = True
        elif self.cfg.curriculum_stage == 2:
            random_values = torch.rand(num_envs, device=self.device)
            stand_mask = random_values < self.cfg.stage2_standing_ratio
            walking_mask = ~stand_mask
            self.is_standing_env[env_ids_tensor] = stand_mask
            if torch.any(walking_mask):
                walking_env_ids = env_ids_tensor[walking_mask]
                if self.cfg.stage2_vx_abs_range is not None:
                    abs_vx = torch.empty(len(walking_env_ids), device=self.device).uniform_(
                        *self.cfg.stage2_vx_abs_range
                    )
                    signs = torch.where(
                        torch.rand(len(walking_env_ids), device=self.device) < 0.5,
                        -torch.ones_like(abs_vx),
                        torch.ones_like(abs_vx),
                    )
                    self.vel_command_b[walking_env_ids, 0] = signs * abs_vx
                else:
                    self.vel_command_b[walking_env_ids, 0] = torch.empty(
                        len(walking_env_ids), device=self.device
                    ).uniform_(*self.cfg.stage2_vx_range)
        else:
            random_values = torch.rand(num_envs, device=self.device)
            stand_mask = random_values < self.cfg.stage3_standing_ratio
            walking_mask = (random_values >= self.cfg.stage3_standing_ratio) & (
                random_values < self.cfg.stage3_standing_ratio + self.cfg.stage3_walking_ratio
            )
            turning_mask = ~(stand_mask | walking_mask)
            self.is_standing_env[env_ids_tensor] = stand_mask
            if torch.any(walking_mask):
                walking_env_ids = env_ids_tensor[walking_mask]
                self.vel_command_b[walking_env_ids, 0] = torch.empty(
                    len(walking_env_ids), device=self.device
                ).uniform_(*self.cfg.stage2_vx_range)
            if torch.any(turning_mask):
                turning_env_ids = env_ids_tensor[turning_mask]
                self.vel_command_b[turning_env_ids, 0] = torch.empty(
                    len(turning_env_ids), device=self.device
                ).uniform_(*self.cfg.stage3_vx_range)
                self.vel_command_b[turning_env_ids, 1] = torch.empty(
                    len(turning_env_ids), device=self.device
                ).uniform_(*self.cfg.stage3_vy_range)
                self.vel_command_b[turning_env_ids, 2] = torch.empty(
                    len(turning_env_ids), device=self.device
                ).uniform_(*self.cfg.stage3_yaw_rate_range)

        if self.cfg.heading_command:
            self.is_heading_env[env_ids_tensor] = False

    def _update_metrics(self):
        super()._update_metrics()

    def _masked_mean_value(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if torch.any(mask):
            return torch.mean(value[mask])
        return torch.zeros((), device=self.device)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        goal_pos_w = base_pos_w.clone()
        current_pos_w = base_pos_w.clone()
        goal_pos_w[:, 2] += self.cfg.goal_arrow_height
        current_pos_w[:, 2] += self.cfg.current_arrow_height

        goal_scale, goal_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        current_scale, current_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.goal_vel_visualizer.visualize(goal_pos_w, goal_quat, goal_scale)
        self.current_vel_visualizer.visualize(current_pos_w, current_quat, current_scale)
        self._visualize_yaw_arcs()

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

    def _visualize_yaw_arcs(self):
        if not hasattr(self, "goal_yaw_visualizer"):
            return

        goal_pos, goal_quat, goal_scale = self._resolve_yaw_rate_to_arc(
            self.command[:, 2],
            height=self.cfg.goal_yaw_arrow_height,
            radius=self.cfg.yaw_arc_radius,
            max_yaw_rate=self.cfg.yaw_arc_max_rate,
        )
        current_pos, current_quat, current_scale = self._resolve_yaw_rate_to_arc(
            self.robot.data.root_ang_vel_b[:, 2],
            height=self.cfg.current_yaw_arrow_height,
            radius=self.cfg.yaw_arc_radius * 0.78,
            max_yaw_rate=self.cfg.yaw_arc_max_rate,
        )
        self.goal_yaw_visualizer.visualize(goal_pos, goal_quat, goal_scale)
        self.current_yaw_visualizer.visualize(current_pos, current_quat, current_scale)

    def _resolve_yaw_rate_to_arc(
        self,
        yaw_rate: torch.Tensor,
        height: float,
        radius: float,
        max_yaw_rate: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_envs = yaw_rate.shape[0]
        num_segments = self.cfg.yaw_arc_segments
        device = self.device

        base_pos_w = self.robot.data.root_pos_w
        base_quat_w = self.robot.data.root_quat_w
        segment_index = torch.arange(num_segments, device=device, dtype=torch.float32)
        segment_ratio = (segment_index + 1.0) / float(num_segments)
        arc_fraction = torch.clamp(torch.abs(yaw_rate) / max_yaw_rate, 0.0, 1.0)
        visible = segment_ratio.unsqueeze(0) <= arc_fraction.unsqueeze(1)

        direction = torch.where(yaw_rate >= 0.0, 1.0, -1.0)
        arc_span = self.cfg.yaw_arc_span
        start_angle = -0.5 * arc_span
        local_angle = start_angle + segment_ratio.unsqueeze(0) * arc_span
        local_angle = local_angle * direction.unsqueeze(1)

        local_pos = torch.zeros(num_envs, num_segments, 3, device=device)
        local_pos[:, :, 0] = radius * torch.cos(local_angle)
        local_pos[:, :, 1] = radius * torch.sin(local_angle)
        local_pos[:, :, 2] = height
        base_quat_segments = base_quat_w.unsqueeze(1).expand(-1, num_segments, -1)
        world_pos = base_pos_w.unsqueeze(1) + math_utils.quat_apply(base_quat_segments, local_pos)

        tangent_yaw = local_angle + direction.unsqueeze(1) * torch.pi / 2.0
        zeros = torch.zeros_like(tangent_yaw)
        local_quat = math_utils.quat_from_euler_xyz(zeros, zeros, tangent_yaw)
        world_quat = math_utils.quat_mul(base_quat_segments, local_quat)

        default_scale = torch.tensor(self.cfg.goal_yaw_visualizer_cfg.markers["arrow"].scale, device=device)
        scale = default_scale.repeat(num_envs, num_segments, 1)
        magnitude_scale = torch.clamp(torch.abs(yaw_rate) / max_yaw_rate, 0.2, 1.0)
        scale[:, :, 0] *= magnitude_scale.unsqueeze(1)
        scale = torch.where(visible.unsqueeze(-1), scale, torch.zeros_like(scale))

        return world_pos.reshape(-1, 3), world_quat.reshape(-1, 4), scale.reshape(-1, 3)


@configclass
class WheelLegVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = WheelLegVelocityCommand

    curriculum_stage: int = 0
    stage2_standing_ratio: float = 0.2
    stage2_vx_range: tuple[float, float] = (0.0, 1.0)
    stage2_vx_abs_range: tuple[float, float] | None = None
    stage3_standing_ratio: float = 0.1
    stage3_walking_ratio: float = 0.6
    stage3_vx_range: tuple[float, float] = (-1.0, 1.0)
    stage3_vy_range: tuple[float, float] = (-0.3, 0.3)
    stage3_yaw_rate_range: tuple[float, float] = (-1.0, 1.0)
    exclusive_linear_yaw_commands: bool = True
    rel_yaw_envs: float = 0.5
    rel_forward_vx_envs: float = 0.75
    goal_arrow_height: float = 0.75
    current_arrow_height: float = 0.55
    goal_yaw_arrow_height: float = 0.95
    current_yaw_arrow_height: float = 0.82
    yaw_arc_radius: float = 0.38
    yaw_arc_span: float = 4.2
    yaw_arc_segments: int = 5
    yaw_arc_max_rate: float = 3.0
    goal_yaw_visualizer_cfg: VisualizationMarkersCfg | None = None
    current_yaw_visualizer_cfg: VisualizationMarkersCfg | None = None


class UniformRandomBaseHeightCommand(CommandTerm):
    """Randomly resampled base-link height command, matching the Fudan plane task."""

    cfg: "UniformRandomBaseHeightCommandCfg"

    def __init__(self, cfg: "UniformRandomBaseHeightCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_height_up"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_height_down"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.height_command

    def __str__(self) -> str:
        msg = "UniformRandomBaseHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.height_range}"
        return msg

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        height_error = self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2]
        step_error = torch.abs(height_error) / max_command_step
        self.metrics["error_height"] += step_error
        self.metrics["error_height_up"] += torch.where(height_error > 0.0, step_error, torch.zeros_like(step_error))
        self.metrics["error_height_down"] += torch.where(height_error < 0.0, step_error, torch.zeros_like(step_error))

    def _resample_command(self, env_ids: Sequence[int]):
        if isinstance(env_ids, slice):
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids_tensor = torch.as_tensor(env_ids, device=self.device)
        if len(env_ids_tensor) == 0:
            return
        low_height, high_height = self.cfg.height_range
        self.height_command[env_ids_tensor, 0] = (
            torch.rand(len(env_ids_tensor), device=self.device) * (high_height - low_height)
            + low_height
        )
        self.height_command[env_ids_tensor, 1] = 0.0

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError


@configclass
class UniformRandomBaseHeightCommandCfg(CommandTermCfg):
    class_type: type = UniformRandomBaseHeightCommand

    asset_name: str = "robot"
    height_range: tuple[float, float] = (0.27, 0.35)
