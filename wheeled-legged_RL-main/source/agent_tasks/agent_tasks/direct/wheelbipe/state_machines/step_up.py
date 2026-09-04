# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

from __future__ import annotations

import torch

from .base import WheelbipeStateMachineBase


class StepUpStateMachine(WheelbipeStateMachineBase):
    """Wheel-forward scan state machine for step-up assists and wall resets."""

    name = "step_up"

    def _cfg(self, env) -> dict:
        return env._get_wheel_forward_scan_cfg()

    def get_effective_height_cmd(self, env, height_cmd: torch.Tensor) -> torch.Tensor:
        # 台阶事件触发后，height command 会被保持为一个抬高后的值；计时结束后恢复原始命令。
        hold_active = env.wheel_forward_height_cmd_hold_remaining_time > 0.0
        return torch.where(hold_active, env.wheel_forward_height_cmd_hold_value, height_cmd)

    def on_command_updated(self, env) -> None:
        # 没启用轮前向扫描时，清空本状态机维护的运行时状态，避免旧 episode 残留。
        cfg = self._cfg(env)
        if not bool(cfg.get("enabled", False)):
            env.wheel_forward_height_cmd_hold_remaining_time.zero_()
            env.wheel_forward_height_cmd_hold_value.zero_()
            env.wheel_forward_wall_reset.zero_()
            env.wheel_forward_step_detect_event.zero_()
            return

        # hold 是一个持续时间状态：每个 control step 递减，直到自动失效。
        env.wheel_forward_height_cmd_hold_remaining_time = torch.clamp(
            env.wheel_forward_height_cmd_hold_remaining_time - env.step_dt,
            min=0.0,
        )
        # step/wall 事件是单步检测结果，每次 command 更新前先清空再重新计算。
        env.wheel_forward_wall_reset.zero_()
        env.wheel_forward_step_detect_event.zero_()

        # 使用轮前方 raycast 的时间差分高度：当前前方地面高度 - 上一步前方地面高度。
        # 这样可以识别“前方突然抬高”的台阶/墙，而不是单纯依赖绝对高度。
        wheel_forward_height_diffs = env._get_wheel_forward_temporal_height_diffs_raw()
        if wheel_forward_height_diffs.shape[1] == 0:
            return

        # 高度差在 step_min 和 step_max 之间视为可跨越台阶；超过 wall_threshold 视为墙。
        detect_cfg = cfg.get("detect", {})
        height_cmd_cfg = cfg.get("height_cmd", {})
        step_min = float(detect_cfg.get("step_height_min", 0.15))
        step_max = float(detect_cfg.get("step_height_max", 0.25))
        wall_threshold = float(detect_cfg.get("wall_height", step_max))
        height_cmd_bias = float(height_cmd_cfg.get("bias", 0.05))
        hold_duration_s = max(float(height_cmd_cfg.get("hold_s", 2.0)), 0.0)

        # 墙优先级高于台阶：如果高度差过大，不再触发 step_up assist，而是结束该 episode。
        wall_reset_mask = torch.any(wheel_forward_height_diffs > wall_threshold, dim=1)
        step_detect_mask = torch.any(
            (wheel_forward_height_diffs > step_min)
            & (wheel_forward_height_diffs < step_max),
            dim=1,
        ) & (~wall_reset_mask)

        env.wheel_forward_wall_reset.copy_(wall_reset_mask)
        env.wheel_forward_step_detect_event.copy_(step_detect_mask)
        if torch.any(wall_reset_mask):
            # 碰到墙时取消任何已保持的抬高命令，避免下一次 reset/可视化继续显示旧状态。
            env.wheel_forward_height_cmd_hold_remaining_time[wall_reset_mask] = 0.0
            env.wheel_forward_height_cmd_hold_value[wall_reset_mask] = 0.0

        if torch.any(step_detect_mask) and hold_duration_s > 0.0:
            # 检测到台阶后，在原始 height_cmd 上加 bias，并按配置或 height_range 上限截断。
            wheel_forward_cmd_max = height_cmd_cfg.get("max", None)
            if wheel_forward_cmd_max is not None:
                max_height_target = float(wheel_forward_cmd_max)
            else:
                height_range = getattr(env.cfg, "height_range", None)
                if height_range is not None and len(height_range) > 0:
                    max_height_target = float(height_range[-1])
                else:
                    max_height_target = float("inf")
            env.wheel_forward_height_cmd_hold_value[step_detect_mask] = torch.clamp(
                env.height_cmd[step_detect_mask] + height_cmd_bias,
                max=max_height_target,
            )
            env.wheel_forward_height_cmd_hold_remaining_time[step_detect_mask] = hold_duration_s

    def apply_done_masks(
        self,
        env,
        terminate: torch.Tensor,
        time_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 墙阻挡不是策略失败，而是地形不可通过/需要换环境，因此转成 time_out 而非 terminate。
        time_out = time_out | env.wheel_forward_wall_reset
        terminate = terminate & (~env.wheel_forward_wall_reset)
        return terminate, time_out

    def append_reset_logs(
        self, env, extras: dict[str, float], env_ids: torch.Tensor
    ) -> None:
        # 记录 reset 时仍处于 height_cmd hold 的 env 比例，用于观察台阶辅助触发频率。
        extras["Episode/WheelForward/HeightCmdHoldRatio"] = (
            (env.wheel_forward_height_cmd_hold_remaining_time[env_ids] > 0.0)
            .float()
            .mean()
            .item()
        )

    def apply_visual_marker_state(
        self,
        env,
        marker_indices: torch.Tensor,
        priorities: torch.Tensor,
        state_name_to_index: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_indices = marker_indices
        resolved_priorities = priorities

        # step_up marker 优先级高于 airborne，表示正在执行台阶辅助。
        step_up_state_index = state_name_to_index.get("step_up")
        if step_up_state_index is not None:
            step_up_priority = 20
            step_up_mask = env.wheel_forward_step_detect_event & (
                step_up_priority >= resolved_priorities
            )
            if torch.any(step_up_mask):
                resolved_indices = resolved_indices.clone()
                resolved_priorities = resolved_priorities.clone()
                resolved_indices[step_up_mask] = step_up_state_index
                resolved_priorities[step_up_mask] = step_up_priority

        # wall_blocked marker 优先级最高，覆盖其他状态，提示该 env 将作为 timeout reset。
        wall_blocked_state_index = state_name_to_index.get("wall_blocked")
        if wall_blocked_state_index is not None:
            wall_priority = 30
            wall_mask = env.wheel_forward_wall_reset & (wall_priority >= resolved_priorities)
            if torch.any(wall_mask):
                if resolved_indices.data_ptr() == marker_indices.data_ptr():
                    resolved_indices = resolved_indices.clone()
                if resolved_priorities.data_ptr() == priorities.data_ptr():
                    resolved_priorities = resolved_priorities.clone()
                resolved_indices[wall_mask] = wall_blocked_state_index
                resolved_priorities[wall_mask] = wall_priority

        return resolved_indices, resolved_priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        # 只清理发生 reset 的 env，保留其他并行 env 的台阶扫描历史和 hold 状态。
        env.wheel_forward_height_cmd_hold_remaining_time[env_ids] = 0.0
        env.wheel_forward_height_cmd_hold_value[env_ids] = 0.0
        env.wheel_forward_wall_reset[env_ids] = False
        env.wheel_forward_prev_ground_z[env_ids] = 0.0
        env.wheel_forward_prev_ground_z_valid[env_ids] = False
        if hasattr(env, "wheel_forward_prev_direction_sign"):
            env.wheel_forward_prev_direction_sign[env_ids] = 0.0
        env.wheel_forward_step_detect_event[env_ids] = False
