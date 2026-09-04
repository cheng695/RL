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


class StairStateMachine(WheelbipeStateMachineBase):
    """Near/far wheel-forward scan state machine for stair climb attempts."""

    name = "stair"

    def _cfg(self, env) -> dict:
        return getattr(env.cfg, "stair_state_machine_cfg", {})

    def _zero_state(self, env, env_ids: torch.Tensor | None = None) -> None:
        tensors = (
            "wheel_forward_stair_state",
            "wheel_forward_stair_reference_height",
            "wheel_forward_stair_height_cmd",
            "wheel_forward_stair_success_time",
            "wheel_forward_stair_state_time",
            "wheel_forward_stair_wall_reset",
            "wheel_forward_stair_failure_reset",
            "wheel_forward_stair_detect_event",
            "wheel_forward_stair_success_exit_event",
            "wheel_forward_stair_failure_exit_event",
            "wheel_forward_stair_timeout_exit_event",
            "wheel_forward_stair_fast_timer",
            "wheel_forward_stair_fast_timer_started",
            "wheel_forward_stair_fast_success_time_event",
            "wheel_forward_stair_fast_success_started_event",
            "wheel_forward_stair_prev_body_relative_height",
            "wheel_forward_stair_body_relative_height_progress",
            "wheel_forward_stair_prev_ground_z",
            "wheel_forward_stair_prev_ground_z_valid",
            "wheel_forward_stair_prev_direction_sign",
        )
        for name in tensors:
            value = getattr(env, name, None)
            if value is None:
                continue
            if env_ids is None:
                if value.dtype == torch.bool:
                    value.zero_()
                else:
                    value.zero_()
            else:
                value[env_ids] = False if value.dtype == torch.bool else 0.0

    def _clear_airborne_state(self, env, mask: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        airborne_state = getattr(env, "height_reward_airborne_state", None)
        if airborne_state is not None:
            airborne_state[mask] = False
        wheel_timer = getattr(env, "airborne_wheel_contact_exit_time", None)
        if wheel_timer is not None:
            wheel_timer[mask] = 0.0
        base_timer = getattr(env, "airborne_base_contact_exit_time", None)
        if base_timer is not None:
            base_timer[mask] = 0.0
        current_duration = getattr(env, "airborne_current_duration", None)
        if current_duration is not None:
            current_duration[mask] = 0.0

    def _get_step_detect_masks(
        self, env, height_diffs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if height_diffs.shape[1] == 0:
            empty_env = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            return empty_env.unsqueeze(-1).expand(-1, 0), empty_env
        detect_cfg = env._get_wheel_forward_scan_cfg().get("detect", {})
        step_min = float(detect_cfg.get("step_height_min", 0.15))
        step_max = float(detect_cfg.get("step_height_max", 0.25))
        per_wheel = (height_diffs > step_min) & (height_diffs < step_max)
        return per_wheel, torch.any(per_wheel, dim=1)

    def _as_name_tuple(self, names) -> tuple[str, ...]:
        if names is None:
            return ()
        if isinstance(names, str):
            return (names,) if names else ()
        return tuple(name for name in names if name)

    def _get_wheel_contact_mask(self, env) -> torch.Tensor:
        contact_forces = getattr(getattr(env.contact_sensor, "data", None), "net_forces_w_history", None)
        wheel_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        if contact_forces is None:
            return wheel_contact
        wheel_force_peaks = env._get_wheel_contact_force_peaks(contact_forces)
        if wheel_force_peaks.shape[1] == 0:
            return wheel_contact
        wheel_count = min(2, wheel_force_peaks.shape[1])
        success_cfg = self._cfg(env).get("success", {})
        threshold = float(
            success_cfg.get(
                "contact_force_threshold",
                getattr(env.cfg, "desired_contact_force_threshold", 1.0),
            )
        )
        wheel_contact[:, :wheel_count] = wheel_force_peaks[:, :wheel_count] > threshold
        return wheel_contact

    def _get_wheel_xy_contact_mask(self, env, threshold: float) -> torch.Tensor:
        contact_forces = getattr(getattr(env.contact_sensor, "data", None), "net_forces_w_history", None)
        wheel_contact = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
        if contact_forces is None or len(getattr(env, "_desired_contact_link_idx", [])) == 0:
            return wheel_contact
        wheel_xy_forces = torch.amax(
            torch.norm(contact_forces[:, :, env._desired_contact_link_idx, :2], dim=-1),
            dim=1,
        )
        if wheel_xy_forces.shape[1] == 0:
            return wheel_contact
        wheel_count = min(2, wheel_xy_forces.shape[1])
        wheel_contact[:, :wheel_count] = wheel_xy_forces[:, :wheel_count] > threshold
        return wheel_contact

    def _get_wheel_radius(self, env) -> float:
        cfg = self._cfg(env)
        if "wheel_radius" in cfg:
            return float(cfg["wheel_radius"])
        if getattr(env, "_get_height_measure_wheel_radius", None) is not None:
            return float(env._get_height_measure_wheel_radius())
        return 0.05

    def get_effective_height_cmd(self, env, height_cmd: torch.Tensor) -> torch.Tensor:
        return torch.where(env.wheel_forward_stair_state, env.wheel_forward_stair_height_cmd, height_cmd)

    def on_command_updated(self, env) -> None:
        cfg = self._cfg(env)
        if not bool(cfg.get("enabled", False)):
            self._zero_state(env)
            return

        env.wheel_forward_stair_wall_reset.zero_()
        env.wheel_forward_stair_failure_reset.zero_()
        env.wheel_forward_stair_detect_event.zero_()
        env.wheel_forward_stair_success_exit_event.zero_()
        env.wheel_forward_stair_failure_exit_event.zero_()
        env.wheel_forward_stair_timeout_exit_event.zero_()
        env.wheel_forward_stair_fast_success_time_event.zero_()
        env.wheel_forward_stair_fast_success_started_event.zero_()

        stair_diffs = env._get_wheel_forward_stair_temporal_height_diffs_raw()
        if stair_diffs.shape[1] == 0:
            return
        stair_per_wheel, stair_detect = self._get_step_detect_masks(env, stair_diffs)
        forward_diffs = env._get_wheel_forward_temporal_height_diffs_raw()
        _, forward_detect = self._get_step_detect_masks(env, forward_diffs)
        allowed_terrain_names = self._as_name_tuple(cfg.get("allowed_terrain_names", ()))
        allowed_terrain_mask = env.get_terrain_name_mask(allowed_terrain_names)
        not_allowed_terrain_names = self._as_name_tuple(cfg.get("not_allowed_terrain_names", ()))
        if len(not_allowed_terrain_names) > 0:
            allowed_terrain_mask &= ~env.get_terrain_name_mask(not_allowed_terrain_names)
        stair_detect = stair_detect & allowed_terrain_mask
        forward_detect = forward_detect & allowed_terrain_mask
        stair_per_wheel = stair_per_wheel & allowed_terrain_mask.unsqueeze(-1)

        query_points_w, hit_points_w = env._get_wheel_forward_stair_scan_points()
        if hit_points_w is None:
            return

        env.wheel_forward_stair_detect_event.copy_(stair_detect)
        wall_reset = torch.zeros_like(stair_detect)
        env.wheel_forward_stair_wall_reset.copy_(wall_reset)

        enter_mask = stair_detect & (~env.wheel_forward_stair_state) & (~wall_reset)
        if torch.any(enter_mask):
            hit_z = hit_points_w[:, :, 2]
            reference_candidates = torch.where(
                stair_per_wheel,
                hit_z,
                torch.full_like(hit_z, -torch.inf),
            )
            reference_height = torch.amax(reference_candidates, dim=1)
            reference_height = torch.where(torch.isfinite(reference_height), reference_height, hit_z[:, 0])
            target_range = cfg.get("height_cmd", {}).get("range", (0.37, 0.40))
            target_min = float(target_range[0])
            target_max = float(target_range[-1])
            target_low, target_high = min(target_min, target_max), max(target_min, target_max)
            env.wheel_forward_stair_reference_height[enter_mask] = reference_height[enter_mask]
            env.wheel_forward_stair_height_cmd[enter_mask] = (
                torch.rand(
                    int(torch.count_nonzero(enter_mask).item()),
                    dtype=torch.float,
                    device=env.device,
                )
                * (target_high - target_low)
                + target_low
            )
            env.wheel_forward_stair_success_time[enter_mask] = 0.0

        active = (env.wheel_forward_stair_state | enter_mask) & (~wall_reset)
        prev_active = env.wheel_forward_stair_state & (~wall_reset)
        env.wheel_forward_stair_state_time.copy_(
            torch.where(
                active,
                env.wheel_forward_stair_state_time + env.step_dt,
                torch.zeros_like(env.wheel_forward_stair_state_time),
            )
        )
        wheel_contact = self._get_wheel_contact_mask(env)
        both_wheel_contact = torch.all(wheel_contact, dim=1)
        fast_cfg = cfg.get("fast_success", {})
        xy_contact_threshold = float(
            fast_cfg.get(
                "xy_contact_force_threshold",
                cfg.get("success", {}).get(
                    "contact_force_threshold",
                    getattr(env.cfg, "desired_contact_force_threshold", 1.0),
                ),
            )
        )
        wheel_xy_contact = self._get_wheel_xy_contact_mask(env, xy_contact_threshold)
        fast_timer_started = active & (
            env.wheel_forward_stair_fast_timer_started | torch.any(wheel_xy_contact, dim=1)
        )
        env.wheel_forward_stair_fast_timer_started.copy_(fast_timer_started)
        env.wheel_forward_stair_fast_timer.copy_(
            torch.where(
                fast_timer_started,
                env.wheel_forward_stair_fast_timer + env.step_dt,
                torch.zeros_like(env.wheel_forward_stair_fast_timer),
            )
        )
        body_relative_step_height = (
            env.robot.data.root_pos_w[:, 2] - env.wheel_forward_stair_reference_height
        )
        env.wheel_forward_stair_body_relative_height_progress.copy_(
            torch.where(
                prev_active,
                torch.clamp(
                    body_relative_step_height - env.wheel_forward_stair_prev_body_relative_height,
                    min=0.0,
                ),
                torch.zeros_like(body_relative_step_height),
            )
        )
        env.wheel_forward_stair_prev_body_relative_height.copy_(
            torch.where(
                active,
                body_relative_step_height,
                torch.zeros_like(env.wheel_forward_stair_prev_body_relative_height),
            )
        )
        success_cfg = cfg.get("success", {})
        success_error = max(float(success_cfg.get("height_error", 0.05)), 0.0)
        success_cond = (
            active
            & both_wheel_contact
            & (body_relative_step_height >= env.wheel_forward_stair_height_cmd - success_error)
        )
        env.wheel_forward_stair_success_time.copy_(
            torch.where(
                success_cond,
                env.wheel_forward_stair_success_time + env.step_dt,
                torch.zeros_like(env.wheel_forward_stair_success_time),
            )
        )
        success_duration = max(
            float(success_cfg.get("duration_s", 0.3)),
            0.0,
        )
        success_exit = active & (env.wheel_forward_stair_success_time >= success_duration)

        failure_cfg = cfg.get("failure", {})
        fail_drop = max(
            float(failure_cfg.get("drop_threshold", 0.10)),
            0.0,
        )
        stair_hit_z = hit_points_w[:, :, 2]
        failure_exit = active & torch.all(
            stair_hit_z < (env.wheel_forward_stair_reference_height.unsqueeze(-1) - fail_drop),
            dim=1,
        )
        timeout_cfg = cfg.get("timeout", {})
        timeout_duration_s = max(float(timeout_cfg.get("duration_s", 5.0)), 0.0)
        timeout_exit = (
            active
            & (timeout_duration_s > 0.0)
            & (env.wheel_forward_stair_state_time >= timeout_duration_s)
        )

        failure_exit = failure_exit | timeout_exit
        next_state = active & (~success_exit) & (~failure_exit)
        env.wheel_forward_stair_success_exit_event.copy_(success_exit)
        env.wheel_forward_stair_failure_exit_event.copy_(failure_exit)
        env.wheel_forward_stair_timeout_exit_event.copy_(timeout_exit)
        env.wheel_forward_stair_fast_success_started_event.copy_(
            success_exit & env.wheel_forward_stair_fast_timer_started
        )
        env.wheel_forward_stair_fast_success_time_event.copy_(
            torch.where(
                success_exit,
                env.wheel_forward_stair_fast_timer,
                torch.zeros_like(env.wheel_forward_stair_fast_timer),
            )
        )
        env.wheel_forward_stair_failure_reset.copy_(failure_exit)
        env.wheel_forward_stair_state.copy_(next_state)
        self._clear_airborne_state(env, next_state)

        clear_mask = wall_reset | success_exit | failure_exit | timeout_exit
        if torch.any(clear_mask):
            env.wheel_forward_stair_success_time[clear_mask] = 0.0
            env.wheel_forward_stair_state_time[clear_mask] = 0.0
            env.wheel_forward_stair_height_cmd[clear_mask] = 0.0
            env.wheel_forward_stair_reference_height[clear_mask] = 0.0
            env.wheel_forward_stair_fast_timer[clear_mask] = 0.0
            env.wheel_forward_stair_fast_timer_started[clear_mask] = False
            env.wheel_forward_stair_prev_body_relative_height[clear_mask] = 0.0
            env.wheel_forward_stair_body_relative_height_progress[clear_mask] = 0.0

    def apply_reward_term_scales(
        self, env, reward_terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        stair_mask_f = env.wheel_forward_stair_state.float()
        reward_scales = self._cfg(env).get("reward_scales", {})
        for term_name, boost_scale in reward_scales.items():
            if term_name not in reward_terms:
                continue
            scale = 1.0 + stair_mask_f * (float(boost_scale) - 1.0)
            reward_terms[term_name] = reward_terms[term_name] * scale

        reward_additions = self._cfg(env).get("reward_additions", {})
        for term_name, addition_cfg in reward_additions.items():
            if isinstance(addition_cfg, dict):
                addition_type = addition_cfg.get("type", "success_event")
                if addition_type == "success_event":
                    added_reward = env.wheel_forward_stair_success_exit_event.float()
                elif addition_type == "failure_event":
                    added_reward = env.wheel_forward_stair_failure_exit_event.float()
                elif addition_type == "fast_success":
                    time_sigma = max(float(addition_cfg.get("time_sigma", 1.0)), 1.0e-6)
                    max_reward = float(addition_cfg.get("max_reward", 1.0))
                    success_duration = max(
                        float(self._cfg(env).get("success", {}).get("duration_s", 0.0)),
                        0.0,
                    )
                    effective_success_time = torch.clamp(
                        env.wheel_forward_stair_fast_success_time_event - success_duration,
                        min=0.0,
                    )
                    added_reward = (
                        env.wheel_forward_stair_success_exit_event.float()
                        * env.wheel_forward_stair_fast_success_started_event.float()
                        * max_reward
                        * torch.exp(-effective_success_time / time_sigma)
                    )
                elif addition_type == "height_track_exp":
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    height_error = (
                        env.robot.data.root_pos_w[:, 2]
                        - env.wheel_forward_stair_reference_height
                        - env.wheel_forward_stair_height_cmd
                    )
                    if bool(addition_cfg.get("only_below_target", False)):
                        height_error = torch.clamp(height_error, max=0.0)
                    added_reward = env.wheel_forward_stair_state.float() * torch.exp(
                        -torch.square(height_error) / sigma
                    )
                elif addition_type == "height_progress":
                    max_progress = max(float(addition_cfg.get("max_progress", 0.05)), 1.0e-6)
                    progress = torch.clamp(
                        env.wheel_forward_stair_body_relative_height_progress,
                        min=0.0,
                        max=max_progress,
                    )
                    added_reward = env.wheel_forward_stair_state.float() * (progress / max_progress)
                elif addition_type == "wheel_above_reference_height":
                    margin = float(addition_cfg.get("margin", 0.0))
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    wheel_z = env.robot.data.body_pos_w[:, env._wheel_link_idx, 2]
                    if bool(addition_cfg.get("use_wheel_bottom", False)):
                        wheel_z = wheel_z - self._get_wheel_radius(env)
                    wheel_height_error = wheel_z - env.wheel_forward_stair_reference_height.unsqueeze(-1) - margin
                    if bool(addition_cfg.get("exp", True)):
                        per_wheel_reward = torch.exp(-torch.square(torch.clamp(wheel_height_error, max=0.0)) / sigma)
                    else:
                        max_error = max(float(addition_cfg.get("max_error", 0.2)), 1.0e-6)
                        per_wheel_reward = 1.0 - torch.clamp(-wheel_height_error, min=0.0, max=max_error) / max_error
                    above_bonus_scale = float(addition_cfg.get("above_bonus_scale", 0.0))
                    if above_bonus_scale != 0.0:
                        above_bonus_margin = max(float(addition_cfg.get("above_bonus_margin", 0.05)), 1.0e-6)
                        above_bonus_max = max(float(addition_cfg.get("above_bonus_max", 1.0)), 0.0)
                        above_bonus = torch.clamp(
                            torch.clamp(wheel_height_error, min=0.0) / above_bonus_margin,
                            max=above_bonus_max,
                        )
                        per_wheel_reward = per_wheel_reward + above_bonus_scale * above_bonus
                    added_reward = env.wheel_forward_stair_state.float() * per_wheel_reward.mean(dim=1)
                elif addition_type == "wheel_xy_contact_force":
                    net_contact_forces = getattr(
                        getattr(getattr(env, "contact_sensor", None), "data", None),
                        "net_forces_w_history",
                        None,
                    )
                    if net_contact_forces is None or len(getattr(env, "_desired_contact_link_idx", [])) == 0:
                        added_reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
                    else:
                        threshold = float(addition_cfg.get("force_threshold", 20.0))
                        mode = addition_cfg.get("mode", "l2")
                        square_sigma = float(addition_cfg.get("square_sigma", 1.0))
                        wheel_xy_force = torch.amax(
                            torch.norm(
                                net_contact_forces[:, :, env._desired_contact_link_idx, :2],
                                dim=-1,
                            ),
                            dim=1,
                        )
                        force_over = torch.clamp(wheel_xy_force - threshold, min=0.0)
                        if mode == "l1":
                            penalty = torch.sum(force_over, dim=1)
                        elif mode == "l2":
                            penalty = torch.sum(torch.square(force_over * square_sigma), dim=1)
                        else:
                            raise ValueError(f"Unsupported wheel_xy_contact_force reward mode: {mode}")
                        added_reward = env.wheel_forward_stair_state.float() * penalty
                else:
                    raise ValueError(f"Unsupported stair reward addition type: {addition_type}")
            else:
                added_reward = env.wheel_forward_stair_state.float() * float(addition_cfg)

            if term_name in reward_terms:
                reward_terms[term_name] = reward_terms[term_name] + added_reward
            else:
                reward_terms[term_name] = added_reward
        return reward_terms

    def apply_done_masks(
        self,
        env,
        terminate: torch.Tensor,
        time_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = time_out | env.wheel_forward_stair_wall_reset
        terminate = terminate & (~env.wheel_forward_stair_wall_reset)
        if bool(self._cfg(env).get("failure", {}).get("terminate", True)):
            terminate = terminate | env.wheel_forward_stair_failure_reset
        else:
            time_out = time_out | env.wheel_forward_stair_failure_reset
            terminate = terminate & (~env.wheel_forward_stair_failure_reset)
        return terminate, time_out

    def append_reset_logs(
        self, env, extras: dict[str, float], env_ids: torch.Tensor
    ) -> None:
        extras["Episode/WheelForwardStair/StateRatio"] = (
            env.wheel_forward_stair_state[env_ids].float().mean().item()
        )
        extras["Episode/WheelForwardStair/WallResetRatio"] = (
            env.wheel_forward_stair_wall_reset[env_ids].float().mean().item()
        )
        extras["Episode/WheelForwardStair/FailureResetRatio"] = (
            env.wheel_forward_stair_failure_reset[env_ids].float().mean().item()
        )
        extras["Episode/WheelForwardStair/TimeoutExitRatio"] = (
            env.wheel_forward_stair_timeout_exit_event[env_ids].float().mean().item()
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

        stair_state_index = state_name_to_index.get("stair")
        if stair_state_index is not None:
            stair_priority = 25
            stair_mask = env.wheel_forward_stair_state & (stair_priority >= resolved_priorities)
            if torch.any(stair_mask):
                resolved_indices = resolved_indices.clone()
                resolved_priorities = resolved_priorities.clone()
                resolved_indices[stair_mask] = stair_state_index
                resolved_priorities[stair_mask] = stair_priority

        wall_blocked_state_index = state_name_to_index.get("wall_blocked")
        if wall_blocked_state_index is not None:
            wall_priority = 30
            wall_mask = env.wheel_forward_stair_wall_reset & (wall_priority >= resolved_priorities)
            if torch.any(wall_mask):
                if resolved_indices.data_ptr() == marker_indices.data_ptr():
                    resolved_indices = resolved_indices.clone()
                if resolved_priorities.data_ptr() == priorities.data_ptr():
                    resolved_priorities = resolved_priorities.clone()
                resolved_indices[wall_mask] = wall_blocked_state_index
                resolved_priorities[wall_mask] = wall_priority

        return resolved_indices, resolved_priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        self._zero_state(env, env_ids)
