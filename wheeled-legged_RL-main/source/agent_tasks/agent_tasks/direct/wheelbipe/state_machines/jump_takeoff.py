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

from isaaclab.utils.math import quat_apply_inverse

from .base import WheelbipeStateMachineBase


class JumpTakeoffStateMachine(WheelbipeStateMachineBase):
    """Ballistic jump-takeoff helper.

    The machine provides jump flags, sampled peak height, phase, and reward
    additions. It intentionally leaves XY/yaw velocity commands and height_cmd
    intact so ordinary velocity/height tracking remains part of pretraining.
    """

    name = "jump_takeoff"

    PHASE_IDLE = 0
    PHASE_PUSH = 1
    PHASE_TUCK = 2

    def _cfg(self, env) -> dict:
        return getattr(env.cfg, "jump_takeoff_state_machine_cfg", {})

    def _zero_state(self, env, env_ids: torch.Tensor | None = None) -> None:
        tensor_names = (
            "jump_takeoff_request",
            "jump_takeoff_phase",
            "jump_takeoff_phase_time",
            "jump_takeoff_cooldown_time",
            "jump_takeoff_height_cmd",
            "jump_takeoff_base_height_cmd",
            "jump_takeoff_ref_vel_z",
            "jump_takeoff_ref_release_vel_z",
            "jump_takeoff_push_max_vel_z",
            "jump_takeoff_ref_phase",
            "jump_takeoff_ref_target_height",
            "jump_takeoff_ref_peak_time",
            "jump_takeoff_ref_duration",
            "jump_takeoff_prev_vel_z",
            "jump_takeoff_trigger_event",
            "jump_takeoff_exit_event",
            "jump_takeoff_push_event",
            "jump_takeoff_tuck_event",
            "jump_takeoff_assist_event",
            "jump_takeoff_episode_trigger_count",
            "jump_takeoff_episode_exit_count",
            "jump_takeoff_episode_assist_count",
            "jump_takeoff_episode_max_height",
            "jump_takeoff_episode_max_vel_z",
            "jump_takeoff_episode_target_peak_height",
            "jump_takeoff_assist_force_selected",
            "jump_takeoff_assist_force_active",
            "airborne_force_enter_request",
        )
        for name in tensor_names:
            value = getattr(env, name, None)
            if value is None:
                continue
            if env_ids is None:
                value.zero_()
            else:
                value[env_ids] = False if value.dtype == torch.bool else 0

    def _get_allowed_terrain_mask(self, env, cfg: dict) -> torch.Tensor:
        allowed_names = cfg.get("allowed_terrain_names", ())
        if isinstance(allowed_names, str):
            allowed_names = (allowed_names,) if allowed_names else ()
        allowed_mask = env.get_terrain_name_mask(tuple(allowed_names))
        not_allowed_names = cfg.get("not_allowed_terrain_names", ())
        if isinstance(not_allowed_names, str):
            not_allowed_names = (not_allowed_names,) if not_allowed_names else ()
        not_allowed_names = tuple(name for name in not_allowed_names if name)
        if len(not_allowed_names) > 0:
            allowed_mask &= ~env.get_terrain_name_mask(not_allowed_names)
        return allowed_mask

    def _get_body_relative_height(self, env) -> torch.Tensor:
        return env.robot.data.root_pos_w[:, 2] - env.ground_z_est

    def _get_mean_leg_length(self, env) -> torch.Tensor:
        _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
        return torch.norm(wheel_pos_heading_b, dim=-1).mean(dim=1)

    def _get_wheel_radius(self, env) -> float:
        return float(
            self._cfg(env).get("enter", {}).get(
                "wheel_radius",
                env._get_height_measure_wheel_radius()
                if getattr(env, "_get_height_measure_wheel_radius", None) is not None
                else 0.06,
            )
        )

    def _get_wheel_air_mask(self, env, margin: float) -> torch.Tensor:
        wheel_heights = env._get_wheel_relative_ground_heights_raw()
        wheel_radius = self._get_wheel_radius(env)
        return torch.all(wheel_heights > wheel_radius + margin, dim=1)

    def _trajectory_enabled(self, cfg: dict) -> bool:
        return bool(cfg.get("trajectory", {}).get("enabled", False))

    def _get_training_iteration(self, env, cfg: dict) -> int:
        get_iteration = getattr(env, "_get_training_iteration", None)
        if callable(get_iteration):
            return max(int(get_iteration()), 0)
        runner_iteration = max(int(getattr(env, "_training_iteration", 0)), 0)
        steps_per_iteration = int(
            cfg.get("assist", {}).get(
                "steps_per_iteration",
                getattr(env.cfg, "training_progress_steps_per_iteration", 24),
            )
        )
        if steps_per_iteration <= 0:
            return runner_iteration
        return runner_iteration + max(int(getattr(env, "common_step_counter", 0)), 0) // steps_per_iteration

    def _get_assist_probability(self, env, cfg: dict) -> float:
        assist_cfg = cfg.get("assist", {})
        decay_mode = assist_cfg.get("decay_mode", "probability")
        if decay_mode == "force":
            probability = float(assist_cfg.get("probability", assist_cfg.get("probability_start", 0.0)))
            return max(min(probability, 1.0), 0.0)
        start_iteration = int(assist_cfg.get("iteration_start", 0))
        end_iteration = int(assist_cfg.get("iteration_end", start_iteration))
        start_prob = float(assist_cfg.get("probability_start", assist_cfg.get("probability", 0.0)))
        end_prob = float(assist_cfg.get("probability_end", 0.0))
        iteration = self._get_training_iteration(env, cfg)
        if end_iteration <= start_iteration:
            return max(min(end_prob if iteration >= end_iteration else start_prob, 1.0), 0.0)
        progress = (iteration - start_iteration) / max(end_iteration - start_iteration, 1)
        progress = max(min(progress, 1.0), 0.0)
        probability = start_prob + (end_prob - start_prob) * progress
        return max(min(probability, 1.0), 0.0)

    def _get_assist_force_scale(self, env, cfg: dict) -> float:
        assist_cfg = cfg.get("assist", {})
        if assist_cfg.get("decay_mode", "probability") not in ("force", "both"):
            return 1.0

        start_iteration = int(assist_cfg.get("force_scale_iteration_start", assist_cfg.get("iteration_start", 0)))
        end_iteration = int(assist_cfg.get("force_scale_iteration_end", assist_cfg.get("iteration_end", start_iteration)))
        start_scale = float(assist_cfg.get("force_scale_start", 1.0))
        end_scale = float(assist_cfg.get("force_scale_end", 0.0))
        iteration = self._get_training_iteration(env, cfg)
        if end_iteration <= start_iteration:
            scale = end_scale if iteration >= end_iteration else start_scale
        else:
            progress = (iteration - start_iteration) / max(end_iteration - start_iteration, 1)
            progress = max(min(progress, 1.0), 0.0)
            scale = start_scale + (end_scale - start_scale) * progress
        return max(scale, 0.0)

    def _get_ballistic_push_reference(
        self, env, cfg: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        traj_cfg = cfg.get("trajectory", {})
        gravity = max(float(traj_cfg.get("gravity", 9.81)), 1.0e-6)
        push_start_height = float(traj_cfg.get("push_start_height", 0.20))
        release_height = float(traj_cfg.get("release_height", 0.38))
        peak_height = env.jump_takeoff_ref_target_height
        timing_mode = traj_cfg.get("tuck_timing_mode", "dynamic_push_time")
        if timing_mode == "fixed_tuck_time":
            push_time = torch.full_like(
                peak_height, max(float(traj_cfg.get("fixed_tuck_time_s", 0.14)), env.step_dt)
            )
            discriminant = torch.square(gravity * push_time) + 8.0 * gravity * torch.clamp(
                peak_height - push_start_height, min=0.0
            )
            release_vel_z = 0.5 * (-gravity * push_time + torch.sqrt(discriminant))
            raw_release_height = push_start_height + 0.5 * release_vel_z * push_time
            max_release_height = float(traj_cfg.get("max_release_height", release_height))
            release_height_for_flight = torch.clamp(raw_release_height, max=max_release_height)
            release_vel_z = torch.sqrt(
                torch.clamp(2.0 * gravity * (peak_height - release_height_for_flight), min=0.0)
            )
        else:
            flight_peak_time = torch.sqrt(
                torch.clamp(2.0 * (peak_height - release_height) / gravity, min=0.0)
            )
            release_vel_z = gravity * flight_peak_time
            push_distance = max(release_height - push_start_height, 0.0)
            push_time = torch.where(
                release_vel_z > 1.0e-6,
                2.0 * push_distance / release_vel_z,
                torch.zeros_like(release_vel_z),
            )
        push_accel = torch.where(
            push_time > 1.0e-6,
            release_vel_z / push_time,
            torch.zeros_like(release_vel_z),
        )
        push_elapsed = torch.clamp(env.jump_takeoff_phase_time, min=0.0)
        push_elapsed = torch.minimum(push_elapsed, torch.clamp(push_time, min=0.0))
        ref_height = push_start_height + 0.5 * push_accel * torch.square(push_elapsed)
        ref_vel_z = push_accel * push_elapsed
        return ref_height, ref_vel_z

    def _get_ballistic_reference(
        self, env, cfg: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traj_cfg = cfg.get("trajectory", {})
        gravity = max(float(traj_cfg.get("gravity", 9.81)), 1.0e-6)
        push_start_height = float(traj_cfg.get("push_start_height", 0.20))
        release_height = float(traj_cfg.get("release_height", 0.38))
        peak_height = env.jump_takeoff_ref_target_height
        timing_mode = traj_cfg.get("tuck_timing_mode", "dynamic_push_time")
        if timing_mode == "fixed_tuck_time":
            push_time = torch.full_like(
                peak_height, max(float(traj_cfg.get("fixed_tuck_time_s", 0.14)), env.step_dt)
            )
            discriminant = torch.square(gravity * push_time) + 8.0 * gravity * torch.clamp(
                peak_height - push_start_height, min=0.0
            )
            release_vel_z = 0.5 * (-gravity * push_time + torch.sqrt(discriminant))
            raw_release_height = push_start_height + 0.5 * release_vel_z * push_time
            max_release_height = float(traj_cfg.get("max_release_height", release_height))
            release_height_for_flight = torch.clamp(raw_release_height, max=max_release_height)
            release_vel_z = torch.sqrt(
                torch.clamp(2.0 * gravity * (peak_height - release_height_for_flight), min=0.0)
            )
        else:
            flight_peak_time = torch.sqrt(
                torch.clamp(2.0 * (peak_height - release_height) / gravity, min=0.0)
            )
            release_vel_z = gravity * flight_peak_time
            push_distance = max(release_height - push_start_height, 0.0)
            push_time = torch.where(
                release_vel_z > 1.0e-6,
                2.0 * push_distance / release_vel_z,
                torch.zeros_like(release_vel_z),
            )
            release_height_for_flight = torch.full_like(peak_height, release_height)

        push_accel = torch.where(
            push_time > 1.0e-6,
            release_vel_z / push_time,
            torch.zeros_like(release_vel_z),
        )
        elapsed = torch.clamp(env.jump_takeoff_phase_time, min=0.0)
        push_elapsed = torch.minimum(elapsed, torch.clamp(push_time, min=0.0))
        push_height = push_start_height + 0.5 * push_accel * torch.square(push_elapsed)
        push_vel_z = push_accel * push_elapsed
        flight_elapsed = torch.clamp(elapsed - push_time, min=0.0)
        flight_height = (
            release_height_for_flight
            + release_vel_z * flight_elapsed
            - 0.5 * gravity * torch.square(flight_elapsed)
        )
        flight_vel_z = release_vel_z - gravity * flight_elapsed
        ref_height = torch.where(elapsed <= push_time, push_height, flight_height)
        ref_vel_z = torch.where(elapsed <= push_time, push_vel_z, flight_vel_z)
        return ref_height, ref_vel_z, push_time

    def _get_reward_addition_mask(
        self,
        env,
        cfg: dict,
        addition_cfg: dict,
        default_mask: torch.Tensor,
    ) -> torch.Tensor:
        window = addition_cfg.get("window", None)
        if window is None:
            return default_mask
        phase = env.jump_takeoff_phase
        if window == "active":
            return (phase != self.PHASE_IDLE).float()
        if window == "push":
            return (phase == self.PHASE_PUSH).float()
        if window == "tuck":
            return (phase == self.PHASE_TUCK).float()
        if window == "pre_release":
            release_height = float(cfg.get("trajectory", {}).get("release_height", 0.38))
            body_height = self._get_body_relative_height(env)
            return ((phase == self.PHASE_PUSH) & (body_height < release_height)).float()
        if window == "post_release":
            release_height = float(cfg.get("trajectory", {}).get("release_height", 0.38))
            body_height = self._get_body_relative_height(env)
            return ((phase != self.PHASE_IDLE) & (body_height >= release_height)).float()
        _, _, push_time = self._get_ballistic_reference(env, cfg)
        if window == "push_accel":
            return ((phase == self.PHASE_PUSH) & (env.jump_takeoff_phase_time <= push_time)).float()
        if window == "rise_to_tuck":
            return ((phase == self.PHASE_PUSH) & (env.jump_takeoff_phase_time > push_time)).float()
        raise ValueError(f"Unsupported jump_takeoff reward addition window: {window}")

    def _ensure_force_assist_state(self, env) -> None:
        if not hasattr(env, "jump_takeoff_assist_force_selected"):
            env.jump_takeoff_assist_force_selected = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )
        if not hasattr(env, "jump_takeoff_assist_force_active"):
            env.jump_takeoff_assist_force_active = torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            )

    def _get_assist_base_body_idx(self, env) -> int:
        body_idx = getattr(env, "_jump_takeoff_assist_base_body_idx", None)
        if body_idx is None:
            body_indices, _ = env.robot.find_bodies("base_link")
            if len(body_indices) == 0:
                raise RuntimeError("jump_takeoff force assist requires a base_link body.")
            body_idx = int(body_indices[0])
            env._jump_takeoff_assist_base_body_idx = body_idx
        return body_idx

    def _clear_takeoff_force_assist(self, env, env_ids: torch.Tensor | None = None) -> None:
        if not hasattr(env, "jump_takeoff_assist_force_active"):
            return
        active_mask = env.jump_takeoff_assist_force_active
        if env_ids is not None:
            selected = torch.zeros_like(active_mask)
            selected[env_ids] = True
            active_mask = active_mask & selected
        active_ids = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_ids.numel() == 0:
            return
        body_idx = self._get_assist_base_body_idx(env)
        zeros = torch.zeros((active_ids.numel(), 1, 3), dtype=torch.float, device=env.device)
        env.robot.set_external_force_and_torque(
            zeros,
            zeros,
            env_ids=active_ids,
            body_ids=[body_idx],
        )
        env.jump_takeoff_assist_force_active[active_ids] = False

    def _sample_takeoff_force_assist(self, env, cfg: dict, enter_mask: torch.Tensor) -> None:
        assist_cfg = cfg.get("assist", {})
        if not bool(assist_cfg.get("enabled", False)):
            return
        assist_type = assist_cfg.get("type", "velocity")
        velocity_timing = assist_cfg.get("velocity_timing", "tuck_start")
        if assist_type != "force" and not (
            assist_type == "velocity" and velocity_timing == "push_start"
        ):
            return
        self._ensure_force_assist_state(env)
        env.jump_takeoff_assist_force_selected[enter_mask] = False
        if assist_type == "force" and self._get_assist_force_scale(env, cfg) <= 0.0:
            return
        probability = self._get_assist_probability(env, cfg)
        if probability <= 0.0 or not torch.any(enter_mask):
            return
        selected = enter_mask & (
            torch.rand(env.num_envs, dtype=torch.float, device=env.device) < probability
        )
        env.jump_takeoff_assist_force_selected[selected] = True
        if torch.any(selected) and hasattr(env, "jump_takeoff_episode_assist_count"):
            env.jump_takeoff_episode_assist_count[selected] += 1.0

    def _apply_takeoff_force_assist(self, env, cfg: dict, force_mask: torch.Tensor) -> None:
        if not torch.any(force_mask):
            return
        assist_cfg = cfg.get("assist", {})
        force_scale = self._get_assist_force_scale(env, cfg)
        if force_scale <= 0.0:
            self._clear_takeoff_force_assist(
                env,
                env_ids=torch.nonzero(force_mask, as_tuple=False).flatten(),
            )
            return
        base_force_z = float(assist_cfg.get("force_z", 0.0))
        missing_gain = float(assist_cfg.get("missing_velocity_force_gain", 0.0))
        missing_vel = torch.clamp(
            env.jump_takeoff_ref_release_vel_z - env.robot.data.root_lin_vel_w[:, 2],
            min=0.0,
        )
        force_z = (base_force_z + missing_gain * missing_vel) * force_scale
        max_force_z = assist_cfg.get("max_force_z", None)
        if max_force_z is not None:
            force_z = torch.clamp(force_z, max=float(max_force_z))
        min_force_z = assist_cfg.get("min_force_z", None)
        if min_force_z is not None:
            force_z = torch.clamp(force_z, min=float(min_force_z))

        env_ids = torch.nonzero(force_mask, as_tuple=False).flatten()
        body_idx = self._get_assist_base_body_idx(env)
        force_w = torch.zeros((env_ids.numel(), 3), dtype=torch.float, device=env.device)
        force_w[:, 2] = force_z[env_ids]
        body_quat_w = env.robot.data.body_quat_w[env_ids, body_idx]
        force_b = quat_apply_inverse(body_quat_w, force_w).unsqueeze(1)
        torques = torch.zeros_like(force_b)
        env.robot.set_external_force_and_torque(
            force_b,
            torques,
            env_ids=env_ids,
            body_ids=[body_idx],
        )
        env.jump_takeoff_assist_force_active[env_ids] = True
        env.jump_takeoff_assist_event[env_ids] = True

    def _apply_takeoff_assist(
        self,
        env,
        cfg: dict,
        assist_mask: torch.Tensor,
        *,
        sample_probability: bool = True,
    ) -> None:
        if not torch.any(assist_mask):
            return
        assist_cfg = cfg.get("assist", {})
        if assist_cfg.get("type", "velocity") != "velocity":
            return
        if sample_probability:
            probability = self._get_assist_probability(env, cfg)
            if probability <= 0.0:
                return
            sampled = assist_mask & (
                torch.rand(env.num_envs, dtype=torch.float, device=env.device) < probability
            )
            if not torch.any(sampled):
                return
        else:
            sampled = assist_mask

        z_velocity_cfg = assist_cfg.get("z_velocity", None)
        if z_velocity_cfg is None or z_velocity_cfg == "ref_release":
            target_z_velocity = env.jump_takeoff_ref_release_vel_z
        else:
            target_z_velocity = torch.full(
                (env.num_envs,),
                float(z_velocity_cfg),
                dtype=torch.float,
                device=env.device,
            )
        max_z_velocity = assist_cfg.get("max_z_velocity", None)
        if max_z_velocity is not None:
            target_z_velocity = torch.clamp(target_z_velocity, max=float(max_z_velocity))

        root_velocity = env.robot.data.root_vel_w[sampled].clone()
        if assist_cfg.get("mode", "set_if_lower") == "set":
            root_velocity[:, 2] = target_z_velocity[sampled]
        else:
            root_velocity[:, 2] = torch.maximum(root_velocity[:, 2], target_z_velocity[sampled])
        env.robot.write_root_velocity_to_sim(root_velocity, env_ids=sampled.nonzero(as_tuple=False).flatten())
        env.jump_takeoff_assist_event[sampled] = True
        if hasattr(env, "jump_takeoff_episode_assist_count"):
            env.jump_takeoff_episode_assist_count[sampled] += 1.0

    def _sample_peak_height_from_bins(self, env, count: int, bins_cfg) -> torch.Tensor:
        if not isinstance(bins_cfg, (tuple, list)) or len(bins_cfg) == 0:
            raise ValueError("jump_takeoff peak_height_bins.enabled=True requires non-empty bins.")

        ranges = []
        probabilities = []
        for bin_cfg in bins_cfg:
            if not isinstance(bin_cfg, dict):
                raise ValueError("Each jump_takeoff peak_height bin must be a dict.")
            range_cfg = bin_cfg.get("range", bin_cfg.get("peak_height", None))
            if not isinstance(range_cfg, (tuple, list)) or len(range_cfg) != 2:
                raise ValueError("Each jump_takeoff peak_height bin requires range=(low, high).")
            low, high = float(range_cfg[0]), float(range_cfg[1])
            if high < low:
                raise ValueError(f"Invalid jump_takeoff peak_height bin range: {(low, high)}.")
            probability = float(bin_cfg.get("prob", bin_cfg.get("probability", 0.0)))
            if probability < 0.0:
                raise ValueError("jump_takeoff peak_height bin probability must be non-negative.")
            ranges.append((low, high))
            probabilities.append(probability)

        probs = torch.tensor(probabilities, dtype=torch.float, device=env.device)
        prob_sum = torch.sum(probs)
        if prob_sum <= 0.0:
            raise ValueError("jump_takeoff peak_height bin probabilities must sum to a positive value.")
        bin_ids = torch.multinomial(probs / prob_sum, count, replacement=True)
        peak_height = torch.empty(count, dtype=torch.float, device=env.device)
        for bin_idx, (low, high) in enumerate(ranges):
            mask = bin_ids == bin_idx
            if not torch.any(mask):
                continue
            peak_height[mask] = torch.empty(
                int(mask.sum().item()), dtype=torch.float, device=env.device
            ).uniform_(low, high)
        return peak_height

    def _sample_ballistic_peak(self, env, enter_mask: torch.Tensor, cfg: dict) -> None:
        if not torch.any(enter_mask):
            return
        traj_cfg = cfg.get("trajectory", {})
        peak_cfg = traj_cfg.get("peak_height", traj_cfg.get("peak_height_range", (0.50, 0.70)))
        curriculum_cfg = traj_cfg.get("peak_height_curriculum", {})
        if bool(curriculum_cfg.get("enabled", False)):
            start_iteration = int(curriculum_cfg.get("iteration_start", 0))
            end_iteration = int(curriculum_cfg.get("iteration_end", start_iteration))
            start_range = curriculum_cfg.get("start", peak_cfg)
            end_range = curriculum_cfg.get("end", peak_cfg)
            iteration = self._get_training_iteration(env, cfg)
            if end_iteration <= start_iteration:
                ratio = 1.0 if iteration >= end_iteration else 0.0
            else:
                ratio = (iteration - start_iteration) / max(end_iteration - start_iteration, 1)
                ratio = max(min(ratio, 1.0), 0.0)

            def _range_pair(value):
                if isinstance(value, (tuple, list)):
                    return float(value[0]), float(value[1])
                scalar = float(value)
                return scalar, scalar

            start_low, start_high = _range_pair(start_range)
            end_low, end_high = _range_pair(end_range)
            peak_cfg = (
                start_low + (end_low - start_low) * ratio,
                start_high + (end_high - start_high) * ratio,
            )
        count = int(enter_mask.sum().item())
        bins_cfg = traj_cfg.get("peak_height_bins", {})
        if bool(bins_cfg.get("enabled", False)):
            peak_height = self._sample_peak_height_from_bins(env, count, bins_cfg.get("bins", ()))
        elif isinstance(peak_cfg, (tuple, list)):
            low, high = float(peak_cfg[0]), float(peak_cfg[1])
            peak_height = torch.empty(count, dtype=torch.float, device=env.device).uniform_(low, high)
        else:
            peak_height = torch.full((count,), float(peak_cfg), dtype=torch.float, device=env.device)

        push_start_height = float(traj_cfg.get("push_start_height", 0.20))
        release_height = float(traj_cfg.get("release_height", 0.38))
        gravity = max(float(traj_cfg.get("gravity", 9.81)), 1.0e-6)
        exit_scale = max(float(traj_cfg.get("exit_time_scale_after_peak", 1.3)), 1.0)
        min_duration_s = max(float(traj_cfg.get("min_duration_s", 0.20)), env.step_dt)
        timing_mode = traj_cfg.get("tuck_timing_mode", "dynamic_push_time")
        if timing_mode == "fixed_tuck_time":
            push_time = torch.full_like(
                peak_height, max(float(traj_cfg.get("fixed_tuck_time_s", 0.14)), env.step_dt)
            )
            discriminant = torch.square(gravity * push_time) + 8.0 * gravity * torch.clamp(
                peak_height - push_start_height, min=0.0
            )
            release_vel_z = 0.5 * (-gravity * push_time + torch.sqrt(discriminant))
            raw_release_height = push_start_height + 0.5 * release_vel_z * push_time
            max_release_height = float(traj_cfg.get("max_release_height", release_height))
            release_height_for_flight = torch.clamp(raw_release_height, max=max_release_height)
            release_vel_z = torch.sqrt(
                torch.clamp(2.0 * gravity * (peak_height - release_height_for_flight), min=0.0)
            )
            flight_peak_time = release_vel_z / gravity
        else:
            delta_h = torch.clamp(peak_height - release_height, min=0.0)
            flight_peak_time = torch.sqrt(2.0 * delta_h / gravity)
            release_vel_z = gravity * flight_peak_time
            push_distance = max(release_height - push_start_height, 0.0)
            push_time = torch.where(
                release_vel_z > 1.0e-6,
                2.0 * push_distance / release_vel_z,
                torch.zeros_like(release_vel_z),
            )
        peak_time = push_time + flight_peak_time
        if bool(traj_cfg.get("use_fixed_phase_duration", False)):
            fixed_duration_s = max(float(traj_cfg.get("phase_duration_s", 0.50)), env.step_dt)
            duration = torch.full_like(peak_time, fixed_duration_s)
        else:
            duration = torch.clamp(push_time + flight_peak_time * exit_scale, min=min_duration_s)

        env.jump_takeoff_ref_target_height[enter_mask] = peak_height
        env.jump_takeoff_ref_peak_time[enter_mask] = peak_time
        env.jump_takeoff_ref_duration[enter_mask] = duration

    def _update_ballistic_mode(self, env, cfg: dict, enter_mask: torch.Tensor) -> None:
        traj_cfg = cfg.get("trajectory", {})
        trigger_cfg = cfg.get("trigger", {})
        if torch.any(enter_mask):
            self._sample_ballistic_peak(env, enter_mask, cfg)
            env.jump_takeoff_phase[enter_mask] = self.PHASE_PUSH
            env.jump_takeoff_base_height_cmd[enter_mask] = env.height_cmd[enter_mask]
            env.jump_takeoff_height_cmd[enter_mask] = env.height_cmd[enter_mask]
            env.jump_takeoff_phase_time[enter_mask] = 0.0
            env.jump_takeoff_push_max_vel_z[enter_mask] = 0.0
            if not hasattr(env, "jump_takeoff_prev_vel_z"):
                env.jump_takeoff_prev_vel_z = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
            env.jump_takeoff_prev_vel_z[enter_mask] = env.robot.data.root_lin_vel_w[enter_mask, 2]
            env.jump_takeoff_trigger_event[enter_mask] = True
            env.jump_takeoff_push_event[enter_mask] = True
            self._sample_takeoff_force_assist(env, cfg, enter_mask)
            assist_cfg = cfg.get("assist", {})
            if (
                bool(assist_cfg.get("enabled", False))
                and assist_cfg.get("type", "velocity") == "velocity"
                and assist_cfg.get("velocity_timing", "tuck_start") == "push_start"
            ):
                self._apply_takeoff_assist(
                    env,
                    cfg,
                    enter_mask & env.jump_takeoff_assist_force_selected,
                    sample_probability=False,
                )
            if hasattr(env, "jump_takeoff_episode_trigger_count"):
                env.jump_takeoff_episode_trigger_count[enter_mask] += 1.0
                env.jump_takeoff_episode_target_peak_height[enter_mask] = env.jump_takeoff_ref_target_height[enter_mask]

        active = env.jump_takeoff_phase != self.PHASE_IDLE
        if not torch.any(active):
            return

        duration = torch.clamp(env.jump_takeoff_ref_duration, min=env.step_dt)
        phase = torch.clamp(env.jump_takeoff_phase_time / duration, min=0.0, max=1.0)
        env.jump_takeoff_ref_phase[active] = phase[active]
        env.jump_takeoff_height_cmd[active] = env.jump_takeoff_base_height_cmd[active]
        if hasattr(env, "jump_takeoff_episode_max_height"):
            body_height = self._get_body_relative_height(env)
            env.jump_takeoff_episode_max_height[active] = torch.maximum(
                env.jump_takeoff_episode_max_height[active],
                body_height[active],
            )
            vel_z = torch.clamp(env.robot.data.root_lin_vel_w[:, 2], min=0.0)
            env.jump_takeoff_episode_max_vel_z[active] = torch.maximum(
                env.jump_takeoff_episode_max_vel_z[active],
                vel_z[active],
            )

        gravity = max(float(traj_cfg.get("gravity", 9.81)), 1.0e-6)
        push_start_height = float(traj_cfg.get("push_start_height", 0.20))
        release_height = float(traj_cfg.get("release_height", 0.38))
        peak_height = env.jump_takeoff_ref_target_height
        timing_mode = traj_cfg.get("tuck_timing_mode", "dynamic_push_time")
        if timing_mode == "fixed_tuck_time":
            push_time = torch.full_like(
                peak_height, max(float(traj_cfg.get("fixed_tuck_time_s", 0.14)), env.step_dt)
            )
            discriminant = torch.square(gravity * push_time) + 8.0 * gravity * torch.clamp(
                peak_height - push_start_height, min=0.0
            )
            release_vel_z = 0.5 * (-gravity * push_time + torch.sqrt(discriminant))
            raw_release_height = push_start_height + 0.5 * release_vel_z * push_time
            max_release_height = float(traj_cfg.get("max_release_height", release_height))
            release_height_for_flight = torch.clamp(raw_release_height, max=max_release_height)
            release_vel_z = torch.sqrt(
                torch.clamp(2.0 * gravity * (peak_height - release_height_for_flight), min=0.0)
            )
        else:
            flight_peak_time = torch.sqrt(
                torch.clamp(2.0 * (peak_height - release_height) / gravity, min=0.0)
            )
            release_vel_z = gravity * flight_peak_time
            push_distance = max(release_height - push_start_height, 0.0)
            push_time = torch.where(
                release_vel_z > 1.0e-6,
                2.0 * push_distance / release_vel_z,
                torch.zeros_like(release_vel_z),
            )
        push_accel = torch.where(
            push_time > 1.0e-6,
            release_vel_z / push_time,
            torch.zeros_like(release_vel_z),
        )
        env.jump_takeoff_ref_release_vel_z[active] = release_vel_z[active]
        elapsed_after_release = torch.clamp(env.jump_takeoff_phase_time - push_time, min=0.0)
        env.jump_takeoff_ref_vel_z[active] = torch.where(
            env.jump_takeoff_phase_time <= push_time,
            push_accel * env.jump_takeoff_phase_time,
            release_vel_z - gravity * elapsed_after_release,
        )[active]
        push_active = active & (env.jump_takeoff_phase == self.PHASE_PUSH)
        if torch.any(push_active):
            current_vel_z = torch.clamp(env.robot.data.root_lin_vel_w[:, 2], min=0.0)
            env.jump_takeoff_push_max_vel_z[push_active] = torch.maximum(
                env.jump_takeoff_push_max_vel_z[push_active],
                current_vel_z[push_active],
            )
        assist_cfg = cfg.get("assist", {})
        if bool(assist_cfg.get("enabled", False)) and assist_cfg.get("type", "velocity") == "force":
            self._ensure_force_assist_state(env)
            force_start_time_s = max(float(assist_cfg.get("force_start_time_s", 0.0)), 0.0)
            force_mask = (
                push_active
                & env.jump_takeoff_assist_force_selected
                & (env.jump_takeoff_phase_time >= force_start_time_s)
            )
            self._apply_takeoff_force_assist(env, cfg, force_mask)

        body_height = self._get_body_relative_height(env)
        tuck_start_height = float(traj_cfg.get("tuck_start_height", release_height))
        tuck_start_height_ratio = traj_cfg.get("tuck_start_height_ratio", None)
        manual_tuck_start_time_s = traj_cfg.get("tuck_start_time_s", None)
        tuck_start_time_offset_s = max(float(traj_cfg.get("tuck_start_time_offset_s", 0.05)), 0.0)
        tuck_wheel_air_margin = max(float(traj_cfg.get("tuck_wheel_air_margin", 0.03)), 0.0)
        if timing_mode == "fixed_tuck_time":
            tuck_start_time = push_time
        else:
            tuck_start_time = push_time + tuck_start_time_offset_s
        push_mask = env.jump_takeoff_phase == self.PHASE_PUSH
        if manual_tuck_start_time_s is not None:
            tuck_start_time = torch.full_like(
                env.jump_takeoff_phase_time,
                max(float(manual_tuck_start_time_s), env.step_dt),
            )
            natural_tuck_start = torch.zeros_like(push_mask)
            time_tuck_start = push_mask & (env.jump_takeoff_phase_time >= tuck_start_time)
        elif tuck_start_height_ratio is not None:
            target_height = env.jump_takeoff_ref_target_height * max(float(tuck_start_height_ratio), 0.0)
            natural_tuck_start = push_mask & (body_height >= target_height)
            time_tuck_start = torch.zeros_like(push_mask)
        else:
            natural_tuck_start = push_mask & (
                (body_height >= tuck_start_height)
                | self._get_wheel_air_mask(env, tuck_wheel_air_margin)
            )
            time_tuck_start = push_mask & (env.jump_takeoff_phase_time >= tuck_start_time)
        if (
            bool(cfg.get("assist", {}).get("enabled", False))
            and cfg.get("assist", {}).get("velocity_timing", "tuck_start") == "tuck_start"
        ):
            self._apply_takeoff_assist(env, cfg, time_tuck_start & ~natural_tuck_start)
        tuck_start = natural_tuck_start | time_tuck_start
        if torch.any(tuck_start):
            env.jump_takeoff_phase[tuck_start] = self.PHASE_TUCK
            env.jump_takeoff_tuck_event[tuck_start] = True

        done = active & (env.jump_takeoff_phase_time >= duration)
        if torch.any(done):
            if bool(cfg.get("exit", {}).get("enter_airborne_on_exit", False)):
                force_enter_request = getattr(env, "airborne_force_enter_request", None)
                if force_enter_request is not None:
                    force_enter_request[done] = True
            if hasattr(env, "jump_takeoff_assist_force_selected"):
                env.jump_takeoff_assist_force_selected[done] = False
            env.jump_takeoff_phase[done] = self.PHASE_IDLE
            env.jump_takeoff_phase_time[done] = 0.0
            env.jump_takeoff_height_cmd[done] = 0.0
            env.jump_takeoff_cooldown_time[done] = max(float(trigger_cfg.get("cooldown_s", 1.0)), 0.0)
            env.jump_takeoff_exit_event[done] = True
            if hasattr(env, "jump_takeoff_episode_exit_count"):
                env.jump_takeoff_episode_exit_count[done] += 1.0

    def get_effective_height_cmd(self, env, height_cmd: torch.Tensor) -> torch.Tensor:
        return height_cmd

    def on_command_updated(self, env) -> None:
        cfg = self._cfg(env)
        if not bool(cfg.get("enabled", False)):
            self._clear_takeoff_force_assist(env)
            self._zero_state(env)
            return
        self._clear_takeoff_force_assist(env)

        env.jump_takeoff_trigger_event.zero_()
        env.jump_takeoff_exit_event.zero_()
        env.jump_takeoff_push_event.zero_()
        env.jump_takeoff_tuck_event.zero_()
        if hasattr(env, "jump_takeoff_assist_event"):
            env.jump_takeoff_assist_event.zero_()

        idle = env.jump_takeoff_phase == self.PHASE_IDLE
        active = ~idle
        env.jump_takeoff_phase_time.copy_(
            torch.where(
                active,
                env.jump_takeoff_phase_time + env.step_dt,
                torch.zeros_like(env.jump_takeoff_phase_time),
            )
        )
        env.jump_takeoff_cooldown_time.copy_(
            torch.clamp(env.jump_takeoff_cooldown_time - env.step_dt, min=0.0)
        )

        trigger_cfg = cfg.get("trigger", {})
        request_mask = env.jump_takeoff_request.clone()
        mode = trigger_cfg.get("mode", "flag")
        if mode == "random":
            probability = trigger_cfg.get("probability_per_step", None)
            if probability is None:
                probability = float(trigger_cfg.get("rate_per_s", 0.0)) * env.step_dt
            random_request = (
                torch.rand(env.num_envs, dtype=torch.float, device=env.device)
                < float(probability)
            )
            request_mask |= random_request
        elif mode not in ("flag", "manual"):
            raise ValueError(f"Unsupported jump_takeoff trigger mode: {mode}")

        permission_cfg = getattr(env.cfg, "jump_takeoff_permission_cfg", {}) or {}
        if bool(permission_cfg.get("enabled", False)):
            permission_mask = getattr(env, "jump_takeoff_permission_mask", None)
            if permission_mask is None:
                request_mask &= False
            else:
                request_mask &= permission_mask
        get_disabled_mask = getattr(env, "_get_jump_takeoff_disabled_mask", None)
        if callable(get_disabled_mask):
            request_mask &= ~get_disabled_mask()

        min_episode_time_s = max(float(trigger_cfg.get("min_episode_time_s", 0.0)), 0.0)
        stable_mask = (
            env.episode_length_buf.to(dtype=torch.float) * env.step_dt
            >= min_episode_time_s
        )
        if bool(trigger_cfg.get("require_tracking", False)):
            height_error_max = float(trigger_cfg.get("height_error_max", 0.05))
            lin_vel_error_max = float(trigger_cfg.get("lin_vel_error_max", 0.5))
            body_height = self._get_body_relative_height(env)
            height_ok = torch.abs(body_height - env.height_cmd) <= height_error_max
            lin_vel_error = torch.norm(
                env.robot.data.root_lin_vel_b[:, :2] - env.command[:, :2],
                dim=-1,
            )
            stable_mask &= height_ok & (lin_vel_error <= lin_vel_error_max)

        enter_mask = (
            idle
            & request_mask
            & stable_mask
            & (env.jump_takeoff_cooldown_time <= 0.0)
            & self._get_allowed_terrain_mask(env, cfg)
        )
        if self._trajectory_enabled(cfg):
            self._update_ballistic_mode(env, cfg, enter_mask)
        env.jump_takeoff_request[enter_mask] = False

    def apply_reward_term_scales(
        self, env, reward_terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        cfg = self._cfg(env)
        phase = env.jump_takeoff_phase
        active_f = (phase != self.PHASE_IDLE).float()
        push_f = (phase == self.PHASE_PUSH).float()
        tuck_f = (phase == self.PHASE_TUCK).float()
        if not hasattr(env, "jump_takeoff_prev_vel_z"):
            env.jump_takeoff_prev_vel_z = env.robot.data.root_lin_vel_w[:, 2].detach().clone()
        current_vel_z = env.robot.data.root_lin_vel_w[:, 2]

        for term_name, scale_cfg in cfg.get("reward_scales", {}).items():
            if term_name not in reward_terms:
                continue
            if isinstance(scale_cfg, dict):
                active_scale = float(scale_cfg.get("active", 1.0))
                push_scale = float(scale_cfg.get("push", active_scale))
                tuck_scale = float(scale_cfg.get("tuck", active_scale))
            else:
                active_scale = push_scale = tuck_scale = float(scale_cfg)
            scale = torch.ones(env.num_envs, dtype=torch.float, device=env.device)
            scale = torch.where(active_f.bool(), torch.full_like(scale, active_scale), scale)
            scale = torch.where(push_f.bool(), torch.full_like(scale, push_scale), scale)
            scale = torch.where(tuck_f.bool(), torch.full_like(scale, tuck_scale), scale)
            reward_terms[term_name] = reward_terms[term_name] * scale

        addition_items = []
        for term_name, addition_cfg in cfg.get("reward_additions", {}).items():
            addition_items.append((term_name, addition_cfg, active_f))
        for phase_name, phase_mask in (("push", push_f), ("tuck", tuck_f)):
            phase_cfg = cfg.get(phase_name, {})
            for term_name, addition_cfg in phase_cfg.get("reward_additions", {}).items():
                addition_items.append((term_name, addition_cfg, phase_mask))

        for term_name, addition_cfg, default_mask in addition_items:
            if not isinstance(addition_cfg, dict):
                added_reward = default_mask * float(addition_cfg)
            else:
                default_mask = self._get_reward_addition_mask(env, cfg, addition_cfg, default_mask)
                addition_type = addition_cfg.get("type", "constant")
                if addition_type == "constant":
                    added_reward = default_mask * float(addition_cfg.get("value", 0.0))
                elif addition_type == "takeoff_vel_z":
                    vel_z = torch.clamp(current_vel_z, min=0.0)
                    max_vel = addition_cfg.get("max", None)
                    if max_vel is not None:
                        vel_z = torch.clamp(vel_z, max=float(max_vel))
                    added_reward = default_mask * vel_z
                elif addition_type == "upward_accel_z":
                    accel_z = torch.clamp(
                        (current_vel_z - env.jump_takeoff_prev_vel_z) / max(env.step_dt, 1.0e-6),
                        min=0.0,
                    )
                    max_accel = addition_cfg.get("max", None)
                    if max_accel is not None:
                        accel_z = torch.clamp(accel_z, max=float(max_accel))
                    added_reward = default_mask * accel_z
                elif addition_type == "push_max_vel_z":
                    max_vel_z = env.jump_takeoff_push_max_vel_z
                    max_vel = addition_cfg.get("max", None)
                    if max_vel is not None:
                        max_vel_z = torch.clamp(max_vel_z, max=float(max_vel))
                    added_reward = default_mask * max_vel_z
                elif addition_type == "push_release_vel_z_threshold":
                    margin = float(addition_cfg.get("margin", 0.0))
                    max_bonus = addition_cfg.get("max", None)
                    exceed = torch.clamp(
                        env.jump_takeoff_push_max_vel_z - env.jump_takeoff_ref_release_vel_z - margin,
                        min=0.0,
                    )
                    if bool(addition_cfg.get("normalized", False)):
                        exceed = exceed / torch.clamp(env.jump_takeoff_ref_release_vel_z, min=1.0e-6)
                    if max_bonus is not None:
                        exceed = torch.clamp(exceed, max=float(max_bonus))
                    event = addition_cfg.get("event", "tuck")
                    if event == "tuck":
                        reward_mask = env.jump_takeoff_tuck_event.float()
                    elif event == "window":
                        reward_mask = default_mask
                    else:
                        raise ValueError(f"Unsupported push_release_vel_z_threshold event: {event}")
                    added_reward = reward_mask * exceed
                elif addition_type == "release_vel_z_shortfall":
                    offset = max(float(addition_cfg.get("offset", 0.0)), 0.0)
                    max_penalty = addition_cfg.get("max", None)
                    measured_vel_z = env.robot.data.root_lin_vel_w[:, 2]
                    if addition_cfg.get("measure", "push_max") == "push_max":
                        measured_vel_z = env.jump_takeoff_push_max_vel_z
                    shortfall = torch.clamp(
                        env.jump_takeoff_ref_release_vel_z - measured_vel_z - offset,
                        min=0.0,
                    )
                    if bool(addition_cfg.get("normalized", False)):
                        shortfall = shortfall / torch.clamp(env.jump_takeoff_ref_release_vel_z, min=1.0e-6)
                    if max_penalty is not None:
                        shortfall = torch.clamp(shortfall, max=float(max_penalty))
                    added_reward = default_mask * shortfall
                elif addition_type == "release_vel_z_tracking_l1":
                    offset = max(float(addition_cfg.get("offset", 0.0)), 0.0)
                    error = torch.abs(env.robot.data.root_lin_vel_w[:, 2] - env.jump_takeoff_ref_release_vel_z)
                    added_reward = default_mask * torch.clamp(error - offset, min=0.0)
                elif addition_type == "release_vel_z_tracking_exp":
                    sigma = max(float(addition_cfg.get("sigma", 1.0)), 1.0e-6)
                    error = env.robot.data.root_lin_vel_w[:, 2] - env.jump_takeoff_ref_release_vel_z
                    added_reward = default_mask * torch.exp(-torch.square(error) / sigma)
                elif addition_type == "push_height_tracking_exp":
                    ref_height, _ = self._get_ballistic_push_reference(env, cfg)
                    body_height = self._get_body_relative_height(env)
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    added_reward = default_mask * torch.exp(
                        -torch.square(body_height - ref_height) / sigma
                    )
                elif addition_type == "push_vel_z_tracking_exp":
                    _, ref_vel_z = self._get_ballistic_push_reference(env, cfg)
                    sigma = max(float(addition_cfg.get("sigma", 1.0)), 1.0e-6)
                    error = env.robot.data.root_lin_vel_w[:, 2] - ref_vel_z
                    added_reward = default_mask * torch.exp(-torch.square(error) / sigma)
                elif addition_type == "peak_height_tracking_exp":
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    error = env.jump_takeoff_episode_max_height - env.jump_takeoff_ref_target_height
                    reward_mask = default_mask
                    event = addition_cfg.get("event", "exit")
                    if event == "exit":
                        reward_mask = env.jump_takeoff_exit_event.float()
                    elif event == "active":
                        reward_mask = default_mask
                    else:
                        raise ValueError(f"Unsupported peak_height_tracking_exp event: {event}")
                    added_reward = reward_mask * torch.exp(-torch.square(error) / sigma)
                elif addition_type == "height_tracking_exp":
                    body_height = self._get_body_relative_height(env)
                    target_height = env.jump_takeoff_height_cmd
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    added_reward = default_mask * torch.exp(
                        -torch.square(body_height - target_height) / sigma
                    )
                elif addition_type == "trajectory_height_tracking_exp":
                    body_height = self._get_body_relative_height(env)
                    if bool(addition_cfg.get("use_ballistic_reference", False)):
                        target_height, _, _ = self._get_ballistic_reference(env, cfg)
                    else:
                        target_height = env.jump_takeoff_height_cmd
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    added_reward = default_mask * torch.exp(
                        -torch.square(body_height - target_height) / sigma
                    )
                elif addition_type == "trajectory_vel_z_tracking_l1":
                    offset = max(float(addition_cfg.get("offset", 0.0)), 0.0)
                    error = torch.abs(env.robot.data.root_lin_vel_w[:, 2] - env.jump_takeoff_ref_vel_z)
                    added_reward = default_mask * torch.clamp(error - offset, min=0.0)
                elif addition_type == "trajectory_vel_z_tracking_exp":
                    sigma = max(float(addition_cfg.get("sigma", 1.0)), 1.0e-6)
                    if bool(addition_cfg.get("use_ballistic_reference", False)):
                        _, target_vel_z, _ = self._get_ballistic_reference(env, cfg)
                    else:
                        target_vel_z = env.jump_takeoff_ref_vel_z
                    error = env.robot.data.root_lin_vel_w[:, 2] - target_vel_z
                    added_reward = default_mask * torch.exp(-torch.square(error) / sigma)
                elif addition_type == "leg_retraction":
                    _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
                    leg_lengths = torch.norm(wheel_pos_heading_b, dim=-1)
                    mean_leg_length = leg_lengths.mean(dim=1)
                    target = float(addition_cfg.get("target", 0.22))
                    mode = addition_cfg.get("mode", "above_target")
                    if mode == "raw":
                        added_reward = default_mask * mean_leg_length
                    elif mode == "above_target":
                        added_reward = default_mask * torch.clamp(mean_leg_length - target, min=0.0)
                    elif mode == "above_target_per_leg":
                        added_reward = default_mask * torch.clamp(leg_lengths - target, min=0.0).mean(dim=1)
                    elif mode == "exp":
                        sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                        added_reward = default_mask * torch.exp(
                            -torch.square(mean_leg_length - target) / sigma
                        )
                    elif mode == "exp_per_leg":
                        sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                        added_reward = default_mask * torch.exp(
                            -torch.square(leg_lengths - target) / sigma
                        ).mean(dim=1)
                    else:
                        raise ValueError(f"Unsupported leg_retraction reward mode: {mode}")
                elif addition_type == "leg_length_tracking_exp":
                    _, _, wheel_pos_heading_b, _, _ = env._get_root_quat_inv_and_wheel_pos_b()
                    leg_lengths = torch.norm(wheel_pos_heading_b, dim=-1)
                    target = float(addition_cfg.get("target", 0.16))
                    sigma = max(float(addition_cfg.get("sigma", 0.005)), 1.0e-6)
                    per_leg = bool(addition_cfg.get("per_leg", True))
                    if per_leg:
                        error = leg_lengths - target
                        added_reward = default_mask * torch.exp(-torch.square(error) / sigma).mean(dim=1)
                    else:
                        mean_leg_length = leg_lengths.mean(dim=1)
                        added_reward = default_mask * torch.exp(
                            -torch.square(mean_leg_length - target) / sigma
                        )
                elif addition_type == "wheel_airtime":
                    margin = max(float(addition_cfg.get("wheel_air_margin", 0.03)), 0.0)
                    added_reward = default_mask * self._get_wheel_air_mask(env, margin).float()
                elif addition_type == "wheel_height_below_base_exp":
                    target = float(addition_cfg.get("target", 0.22))
                    sigma = max(float(addition_cfg.get("sigma", 0.01)), 1.0e-6)
                    wheel_bottom_z = (
                        env.robot.data.body_pos_w[:, env._wheel_link_idx, 2]
                        - self._get_wheel_radius(env)
                    )
                    rel_wheel_bottom_z = wheel_bottom_z - env.robot.data.root_pos_w[:, 2].unsqueeze(-1)
                    target_rel_wheel_bottom_z = -target
                    added_reward = default_mask * torch.exp(
                        -torch.square(rel_wheel_bottom_z - target_rel_wheel_bottom_z) / sigma
                    ).mean(dim=1)
                else:
                    raise ValueError(f"Unsupported jump_takeoff reward addition type: {addition_type}")

            if term_name in reward_terms:
                reward_terms[term_name] = reward_terms[term_name] + added_reward
            else:
                reward_terms[term_name] = added_reward
        env.jump_takeoff_prev_vel_z.copy_(current_vel_z.detach())
        return reward_terms

    def append_reset_logs(
        self, env, extras: dict[str, float], env_ids: torch.Tensor
    ) -> None:
        phase = env.jump_takeoff_phase[env_ids]
        extras["Episode/JumpTakeoff/ActiveRatio"] = (
            (phase != self.PHASE_IDLE).float().mean().item()
        )
        extras["Episode/JumpTakeoff/PushRatio"] = (
            (phase == self.PHASE_PUSH).float().mean().item()
        )
        extras["Episode/JumpTakeoff/TuckRatio"] = (
            (phase == self.PHASE_TUCK).float().mean().item()
        )
        if hasattr(env, "jump_takeoff_assist_event"):
            extras["Episode/JumpTakeoff/AssistRatio"] = (
                env.jump_takeoff_assist_event[env_ids].float().mean().item()
            )
        if hasattr(env, "jump_takeoff_episode_trigger_count"):
            trigger_count = env.jump_takeoff_episode_trigger_count[env_ids]
            exit_count = env.jump_takeoff_episode_exit_count[env_ids]
            assist_count = env.jump_takeoff_episode_assist_count[env_ids]
            max_height = env.jump_takeoff_episode_max_height[env_ids]
            max_vel_z = env.jump_takeoff_episode_max_vel_z[env_ids]
            target_peak = env.jump_takeoff_episode_target_peak_height[env_ids]
            triggered = trigger_count > 0.0
            extras["Episode/JumpTakeoff/TriggeredRatio"] = triggered.float().mean().item()
            extras["Episode/JumpTakeoff/ExitRatio"] = (exit_count > 0.0).float().mean().item()
            extras["Episode/JumpTakeoff/AssistEpisodeRatio"] = (assist_count > 0.0).float().mean().item()
            extras["Episode/JumpTakeoff/TriggerCountMean"] = trigger_count.mean().item()
            extras["Episode/JumpTakeoff/AssistCountMean"] = assist_count.mean().item()
            if torch.any(triggered):
                triggered_height = max_height[triggered]
                triggered_vel_z = max_vel_z[triggered]
                triggered_target = target_peak[triggered]
                extras["Episode/JumpTakeoff/MaxHeightMean"] = triggered_height.mean().item()
                extras["Episode/JumpTakeoff/MaxHeightMax"] = triggered_height.max().item()
                extras["Episode/JumpTakeoff/MaxVelZMean"] = triggered_vel_z.mean().item()
                extras["Episode/JumpTakeoff/TargetPeakHeightMean"] = triggered_target.mean().item()
                extras["Episode/JumpTakeoff/PeakHeightErrorMean"] = (
                    triggered_height - triggered_target
                ).mean().item()
            else:
                extras["Episode/JumpTakeoff/MaxHeightMean"] = 0.0
                extras["Episode/JumpTakeoff/MaxHeightMax"] = 0.0
                extras["Episode/JumpTakeoff/MaxVelZMean"] = 0.0
                extras["Episode/JumpTakeoff/TargetPeakHeightMean"] = 0.0
                extras["Episode/JumpTakeoff/PeakHeightErrorMean"] = 0.0

    def apply_visual_marker_state(
        self,
        env,
        marker_indices: torch.Tensor,
        priorities: torch.Tensor,
        state_name_to_index: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        resolved_indices = marker_indices
        resolved_priorities = priorities
        phase_to_marker = (
            (self.PHASE_PUSH, "jump_push", 28),
            (self.PHASE_TUCK, "jump_tuck", 28),
        )
        for phase_id, marker_name, priority in phase_to_marker:
            marker_index = state_name_to_index.get(marker_name)
            if marker_index is None:
                continue
            mask = (env.jump_takeoff_phase == phase_id) & (priority >= resolved_priorities)
            if torch.any(mask):
                if resolved_indices.data_ptr() == marker_indices.data_ptr():
                    resolved_indices = resolved_indices.clone()
                if resolved_priorities.data_ptr() == priorities.data_ptr():
                    resolved_priorities = resolved_priorities.clone()
                resolved_indices[mask] = marker_index
                resolved_priorities[mask] = priority
        return resolved_indices, resolved_priorities

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        self._clear_takeoff_force_assist(env, env_ids)
        self._zero_state(env, env_ids)
