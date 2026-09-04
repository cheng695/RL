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

import atexit
import csv
import os
from pathlib import Path
from datetime import datetime

import torch
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

from agent_tasks.direct.wheelbipe.wheelbipe_V13.env import WheelbipeV13Env
from agent_tasks.direct.wheelbipe.wheelbipe_V14.cfg_utils import V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG
from agent_tasks.direct.wheelbipe.wheelbipe_V14.env_cfg import WheelbipeV14FlatEnvCfg
from scripts.utils.velocity_trace_html import build_reward_signs, build_velocity_trace_html


def get_rough_height_offset_curriculum_cfg(cfg) -> dict:
    """Return normalized rough height-offset curriculum settings."""
    raw_cfg = getattr(cfg, "rough_height_offset_curriculum_cfg", None)
    normalized = dict(V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG)
    if isinstance(raw_cfg, dict):
        normalized.update(raw_cfg)
    return normalized


def get_rough_terrain_boundary_reset_cfg(cfg) -> dict:
    raw_cfg = getattr(cfg, "rough_terrain_boundary_reset_cfg", None)
    normalized = {
        "enabled": False,
        "margin": 0.5,
        "use_inner_terrain_area": True,
    }
    if isinstance(raw_cfg, dict):
        normalized.update(raw_cfg)
    return normalized


def get_training_progress_steps_per_iteration(cfg) -> int:
    """Resolve PPO steps-per-iteration used to extrapolate training progress from env steps."""
    curriculum_cfg = get_rough_height_offset_curriculum_cfg(cfg)
    for attr in ("training_progress_steps_per_iteration",):
        steps = int(getattr(cfg, attr, 0))
        if steps > 0:
            return steps
    return int(curriculum_cfg["steps_per_iteration"])


def get_extrapolated_training_iteration(env) -> int:
    """Extrapolate current training iteration from runner anchor + ``common_step_counter``."""
    runner_iteration = int(getattr(env, "_training_iteration", 0))
    steps_per_iteration = get_training_progress_steps_per_iteration(getattr(env, "cfg", env))
    step_iteration = int(getattr(env, "_training_iteration_base", runner_iteration)) + (
        max(
            int(getattr(env, "common_step_counter", 0))
            - int(getattr(env, "_training_progress_step_base", 0)),
            0,
        )
        // steps_per_iteration
    )
    return max(runner_iteration, step_iteration, 0)


class WheelbipeV14Env(WheelbipeV13Env):
    cfg: WheelbipeV14FlatEnvCfg

    def __init__(self, cfg: WheelbipeV14FlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._training_iteration = 0
        self._training_iteration_base = 0
        self._training_progress_step_base = 0
        self._mean_total_reward: float | None = None
        self._rough_height_offset_curriculum_last_level = -1
        self._rough_height_offset_curriculum_last_iteration = -1
        self._apply_rough_height_offset_curriculum(self.robot._ALL_INDICES, force=True)

        self._wheel_link_idx, _ = self.robot.find_bodies(".*_wheel_link")
        self._gimbal_yaw_link_idx, _ = self.robot.find_bodies("gimbal_yaw_link")
        self._guide_link_idx = []
        self._use_gimbal = self._is_gimbal_enabled()
        if self._use_gimbal:
            self._gimbal_yaw_idx, self._gimbal_yaw_joint_names = self.robot.find_joints(
                self.cfg.gimbal_yaw_name
            )
            self._gimbal_pitch_idx, self._gimbal_pitch_joint_names = self.robot.find_joints(
                self.cfg.gimbal_pitch_name
            )
        else:
            self._gimbal_yaw_idx, self._gimbal_yaw_joint_names = [], []
            self._gimbal_pitch_idx, self._gimbal_pitch_joint_names = [], []
        self._gimbal_idx = list(self._gimbal_yaw_idx) + list(self._gimbal_pitch_idx)
        self._ordered_leg_joint_idx = self._resolve_names_to_indices(
            self.cfg.ordered_leg_joint_names,
            self.robot.joint_names,
            kind="joint",
        )
        self._ordered_leg_body_idx = self._resolve_names_to_indices(
            self.cfg.ordered_leg_body_names,
            self.robot.body_names,
            kind="body",
        )
        self.reorder_reset_joint_idx = list(self._ordered_leg_joint_idx)
        self._undesired_contact_link_idx = self._find_contact_sensor_indices(
            [
                "base_link",
                ".*_rear1_link",
                ".*_rear2_link",
                ".*_front1_link",
                ".*_front2_link",
                ".*_front3_link",
                ".*_front4_link",
                "gimbal_yaw_link",
                "gimbal_pitch_link",
                ".*_guide_link",
            ]
        )
        self._desired_contact_link_idx = self._find_contact_sensor_indices([".*_wheel_link"])
        
        reset_contact_body_names = [
                                    "base_link", 
                                    "gimbal_yaw_link", 
                                    "gimbal_pitch_link",
                                    ".*_guide_link",
                                    ]
        terrain_cfg = getattr(self.cfg, "terrain", None)
        if self._use_gimbal and getattr(terrain_cfg, "terrain_type", None) != "plane":
            reset_contact_body_names = ["gimbal_yaw_link", "gimbal_pitch_link"]
        self._reset_contact_link_idx = self._find_contact_sensor_indices(
            reset_contact_body_names
        )
        self._gimbal_pitch_target = torch.full(
            (self.num_envs, len(self._gimbal_pitch_idx)),
            fill_value=float(getattr(self.cfg, "gimbal_pitch_target_pos", 0.0)),
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_yaw_velocity_target = torch.zeros(
            (self.num_envs, len(self._gimbal_yaw_idx)),
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_heading_target_w = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_heading_target_initialized = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._ensure_gimbal_heading_pd_gain_tensors()
        self._gimbal_spin_translate_lin_vel_yaw = torch.zeros(
            self.num_envs,
            2,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_height_cmd = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_sin_heading = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_cos_heading = torch.ones(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
        )
        self._gimbal_spin_translate_active = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._gimbal_spin_translate_last_command_counter = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._gimbal_spin_translate_marker: VisualizationMarkers | None = None
        self._create_gimbal_spin_translate_marker()
        self._gimbal_yaw_actuator_default_stiffness = None
        self._gimbal_yaw_actuator_default_damping = None
        self._capture_gimbal_yaw_actuator_gains()
        self._validate_v14_bookkeeping()
        self._reset_gimbal_joints(self.robot._ALL_INDICES)
        self._velocity_trace_initialized = False
        self._velocity_trace_selected_env_id: int | None = None
        self._velocity_trace_last_sample_time = -1.0e9
        self._velocity_trace_last_html_time = -1.0e9
        self._velocity_trace_file = None
        self._velocity_trace_writer = None
        self._velocity_trace_rows: list[dict[str, float | int | str]] = []
        if self._is_velocity_trace_enabled():
            atexit.register(self._close_velocity_trace)

    def _is_gimbal_enabled(self) -> bool:
        use_gimbal = getattr(self.cfg, "use_gimbal", None)
        if use_gimbal is not None:
            return bool(use_gimbal)
        return bool(getattr(self.cfg, "gimbal_yaw_name", None)) or bool(
            getattr(self.cfg, "gimbal_pitch_name", None)
        )

    def set_training_progress(
        self,
        iteration: int | None = None,
        mean_total_reward: float | None = None,
    ) -> None:
        """Receive optional runner-side progress for curriculum scheduling."""

        if iteration is not None:
            self._training_iteration = int(iteration)
            self._training_iteration_base = int(iteration)
            self._training_progress_step_base = int(getattr(self, "common_step_counter", 0))
        if mean_total_reward is not None:
            self._mean_total_reward = float(mean_total_reward)
        self._sync_command_generator_training_iteration()

    def _get_training_iteration(self) -> int:
        return get_extrapolated_training_iteration(self)

    def _sync_command_generator_training_iteration(self) -> None:
        """Push extrapolated training iteration to special-mode command generators."""
        command_gen = getattr(self, "command_generator", None)
        if command_gen is not None and hasattr(command_gen, "set_training_iteration"):
            command_gen.set_training_iteration(self._get_training_iteration())

    def _rough_height_offset_curriculum_enabled(self) -> bool:
        return bool(get_rough_height_offset_curriculum_cfg(self.cfg)["enabled"])

    def _get_rough_height_offset_curriculum_iteration(self) -> int:
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        steps_per_iteration = int(curriculum_cfg["steps_per_iteration"])
        if steps_per_iteration <= 0:
            return max(int(getattr(self, "_training_iteration", 0)), 0)
        return self._get_training_iteration()

    def _get_rough_height_offset_curriculum_level(self) -> tuple[int, float, int]:
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        num_levels = max(int(curriculum_cfg["num_levels"]), 1)
        if num_levels <= 1:
            return 0, 1.0, self._get_rough_height_offset_curriculum_iteration()

        iteration = self._get_rough_height_offset_curriculum_iteration()
        max_iteration = int(curriculum_cfg["max_iteration"])
        interval = int(curriculum_cfg["interval"])
        capped_iteration = min(iteration, max(max_iteration, 0))
        level = num_levels - 1 if interval <= 0 else capped_iteration // max(interval, 1)
        level = int(max(0, min(level, num_levels - 1)))
        scale = float(level) / float(num_levels - 1)
        return level, scale, iteration

    def _apply_rough_height_offset_curriculum(
        self,
        env_ids: torch.Tensor | None,
        *,
        force: bool = False,
    ) -> None:
        if not self._rough_height_offset_curriculum_enabled():
            return

        terrain = getattr(self, "terrain", None)
        terrain_origins = getattr(terrain, "terrain_origins", None)
        terrain_levels = getattr(terrain, "terrain_levels", None)
        terrain_types = getattr(terrain, "terrain_types", None)
        env_origins = getattr(terrain, "env_origins", None)
        if terrain_origins is None or terrain_levels is None or terrain_types is None or env_origins is None:
            return

        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return

        level, scale, iteration = self._get_rough_height_offset_curriculum_level()
        max_level = int(terrain_origins.shape[0]) - 1
        num_types = int(terrain_origins.shape[1])
        level = min(level, max_level)
        if level < 0 or num_types <= 0:
            return
        curriculum_cfg = get_rough_height_offset_curriculum_cfg(self.cfg)
        num_levels = max(int(curriculum_cfg["num_levels"]), 1)
        terminal_level = min(num_levels - 1, max_level)
        random_up_to_current = bool(curriculum_cfg["random_reset_up_to_current_level"])
        random_after_max = bool(curriculum_cfg["random_reset_after_max"])
        random_reset = random_up_to_current or (random_after_max and level >= terminal_level)
        random_max_level = level if random_up_to_current else max_level

        current_levels = terrain_levels[env_ids_t]
        if not random_reset and not force and current_levels.numel() > 0 and torch.all(current_levels == level):
            return

        if random_reset:
            reset_levels = torch.randint(
                0,
                random_max_level + 1,
                (env_ids_t.numel(),),
                device=env_ids_t.device,
                dtype=terrain_levels.dtype,
            )
            randomize_type = bool(curriculum_cfg["randomize_type_on_random_reset"])
            if randomize_type:
                terrain_types[env_ids_t] = torch.randint(
                    0,
                    num_types,
                    (env_ids_t.numel(),),
                    device=env_ids_t.device,
                    dtype=terrain_types.dtype,
                )
        else:
            reset_levels = torch.full(
                (env_ids_t.numel(),), level, device=env_ids_t.device, dtype=terrain_levels.dtype
            )

        terrain_levels[env_ids_t] = reset_levels
        terrain.env_origins[env_ids_t] = terrain_origins[reset_levels.long(), terrain_types[env_ids_t].long()]
        if level != self._rough_height_offset_curriculum_last_level:
            print(
                "[TerrainCurriculum] V14 rough height_offset_range "
                f"scale={scale:.2f}, level={level}, iteration={iteration}, "
                f"random_reset={random_reset}, random_max_level={random_max_level}"
            )
        self._rough_height_offset_curriculum_last_level = level
        self._rough_height_offset_curriculum_last_iteration = iteration

    def _append_rough_height_offset_curriculum_log(self) -> None:
        if not self._rough_height_offset_curriculum_enabled():
            return
        level, scale, iteration = self._get_rough_height_offset_curriculum_level()
        self.extras.setdefault("log", {})
        self.extras["log"]["TerrainCurriculum/height_offset_scale"] = scale
        self.extras["log"]["TerrainCurriculum/height_offset_level"] = float(level)
        self.extras["log"]["TerrainCurriculum/iteration"] = float(iteration)

    def _get_velocity_trace_cfg(self) -> dict:
        cfg = getattr(self.cfg, "velocity_trace_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_velocity_trace_enabled(self) -> bool:
        cfg = self._get_velocity_trace_cfg()
        return bool(cfg.get("enabled", False))

    def _ensure_velocity_trace(self) -> None:
        if self._velocity_trace_initialized:
            return
        self._velocity_trace_initialized = True

        cfg = self._get_velocity_trace_cfg()
        csv_path = Path(str(cfg.get("csv_path", "logs/debug/velocity_trace.csv")))
        html_path = Path(str(cfg.get("html_path", csv_path.with_suffix(".html"))))
        if bool(cfg.get("unique_path", False)):
            suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
            csv_path = csv_path.with_name(f"{csv_path.stem}_{suffix}{csv_path.suffix}")
            html_path = html_path.with_name(f"{html_path.stem}_{suffix}{html_path.suffix}")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        self._velocity_trace_csv_path = csv_path
        self._velocity_trace_html_path = html_path
        self._velocity_trace_max_rows = max(int(cfg.get("max_rows", 20000)), 1)
        self._velocity_trace_reward_keys = tuple(getattr(self.cfg, "rewards", {}).keys())

        self._velocity_trace_file = csv_path.open("w", newline="")
        fieldnames = [
            "sim_time_s",
            "episode_time_s",
            "env_id",
            "terrain",
            "cmd_x",
            "cmd_y",
            "cmd_yaw",
            "vel_x_b",
            "vel_y_b",
            "yaw_rate_b",
            "height_cmd",
            "height_obs",
            "height_relative",
            "height_reward_ref",
            "airborne",
        ]
        fieldnames += ["reward_total"] + [f"reward_{key}" for key in self._velocity_trace_reward_keys]
        self._velocity_trace_writer = csv.DictWriter(self._velocity_trace_file, fieldnames=fieldnames)
        self._velocity_trace_writer.writeheader()
        self._velocity_trace_file.flush()
        print(f"[VelocityTrace] CSV: {csv_path}")
        print(f"[VelocityTrace] HTML: {html_path}")

    def _close_velocity_trace(self) -> None:
        if getattr(self, "_velocity_trace_file", None) is not None:
            try:
                self._write_velocity_trace_html()
                self._velocity_trace_file.flush()
                self._velocity_trace_file.close()
            except Exception:
                pass
            self._velocity_trace_file = None

    def _get_velocity_trace_terrain_name(self, env_id: int) -> str:
        manager = self._get_terrain_task_manager()
        if manager is not None and manager.enabled:
            masks = manager.get_task_masks(self.robot.data.root_pos_w)
            for name, mask in masks.items():
                if bool(mask[env_id].item()):
                    return str(name)

        terrain_command_manager = getattr(self, "_terrain_command_manager", None)
        if terrain_command_manager is not None and getattr(terrain_command_manager, "enabled", False):
            key_indices = terrain_command_manager.get_current_terrain_key_indices()
            key_idx = int(key_indices[env_id].item())
            terrain_keys = getattr(terrain_command_manager, "terrain_keys", ())
            if 0 <= key_idx < len(terrain_keys):
                return str(terrain_keys[key_idx])
        return "unknown"

    def _select_velocity_trace_env(self) -> int | None:
        cfg = self._get_velocity_trace_cfg()
        requested_env = cfg.get("env_id", cfg.get("agent_index", None))
        if requested_env is not None:
            env_id = int(requested_env)
            if 0 <= env_id < self.num_envs:
                return env_id
            return None

        selected = self._velocity_trace_selected_env_id
        lock_agent = bool(cfg.get("lock_agent", True))
        if selected is not None and lock_agent and 0 <= selected < self.num_envs:
            return selected

        terrain_name = cfg.get("terrain_name", None)
        if terrain_name:
            mask = self.get_terrain_name_mask(str(terrain_name))
            candidates = mask.nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                return selected if selected is not None and lock_agent else None
            selected = int(candidates[0].item())
        else:
            selected = 0

        self._velocity_trace_selected_env_id = selected
        print(f"[VelocityTrace] selected env_id={selected}, terrain={self._get_velocity_trace_terrain_name(selected)}")
        return selected

    def _record_velocity_trace(self) -> None:
        if not self._is_velocity_trace_enabled():
            return
        self._ensure_velocity_trace()
        cfg = self._get_velocity_trace_cfg()

        sim_time = float(getattr(self, "common_step_counter", 0)) * float(self.step_dt)
        sample_dt = max(float(cfg.get("sample_dt", self.step_dt)), float(self.step_dt))
        if sim_time - self._velocity_trace_last_sample_time + 1.0e-9 < sample_dt:
            return

        env_id = self._select_velocity_trace_env()
        if env_id is None:
            return

        airborne_state = getattr(self, "height_reward_airborne_state", None)
        effective_height_cmd = self._get_effective_height_cmd()
        obs_height = self._get_observed_height()
        if self._use_absolute_height() or self._use_leg_length_height():
            relative_obs_height = obs_height
        else:
            relative_obs_height = obs_height - self.ground_z_est
        wheel_height_w = self.robot.data.body_pos_w[:, self._wheel_link_idx, 2]
        height_reward_ref = self._get_height_reward_reference_height(relative_obs_height, wheel_height_w)
        reward_terms = getattr(self, "_last_reward_terms", {}) or {}
        total_reward = getattr(self, "_last_total_reward", None)
        row = {
            "sim_time_s": sim_time,
            "episode_time_s": float(self.episode_length_buf[env_id].item()) * float(self.step_dt),
            "env_id": int(env_id),
            "terrain": self._get_velocity_trace_terrain_name(env_id),
            "cmd_x": float(self.command[env_id, 0].item()),
            "cmd_y": float(self.command[env_id, 1].item()),
            "cmd_yaw": float(self.command[env_id, 2].item()),
            "vel_x_b": float(self.robot.data.root_lin_vel_b[env_id, 0].item()),
            "vel_y_b": float(self.robot.data.root_lin_vel_b[env_id, 1].item()),
            "yaw_rate_b": float(self.robot.data.root_ang_vel_b[env_id, 2].item()),
            "height_cmd": float(effective_height_cmd[env_id].item()) if hasattr(self, "height_cmd") else 0.0,
            "height_obs": float(obs_height[env_id].item()),
            "height_relative": float(relative_obs_height[env_id].item()),
            "height_reward_ref": float(height_reward_ref[env_id].item()),
            "airborne": int(bool(airborne_state[env_id].item())) if airborne_state is not None else 0,
            "reward_total": float(total_reward[env_id].item()) if total_reward is not None else 0.0,
        }
        for key in getattr(self, "_velocity_trace_reward_keys", ()):
            value = reward_terms.get(key, None)
            row[f"reward_{key}"] = float(value[env_id].item()) if value is not None else 0.0
        self._velocity_trace_rows.append(row)
        if len(self._velocity_trace_rows) > self._velocity_trace_max_rows:
            self._velocity_trace_rows = self._velocity_trace_rows[-self._velocity_trace_max_rows :]
        if self._velocity_trace_writer is not None:
            self._velocity_trace_writer.writerow(row)
            self._velocity_trace_file.flush()
        self._velocity_trace_last_sample_time = sim_time

        html_update_dt = float(cfg.get("html_update_interval_s", 1.0))
        if sim_time - self._velocity_trace_last_html_time + 1.0e-9 >= html_update_dt:
            self._write_velocity_trace_html()
            self._velocity_trace_last_html_time = sim_time

    def _write_velocity_trace_html(self) -> None:
        html_path = getattr(self, "_velocity_trace_html_path", None)
        if html_path is None:
            return
        reward_signs = build_reward_signs(getattr(self.cfg, "rewards", {}), rows=self._velocity_trace_rows)
        html = build_velocity_trace_html(self._velocity_trace_rows, reward_signs)
        html_path.write_text(html)

    def _get_observations(self) -> dict:
        observations = super()._get_observations()
        self._record_velocity_trace()
        return observations

    def _reset_idx(self, env_ids):
        self._apply_rough_height_offset_curriculum(env_ids)
        super()._reset_idx(env_ids)
        self._reset_gimbal_joints(env_ids)
        self._append_rough_height_offset_curriculum_log()

    def _get_rough_terrain_boundary_time_out(self) -> torch.Tensor:
        cfg = get_rough_terrain_boundary_reset_cfg(self.cfg)
        if not bool(cfg["enabled"]):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        terrain_cfg = getattr(self.cfg, "terrain", None)
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
        if terrain_gen is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        size = getattr(terrain_gen, "size", None)
        if size is None or len(size) < 2:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        half_x = 0.5 * float(getattr(terrain_gen, "num_rows", 0)) * float(size[0])
        half_y = 0.5 * float(getattr(terrain_gen, "num_cols", 0)) * float(size[1])
        if not bool(cfg["use_inner_terrain_area"]):
            border_width = float(getattr(terrain_gen, "border_width", 0.0))
            half_x += border_width
            half_y += border_width

        margin = max(float(cfg["margin"]), 0.0)
        half_x = max(half_x - margin, 0.0)
        half_y = max(half_y - margin, 0.0)
        root_pos_w = self.robot.data.root_pos_w
        return (torch.abs(root_pos_w[:, 0]) > half_x) | (torch.abs(root_pos_w[:, 1]) > half_y)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminate, time_out = super()._get_dones()
        # Boundary resets are treated as time-outs so they do not contribute to termination reward.
        time_out |= self._get_rough_terrain_boundary_time_out()
        return terminate, time_out

    def _resolve_names_to_indices(
        self,
        ordered_names: tuple[str, ...] | list[str],
        available_names: list[str],
        *,
        kind: str,
    ) -> list[int]:
        name_to_idx = {name: idx for idx, name in enumerate(available_names)}
        missing_names = [name for name in ordered_names if name not in name_to_idx]
        if missing_names:
            raise RuntimeError(
                f"Wheelbipe V14 {kind} discovery failed, missing names: {missing_names}"
            )
        return [name_to_idx[name] for name in ordered_names]

    def _validate_v14_bookkeeping(self) -> None:
        expected_joint_name = "left_front2_joint"
        expected_body_name = "left_front2_link"
        if expected_joint_name not in self.robot.joint_names:
            raise RuntimeError(f"Wheelbipe V14 is missing required joint: {expected_joint_name}")
        if expected_body_name not in self.robot.body_names:
            raise RuntimeError(f"Wheelbipe V14 is missing required body: {expected_body_name}")

        spring_count = len(self._spring_idx) if self._spring_idx is not None else 0
        checks = {
            "legs_act": (len(self._legs_act_idx), 4),
            "wheel": (len(self._wheel_idx), 2),
            "spring2": (spring_count, 2),
            "ordered_leg_joint_idx": (len(self._ordered_leg_joint_idx), 12),
            "ordered_leg_body_idx": (len(self._ordered_leg_body_idx), 12),
        }
        if self._use_gimbal:
            checks["gimbal_yaw"] = (len(self._gimbal_yaw_idx), 1)
            checks["gimbal_pitch"] = (len(self._gimbal_pitch_idx), 1)
        bad_checks = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in checks.items()
            if actual != expected
        ]
        if bad_checks:
            raise RuntimeError(
                "Wheelbipe V14 asset discovery mismatch: " + ", ".join(bad_checks)
            )

    def _sample_gimbal_yaw_velocity_targets(self, env_ids: torch.Tensor) -> None:
        if len(self._gimbal_yaw_idx) == 0 or env_ids.numel() == 0:
            return
        low, high = getattr(self.cfg, "gimbal_yaw_velocity_range", (-1.0, 1.0))
        if high < low:
            low, high = high, low
        if abs(high - low) < 1.0e-6:
            self._gimbal_yaw_velocity_target[env_ids].fill_(float(low))
            return
        sampled = torch.empty(
            (env_ids.numel(), len(self._gimbal_yaw_idx)),
            dtype=torch.float,
            device=self.device,
        ).uniform_(float(low), float(high))
        self._gimbal_yaw_velocity_target[env_ids] = sampled

    def _get_gimbal_heading_control_cfg(self) -> dict:
        cfg = getattr(self.cfg, "gimbal_heading_control_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_gimbal_heading_control_enabled(self) -> bool:
        cfg = self._get_gimbal_heading_control_cfg()
        return bool(cfg.get("enabled", False)) and len(self._gimbal_yaw_idx) > 0

    def _get_gimbal_spin_translate_cfg(self) -> dict:
        cfg = getattr(self.cfg, "gimbal_spin_translate_cfg", {}) or {}
        return dict(cfg) if isinstance(cfg, dict) else {}

    def _is_gimbal_spin_translate_enabled(self) -> bool:
        cfg = self._get_gimbal_spin_translate_cfg()
        if not bool(cfg.get("enabled", False)):
            return False
        if bool(cfg.get("require_heading_control", True)) and not self._is_gimbal_heading_control_enabled():
            return False
        return len(self._gimbal_yaw_idx) > 0

    def _is_gimbal_spin_translate_marker_enabled(self) -> bool:
        return bool(getattr(self.cfg, "play", False)) and bool(
            getattr(self.cfg, "play_gimbal_spin_translate_debug_vis", False)
        )

    def _create_gimbal_spin_translate_marker(self) -> None:
        self._gimbal_spin_translate_marker = None
        if not self._is_gimbal_spin_translate_marker_enabled():
            return
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/gimbal_spin_translate_marker",
            markers={
                "inactive": sim_utils.SphereCfg(
                    radius=0.001,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.0, 0.0),
                        emissive_color=(0.0, 0.0, 0.0),
                    ),
                ),
                "active": sim_utils.SphereCfg(
                    radius=float(getattr(self.cfg, "play_gimbal_spin_translate_marker_radius", 0.12)),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 1.0, 0.55),
                        emissive_color=(0.0, 0.35, 0.12),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
            },
        )
        self._gimbal_spin_translate_marker = VisualizationMarkers(marker_cfg)
        self._gimbal_spin_translate_marker.set_visibility(True)

    def _update_gimbal_spin_translate_marker(self) -> None:
        marker = getattr(self, "_gimbal_spin_translate_marker", None)
        if marker is None:
            return
        if not self._is_gimbal_spin_translate_marker_enabled():
            marker.set_visibility(False)
            return
        marker.set_visibility(True)
        active = getattr(
            self,
            "_gimbal_spin_translate_active",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        positions = self.robot.data.root_pos_w.clone()
        positions[:, 2] += float(getattr(self.cfg, "play_gimbal_spin_translate_marker_height", 0.85))
        positions[~active, 2] = -1000.0
        marker_indices = active.to(dtype=torch.long)
        marker.visualize(translations=positions, marker_indices=marker_indices)

    def _get_command_special_mode_mask(self, mode_name: str) -> torch.Tensor:
        command_generator = getattr(self, "command_generator", None)
        special_mode_id = getattr(command_generator, "special_mode_id", None)
        mode_names = tuple(getattr(command_generator, "_mode_names", ()))
        if command_generator is None or special_mode_id is None or len(mode_names) == 0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        try:
            mode_idx = mode_names.index(mode_name)
        except ValueError:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return special_mode_id == mode_idx

    def _get_gimbal_heading_control_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        if not self._is_gimbal_heading_control_enabled() or env_ids.numel() == 0:
            return torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        cfg = self._get_gimbal_heading_control_cfg()
        if not bool(cfg.get("apply_only_in_special_mode", False)):
            return torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
        mode_name = str(
            cfg.get(
                "special_mode_name",
                self._get_gimbal_spin_translate_cfg().get("special_mode_name", "gimbal_spin_translate"),
            )
        )
        return self._get_command_special_mode_mask(mode_name)[env_ids]

    def _ensure_gimbal_heading_pd_gain_tensors(self) -> None:
        cfg = self._get_gimbal_heading_control_cfg()
        kp = float(cfg.get("kp", 2.0))
        kd = float(cfg.get("kd", 0.15))
        if not hasattr(self, "_gimbal_heading_kp") or self._gimbal_heading_kp.shape[0] != self.num_envs:
            self._gimbal_heading_kp = torch.full(
                (self.num_envs,),
                kp,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self._gimbal_heading_kp = self._gimbal_heading_kp.to(device=self.device, dtype=torch.float)
        if not hasattr(self, "_gimbal_heading_kd") or self._gimbal_heading_kd.shape[0] != self.num_envs:
            self._gimbal_heading_kd = torch.full(
                (self.num_envs,),
                kd,
                dtype=torch.float,
                device=self.device,
            )
        else:
            self._gimbal_heading_kd = self._gimbal_heading_kd.to(device=self.device, dtype=torch.float)

    def _get_gimbal_yaw_actuator(self):
        actuators = getattr(self.robot, "actuators", None)
        if not isinstance(actuators, dict):
            return None
        return actuators.get("gimbal_yaw", None)

    def _capture_gimbal_yaw_actuator_gains(self) -> None:
        actuator = self._get_gimbal_yaw_actuator()
        if actuator is None:
            return
        if hasattr(actuator, "stiffness"):
            self._gimbal_yaw_actuator_default_stiffness = actuator.stiffness.detach().clone()
        if hasattr(actuator, "damping"):
            self._gimbal_yaw_actuator_default_damping = actuator.damping.detach().clone()

    def _set_gimbal_yaw_actuator_gains_for_heading_control(self, env_ids: torch.Tensor, enabled: bool) -> None:
        actuator = self._get_gimbal_yaw_actuator()
        if actuator is None or env_ids.numel() == 0:
            return
        if enabled:
            if hasattr(actuator, "stiffness"):
                actuator.stiffness[env_ids] = 0.0
            if hasattr(actuator, "damping"):
                actuator.damping[env_ids] = 0.0
            return
        if hasattr(actuator, "stiffness") and self._gimbal_yaw_actuator_default_stiffness is not None:
            actuator.stiffness[env_ids] = self._gimbal_yaw_actuator_default_stiffness[env_ids]
        if hasattr(actuator, "damping") and self._gimbal_yaw_actuator_default_damping is not None:
            actuator.damping[env_ids] = self._gimbal_yaw_actuator_default_damping[env_ids]

    @staticmethod
    def _sample_uniform_range(
        value_range: tuple[float, float] | list[float],
        count: int,
        device: torch.device,
    ) -> torch.Tensor:
        low, high = float(value_range[0]), float(value_range[1])
        if high < low:
            low, high = high, low
        if abs(high - low) < 1.0e-6:
            return torch.full((count,), low, dtype=torch.float, device=device)
        return torch.empty(count, dtype=torch.float, device=device).uniform_(low, high)

    def _sample_gimbal_heading_targets(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not hasattr(self, "_gimbal_heading_target_w"):
            return
        cfg = self._get_gimbal_heading_control_cfg()
        mode = str(cfg.get("target_mode", "hold_reset_heading"))
        if mode == "fixed":
            self._gimbal_heading_target_w[env_ids] = float(cfg.get("fixed_heading", 0.0))
        elif mode == "sampled":
            heading_range = cfg.get("heading_range", (-torch.pi, torch.pi))
            self._gimbal_heading_target_w[env_ids] = self._sample_uniform_range(
                heading_range,
                int(env_ids.numel()),
                self.device,
            )
        else:
            self._gimbal_heading_target_w[env_ids] = self.robot.data.heading_w[env_ids]
        self._gimbal_heading_target_w[env_ids] = wrap_to_pi(self._gimbal_heading_target_w[env_ids])
        if hasattr(self, "_gimbal_heading_target_initialized"):
            self._gimbal_heading_target_initialized[env_ids] = True

    def _sample_gimbal_spin_translate_velocity(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not hasattr(self, "_gimbal_spin_translate_lin_vel_yaw"):
            return
        cfg = self._get_gimbal_spin_translate_cfg()
        if "lin_vel_yaw_speed_range" in cfg or "lin_vel_yaw_heading_range" in cfg:
            speed_range = cfg.get("lin_vel_yaw_speed_range", (0.0, 0.4))
            heading_range = cfg.get("lin_vel_yaw_heading_range", (-torch.pi, torch.pi))
            if (
                isinstance(speed_range, (list, tuple))
                and len(speed_range) > 0
                and isinstance(speed_range[0], (list, tuple))
            ):
                # Multi-segment: list of (min, max) tuples, randomly select one segment per env
                n_segments = len(speed_range)
                seg_idx = torch.randint(0, n_segments, (int(env_ids.numel()),), device=self.device)
                speed = torch.empty(int(env_ids.numel()), dtype=torch.float, device=self.device)
                for i, seg in enumerate(speed_range):
                    mask = seg_idx == i
                    n = mask.sum().item()
                    if n > 0:
                        speed[mask] = self._sample_uniform_range(seg, n, self.device)
                speed = torch.clamp(speed, min=0.0)
            else:
                # Single tuple: (min, max)
                speed = torch.clamp(
                    self._sample_uniform_range(speed_range, int(env_ids.numel()), self.device),
                    min=0.0,
                )
            speed_deadzone = max(float(cfg.get("lin_vel_yaw_speed_deadzone", 0.0)), 0.0)
            if speed_deadzone > 0.0:
                speed = torch.where(speed < speed_deadzone, torch.zeros_like(speed), speed)
            heading = self._sample_uniform_range(heading_range, int(env_ids.numel()), self.device)
            self._gimbal_spin_translate_sin_heading[env_ids] = torch.sin(heading)
            self._gimbal_spin_translate_cos_heading[env_ids] = torch.cos(heading)
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 0] = speed * torch.cos(heading)
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 1] = speed * torch.sin(heading)
        else:
            x_range = cfg.get("lin_vel_x_yaw_range", (-0.4, 0.4))
            y_range = cfg.get("lin_vel_y_yaw_range", (-0.4, 0.4))
            vx = self._sample_uniform_range(x_range, int(env_ids.numel()), self.device)
            vy = self._sample_uniform_range(y_range, int(env_ids.numel()), self.device)
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 0] = vx
            self._gimbal_spin_translate_lin_vel_yaw[env_ids, 1] = vy
            heading_raw = torch.atan2(vy, vx)
            self._gimbal_spin_translate_sin_heading[env_ids] = torch.sin(heading_raw)
            self._gimbal_spin_translate_cos_heading[env_ids] = torch.cos(heading_raw)
        height_range = cfg.get("lin_vel_yaw_height_range", None)
        if height_range is not None:
            if (
                isinstance(height_range, (list, tuple))
                and len(height_range) > 0
                and isinstance(height_range[0], (list, tuple))
            ):
                n_segments = len(height_range)
                seg_idx = torch.randint(0, n_segments, (int(env_ids.numel()),), device=self.device)
                height_cmd = torch.empty(int(env_ids.numel()), dtype=torch.float, device=self.device)
                for i, seg in enumerate(height_range):
                    mask = seg_idx == i
                    n = mask.sum().item()
                    if n > 0:
                        height_cmd[mask] = self._sample_uniform_range(seg, n, self.device)
                self._gimbal_spin_translate_height_cmd[env_ids] = height_cmd
            else:
                self._gimbal_spin_translate_height_cmd[env_ids] = self._sample_uniform_range(
                    height_range, int(env_ids.numel()), self.device
                )
        else:
            self._gimbal_spin_translate_height_cmd[env_ids] = 0.0

    def _get_gimbal_yaw_joint_angle_wrapped(self) -> torch.Tensor:
        if len(self._gimbal_yaw_idx) == 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        yaw_pos = self.joint_pos[:, self._gimbal_yaw_idx[0]]
        return wrap_to_pi(yaw_pos)

    def _get_gimbal_yaw_link_heading_w(self) -> torch.Tensor:
        if len(self._gimbal_yaw_link_idx) > 0:
            _, _, yaw = euler_xyz_from_quat(self.robot.data.body_quat_w[:, self._gimbal_yaw_link_idx[0]])
            return wrap_to_pi(yaw)
        return self.robot.data.heading_w

    def _get_gimbal_yaw_link_ang_vel_z_w(self) -> torch.Tensor:
        if len(self._gimbal_yaw_link_idx) > 0 and hasattr(self.robot.data, "body_ang_vel_w"):
            return self.robot.data.body_ang_vel_w[:, self._gimbal_yaw_link_idx[0], 2]
        if hasattr(self.robot.data, "root_ang_vel_w"):
            return self.robot.data.root_ang_vel_w[:, 2]
        return self.robot.data.root_ang_vel_b[:, 2]

    def _apply_gimbal_heading_pd(self, env_ids: torch.Tensor) -> None:
        if not self._is_gimbal_heading_control_enabled() or env_ids.numel() == 0:
            return
        if hasattr(self, "_gimbal_heading_target_initialized"):
            init_mask = self._gimbal_heading_target_initialized[env_ids]
            if not torch.all(init_mask):
                self._sample_gimbal_heading_targets(env_ids[~init_mask])
        cfg = self._get_gimbal_heading_control_cfg()
        heading_error = wrap_to_pi(
            self._gimbal_heading_target_w[env_ids] - self._get_gimbal_yaw_link_heading_w()[env_ids]
        )
        heading_rate = self._get_gimbal_yaw_link_ang_vel_z_w()[env_ids]
        self._ensure_gimbal_heading_pd_gain_tensors()
        effort = self._gimbal_heading_kp[env_ids] * heading_error - self._gimbal_heading_kd[env_ids] * heading_rate
        max_effort = float(cfg.get("max_effort", 2.0))
        if max_effort > 0.0:
            effort = torch.clamp(effort, -max_effort, max_effort)
        self.robot.set_joint_effort_target(
            effort.unsqueeze(-1),
            joint_ids=self._gimbal_yaw_idx,
            env_ids=env_ids,
        )

    def _get_gimbal_spin_translate_mode_mask(self) -> torch.Tensor:
        if not self._is_gimbal_spin_translate_enabled():
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mode_name = str(self._get_gimbal_spin_translate_cfg().get("special_mode_name", "gimbal_spin_translate"))
        return self._get_command_special_mode_mask(mode_name)

    def _update_gimbal_spin_translate_samples(self, active_mask: torch.Tensor) -> None:
        if not hasattr(self, "_gimbal_spin_translate_last_command_counter"):
            return
        command_counter = getattr(self.command_generator, "command_counter", None)
        if command_counter is None:
            resample_mask = active_mask & ~self._gimbal_spin_translate_active
        else:
            command_counter = command_counter.to(device=self.device, dtype=torch.long)
            resample_mask = active_mask & (
                self._gimbal_spin_translate_last_command_counter != command_counter
            )
        resample_ids = resample_mask.nonzero(as_tuple=False).flatten()
        if resample_ids.numel() > 0:
            self._sample_gimbal_spin_translate_velocity(resample_ids)
            if command_counter is not None:
                self._gimbal_spin_translate_last_command_counter[resample_ids] = command_counter[resample_ids]
        inactive_ids = (~active_mask).nonzero(as_tuple=False).flatten()
        if inactive_ids.numel() > 0:
            self._gimbal_spin_translate_lin_vel_yaw[inactive_ids] = 0.0
            self._gimbal_spin_translate_last_command_counter[inactive_ids] = -1
        self._gimbal_spin_translate_active = active_mask.clone()

    def _apply_gimbal_spin_translate_command(self) -> None:
        active_mask = self._get_gimbal_spin_translate_mode_mask()
        self._update_gimbal_spin_translate_samples(active_mask)
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        if active_ids.numel() == 0:
            return
        if not bool(self._get_gimbal_spin_translate_cfg().get("project_to_body_command", True)):
            self.command[active_ids, 0:2] = 0.0
            return
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()[active_ids]
        cos_yaw = torch.cos(yaw_angle)
        sin_yaw = torch.sin(yaw_angle)
        vel_yaw = self._gimbal_spin_translate_lin_vel_yaw[active_ids]
        self.command[active_ids, 0] = cos_yaw * vel_yaw[:, 0] - sin_yaw * vel_yaw[:, 1]
        self.command[active_ids, 1] = sin_yaw * vel_yaw[:, 0] + cos_yaw * vel_yaw[:, 1]
        if hasattr(self, "_gimbal_spin_translate_height_cmd"):
            height_cmd = self._gimbal_spin_translate_height_cmd[active_ids]
            override_mask = height_cmd > 0.0
            if torch.any(override_mask):
                self.command[active_ids[override_mask], 2] = height_cmd[override_mask]

    def _get_gimbal_spin_translate_measured_lin_vel_yaw(self) -> torch.Tensor:
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()
        cos_yaw = torch.cos(yaw_angle)
        sin_yaw = torch.sin(yaw_angle)
        vel_b = self.robot.data.root_lin_vel_b[:, :2]
        vel_yaw = torch.zeros_like(vel_b)
        vel_yaw[:, 0] = cos_yaw * vel_b[:, 0] + sin_yaw * vel_b[:, 1]
        vel_yaw[:, 1] = -sin_yaw * vel_b[:, 0] + cos_yaw * vel_b[:, 1]
        return vel_yaw

    def _get_gimbal_spin_translate_reward_terms(self) -> dict[str, torch.Tensor]:
        zero = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        if not hasattr(self, "_gimbal_spin_translate_active") or not torch.any(self._gimbal_spin_translate_active):
            return {
                "gimbal_spin_track_lin_vel_yaw_frame": zero,
                "gimbal_spin_track_lin_speed": zero,
                "gimbal_spin_track_lin_heading": zero,
                "gimbal_spin_lin_vel_yaw_square": zero,
                "gimbal_spin_lin_speed_overshoot": zero,
                "gimbal_spin_heading_error_square": zero,
                "gimbal_spin_stand_still_lin_vel": zero,
                "gimbal_spin_track_lin_heading_v2": zero,
                "gimbal_spin_heading_error_square_v2": zero,
            }

        active = self._gimbal_spin_translate_active.float()
        cmd_yaw = self._gimbal_spin_translate_lin_vel_yaw
        meas_yaw = self._get_gimbal_spin_translate_measured_lin_vel_yaw()
        err_yaw = cmd_yaw - meas_yaw
        err_yaw_sq = torch.sum(torch.square(err_yaw), dim=-1)

        speed_cmd = torch.linalg.norm(cmd_yaw, dim=-1)
        speed_meas = torch.linalg.norm(meas_yaw, dim=-1)
        speed_err = speed_cmd - speed_meas

        heading_cmd = torch.atan2(cmd_yaw[:, 1], cmd_yaw[:, 0])
        heading_meas = torch.atan2(meas_yaw[:, 1], meas_yaw[:, 0])
        heading_err = wrap_to_pi(heading_cmd - heading_meas)
        # Vector-based heading error: single atan2(cross, dot), naturally in [-π, π]
        heading_err_v2 = torch.atan2(
            cmd_yaw[:, 0] * meas_yaw[:, 1] - cmd_yaw[:, 1] * meas_yaw[:, 0],
            cmd_yaw[:, 0] * meas_yaw[:, 0] + cmd_yaw[:, 1] * meas_yaw[:, 1],
        )
        heading_gate = (
            (speed_cmd > float(getattr(self.cfg, "gimbal_spin_heading_cmd_speed_min", 0.1)))
            & (speed_meas > float(getattr(self.cfg, "gimbal_spin_heading_meas_speed_min", 0.05)))
        ).float()

        lin_vel_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_vel_yaw_sigma", 0.25)), 1.0e-6)
        speed_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_speed_sigma", 0.25)), 1.0e-6)
        heading_sigma = max(float(getattr(self.cfg, "gimbal_spin_lin_heading_sigma", 0.25)), 1.0e-6)
        lin_square_sigma = float(getattr(self.cfg, "gimbal_spin_lin_vel_yaw_square_sigma", 1.0))
        overshoot_sigma = float(getattr(self.cfg, "gimbal_spin_lin_speed_overshoot_sigma", 1.0))
        heading_square_sigma = float(getattr(self.cfg, "gimbal_spin_heading_error_square_sigma", 1.0))
        stand_still_speed_threshold = float(
            getattr(
                self.cfg,
                "gimbal_spin_stand_still_speed_threshold",
                self._get_gimbal_spin_translate_cfg().get("lin_vel_yaw_speed_deadzone", 0.05),
            )
        )
        stand_still_mask = (speed_cmd <= max(stand_still_speed_threshold, 0.0)).float()

        return {
            "gimbal_spin_track_lin_vel_yaw_frame": torch.exp(-err_yaw_sq / lin_vel_sigma) * active,
            "gimbal_spin_track_lin_speed": torch.exp(-torch.square(speed_err) / speed_sigma) * active,
            "gimbal_spin_track_lin_heading": torch.exp(-torch.square(heading_err) / heading_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            "gimbal_spin_lin_vel_yaw_square": (lin_square_sigma ** 2) * err_yaw_sq * active,
            "gimbal_spin_lin_speed_overshoot": torch.square(
                torch.clamp(speed_meas - speed_cmd, min=0.0) * overshoot_sigma
            ) * active * (1.0 - stand_still_mask),
            "gimbal_spin_heading_error_square": torch.square(heading_err * heading_square_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            "gimbal_spin_stand_still_lin_vel": torch.sum(torch.abs(meas_yaw), dim=-1) * active * stand_still_mask,
            "gimbal_spin_track_lin_heading_v2": torch.exp(-torch.square(heading_err_v2) / heading_sigma) * active * heading_gate * (1.0 - stand_still_mask),
            "gimbal_spin_heading_error_square_v2": torch.square(heading_err_v2 * heading_square_sigma) * active * heading_gate * (1.0 - stand_still_mask),
        }

    def _apply_gimbal_targets(self, env_ids: torch.Tensor | None = None) -> None:
        if len(self._gimbal_idx) == 0:
            return
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        if len(self._gimbal_pitch_idx) > 0:
            pitch_targets = self._gimbal_pitch_target[env_ids_t]
            pitch_zero_vel = torch.zeros_like(pitch_targets)
            self.robot.set_joint_position_target(
                pitch_targets,
                joint_ids=self._gimbal_pitch_idx,
                env_ids=env_ids_t,
            )
            self.robot.set_joint_velocity_target(
                pitch_zero_vel,
                joint_ids=self._gimbal_pitch_idx,
                env_ids=env_ids_t,
            )
        if len(self._gimbal_yaw_idx) > 0:
            heading_mask = self._get_gimbal_heading_control_mask(env_ids_t)
            heading_ids = env_ids_t[heading_mask]
            velocity_ids = env_ids_t[~heading_mask]
            if heading_ids.numel() > 0:
                self._set_gimbal_yaw_actuator_gains_for_heading_control(heading_ids, enabled=True)
                self._apply_gimbal_heading_pd(heading_ids)
            if velocity_ids.numel() > 0:
                self._set_gimbal_yaw_actuator_gains_for_heading_control(velocity_ids, enabled=False)
                yaw_vel_targets = self._gimbal_yaw_velocity_target[velocity_ids]
                yaw_zero_effort = torch.zeros(
                    (velocity_ids.numel(), len(self._gimbal_yaw_idx)),
                    dtype=torch.float,
                    device=self.device,
                )
                self.robot.set_joint_effort_target(
                    yaw_zero_effort,
                    joint_ids=self._gimbal_yaw_idx,
                    env_ids=velocity_ids,
                )
                self.robot.set_joint_velocity_target(
                    yaw_vel_targets,
                    joint_ids=self._gimbal_yaw_idx,
                    env_ids=velocity_ids,
                )

    def _reset_gimbal_joints(self, env_ids: torch.Tensor | None = None) -> None:
        if len(self._gimbal_idx) == 0:
            return
        env_ids_t = self._as_env_ids_tensor(env_ids)
        if env_ids_t.numel() == 0:
            return
        self._sample_gimbal_yaw_velocity_targets(env_ids_t)
        if len(self._gimbal_pitch_idx) > 0:
            pitch_joint_pos = self._gimbal_pitch_target[env_ids_t]
            pitch_joint_vel = torch.zeros_like(pitch_joint_pos)
            self.robot.write_joint_state_to_sim(
                pitch_joint_pos,
                pitch_joint_vel,
                self._gimbal_pitch_idx,
                env_ids_t,
            )
        if len(self._gimbal_yaw_idx) > 0:
            heading_cfg = self._get_gimbal_heading_control_cfg()
            reset_with_heading_control = self._is_gimbal_heading_control_enabled() and not bool(
                heading_cfg.get("apply_only_in_special_mode", False)
            )
            if reset_with_heading_control:
                self._sample_gimbal_heading_targets(env_ids_t)
                yaw_joint_pos = wrap_to_pi(
                    self._gimbal_heading_target_w[env_ids_t] - self.robot.data.heading_w[env_ids_t]
                ).unsqueeze(-1)
                yaw_joint_vel = torch.zeros_like(yaw_joint_pos)
            else:
                yaw_joint_pos = torch.zeros(
                    (env_ids_t.numel(), len(self._gimbal_yaw_idx)),
                    dtype=torch.float,
                    device=self.device,
                )
                yaw_joint_vel = self._gimbal_yaw_velocity_target[env_ids_t]
                if hasattr(self, "_gimbal_heading_target_initialized"):
                    self._gimbal_heading_target_initialized[env_ids_t] = False
            self.robot.write_joint_state_to_sim(
                yaw_joint_pos,
                yaw_joint_vel,
                self._gimbal_yaw_idx,
                env_ids_t,
            )
        self._sample_gimbal_spin_translate_velocity(env_ids_t)
        self._gimbal_spin_translate_active[env_ids_t] = False
        self._gimbal_spin_translate_last_command_counter[env_ids_t] = -1
        self._apply_gimbal_targets(env_ids_t)

    def _on_command_updated(self) -> None:
        super()._on_command_updated()
        self._apply_gimbal_spin_translate_command()

    def _postprocess_reward_terms(self, reward_terms: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        reward_terms = super()._postprocess_reward_terms(reward_terms)
        if hasattr(self, "_gimbal_spin_translate_active") and torch.any(self._gimbal_spin_translate_active):
            keep_mask = (~self._gimbal_spin_translate_active).float()
            suppressed_terms = getattr(
                self.cfg,
                "gimbal_spin_suppressed_reward_terms",
                (
                    "track_lin_vel_xy",
                    "track_lin_vel_xy_soft",
                    "track_lin_vel_xy_tight",
                    "track_lin_vel_xy_huge_gap",
                    "track_lin_vel_xy_square",
                    "stand_still_lin_vel",
                ),
            )
            for name in suppressed_terms:
                if name in reward_terms:
                    reward_terms[name] = reward_terms[name] * keep_mask
        reward_terms.update(self._get_gimbal_spin_translate_reward_terms())
        return reward_terms

    def _get_ctrl_mode_obs_raw(self) -> torch.Tensor:
        obs = super()._get_ctrl_mode_obs_raw()
        if obs.shape[-1] < 7:
            return obs
        active_mask = self._gimbal_spin_translate_active
        if not torch.any(active_mask):
            return obs
        obs = obs.clone()
        active_ids = active_mask.nonzero(as_tuple=False).flatten()
        obs[active_ids, :7] = 0.0
        obs[active_ids, 1] = 1.0
        vel_yaw = self._gimbal_spin_translate_lin_vel_yaw[active_ids]
        speed = torch.linalg.norm(vel_yaw, dim=-1)
        obs[active_ids, 2] = speed
        if bool(self._get_gimbal_spin_translate_cfg().get("use_sampled_heading_obs", True)):
            obs[active_ids, 3] = self._gimbal_spin_translate_sin_heading[active_ids]
            obs[active_ids, 4] = self._gimbal_spin_translate_cos_heading[active_ids]
        else:
            heading = torch.atan2(vel_yaw[:, 1], vel_yaw[:, 0])
            obs[active_ids, 3] = torch.sin(heading)
            obs[active_ids, 4] = torch.cos(heading)
        if bool(self._get_gimbal_spin_translate_cfg().get("zero_heading_in_deadzone", False)):
            deadzone = max(float(self._get_gimbal_spin_translate_cfg().get("lin_vel_yaw_speed_deadzone", 0.0)), 0.0)
            zero_heading_mask = speed <= deadzone
            if torch.any(zero_heading_mask):
                obs[active_ids[zero_heading_mask], 3] = 0.0
                obs[active_ids[zero_heading_mask], 4] = 0.0
        yaw_angle = self._get_gimbal_yaw_joint_angle_wrapped()[active_ids]
        obs[active_ids, 5] = torch.sin(yaw_angle)
        obs[active_ids, 6] = torch.cos(yaw_angle)
        return obs

    def _apply_action(self) -> None:
        super()._apply_action()
        self._apply_gimbal_spin_translate_command()
        self._apply_gimbal_targets(self.robot._ALL_INDICES)
        self._update_gimbal_spin_translate_marker()

    def _custom_reset_random(self, env_ids):
        super()._custom_reset_random(env_ids)
        self._reset_gimbal_joints(env_ids)
