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

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

from agent_tasks.manager.mdp.state_machine import StateMachineManager

from .airborne import AirborneStateMachine
from .jump_takeoff import JumpTakeoffStateMachine
from .step_up import StepUpStateMachine
from .stair import StairStateMachine


class WheelbipeStateMachineManager(StateMachineManager):
    """Wheelbipe default runtime state-machine stack."""

    _MARKER_NAME_TO_INDEX = {
        "neutral": 0,
        "airborne": 1,
        "step_up": 2,
        "stair": 3,
        "wall_blocked": 4,
        "jump_preload": 5,
        "jump_push": 6,
        "jump_tuck": 7,
    }

    def __init__(self, machines, *, marker_height_offset: float = 1.15):
        super().__init__(
            machines,
            marker_cfg=self._make_marker_cfg(),
            marker_name_to_index=self._MARKER_NAME_TO_INDEX,
            marker_height_offset=marker_height_offset,
        )

    @classmethod
    def from_env(cls, env) -> "WheelbipeStateMachineManager":
        """Create the default wheelbipe state-machine stack for an environment."""
        if not bool(getattr(env.cfg, "enable_state_machines", True)):
            return cls([])
        machines = []
        airborne_cfg = getattr(env.cfg, "airborne_state_machine_cfg", {})
        if bool(airborne_cfg.get("enabled", False)):
            machines.append(AirborneStateMachine())
        jump_takeoff_cfg = getattr(env.cfg, "jump_takeoff_state_machine_cfg", {})
        if bool(jump_takeoff_cfg.get("enabled", False)):
            machines.append(JumpTakeoffStateMachine())
        if bool(env._get_wheel_forward_scan_cfg().get("enabled", False)):
            machines.append(StepUpStateMachine())
        stair_cfg = getattr(env.cfg, "stair_state_machine_cfg", {})
        if bool(stair_cfg.get("enabled", False)):
            machines.append(StairStateMachine())
        return cls(machines)

    @staticmethod
    def _make_marker_cfg() -> VisualizationMarkersCfg:
        return VisualizationMarkersCfg(
            prim_path="/Visuals/state_machine_marker",
            markers={
                "neutral": sim_utils.SphereCfg(
                    radius=0.075,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.45, 0.45, 0.45),
                        emissive_color=(0.08, 0.08, 0.08),
                        metallic=0.0,
                        roughness=0.45,
                    ),
                ),
                "airborne": sim_utils.SphereCfg(
                    radius=0.09,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.45, 0.1),
                        emissive_color=(0.4, 0.12, 0.0),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
                "step_up": sim_utils.SphereCfg(
                    radius=0.09,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.85, 1.0),
                        emissive_color=(0.0, 0.2, 0.28),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
                "stair": sim_utils.SphereCfg(
                    radius=0.095,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.72, 0.25, 1.0),
                        emissive_color=(0.18, 0.02, 0.36),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
                "wall_blocked": sim_utils.SphereCfg(
                    radius=0.095,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.15, 0.15),
                        emissive_color=(0.42, 0.04, 0.04),
                        metallic=0.0,
                        roughness=0.16,
                    ),
                ),
                "jump_preload": sim_utils.SphereCfg(
                    radius=0.09,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.9, 0.35),
                        emissive_color=(0.02, 0.28, 0.06),
                        metallic=0.0,
                        roughness=0.2,
                    ),
                ),
                "jump_push": sim_utils.SphereCfg(
                    radius=0.105,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.85, 0.1),
                        emissive_color=(0.38, 0.26, 0.0),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
                "jump_tuck": sim_utils.SphereCfg(
                    radius=0.095,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.25, 0.45, 1.0),
                        emissive_color=(0.04, 0.08, 0.34),
                        metallic=0.0,
                        roughness=0.18,
                    ),
                ),
            },
        )
