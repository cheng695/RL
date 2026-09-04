from __future__ import annotations

import math

import torch

import isaaclab.utils.string as string_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


def _resolve_joint_values(
    value: float | dict[str, float],
    joint_names: list[str],
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Resolve a scalar or regex-name mapping to a per-joint tensor."""
    values = torch.full((num_envs, len(joint_names)), float(value) if not isinstance(value, dict) else 0.0, device=device)
    if isinstance(value, dict):
        index_list, _, value_list = string_utils.resolve_matching_names_values(value, joint_names)
        values[:, index_list] = torch.tensor(value_list, device=device)
    return values


class BoundedJointPositionAction(JointPositionAction):
    """Joint position action with a hard target clamp around the configured offset."""

    cfg: "BoundedJointPositionActionCfg"

    def __init__(self, cfg: "BoundedJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._target_clip = _resolve_joint_values(cfg.target_clip, self._joint_names, self.num_envs, self.device)
        self._max_target_delta = float(cfg.max_target_delta)
        self._previous_targets = self._offset.clone() if isinstance(self._offset, torch.Tensor) else torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        offset = self._offset if isinstance(self._offset, torch.Tensor) else float(self._offset)
        self._processed_actions = torch.clamp(
            self._processed_actions,
            min=offset - self._target_clip,
            max=offset + self._target_clip,
        )
        if self._max_target_delta > 0.0:
            delta = torch.clamp(
                self._processed_actions - self._previous_targets,
                -self._max_target_delta,
                self._max_target_delta,
            )
            self._processed_actions = self._previous_targets + delta
            self._previous_targets[:] = self._processed_actions

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if self._max_target_delta > 0.0:
            offset = self._offset if isinstance(self._offset, torch.Tensor) else 0.0
            self._previous_targets[env_ids] = offset[env_ids] if isinstance(offset, torch.Tensor) else offset


@configclass
class BoundedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for bounded joint position actions."""

    class_type: type[ActionTerm] = BoundedJointPositionAction

    target_clip: float | dict[str, float] = 0.8
    """Hard target clamp around the configured offset, in rad."""

    max_target_delta: float = 0.08
    """Maximum target position change per policy step, in rad. Use <= 0 to disable."""


class NonlinearJointVelocityAction(JointVelocityAction):
    """Joint velocity action with cubic shaping around zero raw action."""

    cfg: "NonlinearJointVelocityActionCfg"

    def __init__(self, cfg: "NonlinearJointVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        self._cubic_beta = float(cfg.cubic_beta)
        self._standing_scale = float(cfg.standing_scale)
        self._linear_command_scale_ref = float(cfg.linear_command_scale_ref)
        self._yaw_command_scale_ref = float(cfg.yaw_command_scale_ref)
        self._shaped_actions = torch.zeros_like(self._raw_actions)

    @property
    def shaped_actions(self) -> torch.Tensor:
        """Normalized shaped actions before scale/offset are applied."""
        return self._shaped_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        if self._cubic_beta <= 0.0:
            self._shaped_actions = torch.clamp(self._raw_actions, -1.0, 1.0)
        else:
            saturated_actions = torch.clamp(torch.tanh(self._raw_actions) / math.tanh(1.0), -1.0, 1.0)
            self._shaped_actions = (1.0 - self._cubic_beta) * saturated_actions + self._cubic_beta * saturated_actions**3
        scale = self._scale
        if self.cfg.command_name:
            command = self._env.command_manager.get_command(self.cfg.command_name)
            linear_gate = torch.linalg.norm(command[:, :2], dim=1) / max(self._linear_command_scale_ref, 1.0e-6)
            yaw_gate = torch.abs(command[:, 2]) / max(self._yaw_command_scale_ref, 1.0e-6)
            command_gate = torch.clamp(torch.maximum(linear_gate, yaw_gate), 0.0, 1.0).unsqueeze(-1)
            moving_scale = self._scale if isinstance(self._scale, torch.Tensor) else float(self._scale)
            scale = self._standing_scale + (moving_scale - self._standing_scale) * command_gate
        self._processed_actions = self._shaped_actions * scale + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._shaped_actions[:] = 0.0
        else:
            self._shaped_actions[env_ids] = 0.0


@configclass
class NonlinearJointVelocityActionCfg(JointVelocityActionCfg):
    """Configuration for cubic-shaped joint velocity actions."""

    class_type: type[ActionTerm] = NonlinearJointVelocityAction

    cubic_beta: float = 0.8
    """Blend factor for shaped_action = (1-beta)*a + beta*a^3."""

    command_name: str = ""
    """If set, scale wheel velocity targets according to the active velocity command."""

    standing_scale: float = 15.0
    """Wheel velocity scale used when velocity/yaw commands are near zero, in rad/s."""

    linear_command_scale_ref: float = 1.0
    """Linear command magnitude that restores the full configured velocity scale."""

    yaw_command_scale_ref: float = 1.0
    """Yaw-rate command magnitude that restores the full configured velocity scale."""


class SafePhaseJointPositionAction(JointPositionAction):
    """Map normalized phase actions to safe cyclic joint position targets.

    Raw actions are interpreted as phase commands in ``[-1, 1]`` and mapped to
    ``[0, 2*pi]``. The applied joint target is a bounded sinusoid around the
    default joint position, with optional per-step target rate limiting.
    """

    cfg: "SafePhaseJointPositionActionCfg"

    def __init__(self, cfg: "SafePhaseJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._phase_amplitude = _resolve_joint_values(cfg.phase_amplitude, self._joint_names, self.num_envs, self.device)
        self._phase_bias = _resolve_joint_values(cfg.phase_bias, self._joint_names, self.num_envs, self.device)
        self._phase_offset = _resolve_joint_values(cfg.phase_offset, self._joint_names, self.num_envs, self.device)
        self._target_clip = _resolve_joint_values(cfg.target_clip, self._joint_names, self.num_envs, self.device)
        self._max_target_delta = float(cfg.max_target_delta)
        self._previous_targets = self._offset.clone() if isinstance(self._offset, torch.Tensor) else torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        phase_input = torch.clamp(self._raw_actions, -1.0, 1.0)
        phase = (phase_input + 1.0) * math.pi + self._phase_offset

        offset = self._offset if isinstance(self._offset, torch.Tensor) else float(self._offset)
        targets = offset + self._phase_bias + self._phase_amplitude * torch.sin(phase)
        targets = torch.clamp(targets, min=offset - self._target_clip, max=offset + self._target_clip)

        if self._max_target_delta > 0.0:
            delta = torch.clamp(targets - self._previous_targets, -self._max_target_delta, self._max_target_delta)
            targets = self._previous_targets + delta
            self._previous_targets[:] = targets

        self._processed_actions = targets

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        if self._max_target_delta > 0.0:
            offset = self._offset if isinstance(self._offset, torch.Tensor) else 0.0
            self._previous_targets[env_ids] = offset[env_ids] if isinstance(offset, torch.Tensor) else offset


@configclass
class SafePhaseJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for safe cyclic phase-to-position leg actions."""

    class_type: type[ActionTerm] = SafePhaseJointPositionAction

    phase_amplitude: float | dict[str, float] = 0.6
    """Sinusoidal target amplitude around the default joint position, in rad."""

    phase_bias: float | dict[str, float] = 0.0
    """Static target bias added after the default joint position, in rad."""

    phase_offset: float | dict[str, float] = 0.0
    """Per-joint phase offset added to the normalized phase, in rad."""

    target_clip: float | dict[str, float] = 0.65
    """Hard target clamp around the default joint position, in rad."""

    max_target_delta: float = 0.08
    """Maximum target position change per policy step, in rad. Use <= 0 to disable."""
