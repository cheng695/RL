"""Custom action terms for wheel-leg locomotion."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass


class RateLimitedJointPositionAction(JointPositionAction):
    """Joint position action with a target-position slew-rate limit."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        if cfg.max_position_rate <= 0.0:
            raise ValueError("max_position_rate must be positive.")

        self._max_position_delta = float(cfg.max_position_rate) * env.step_dt
        self._initial_target = torch.zeros_like(self.processed_actions)
        if isinstance(self._offset, torch.Tensor):
            self._initial_target.copy_(self._offset)
        else:
            self._initial_target.fill_(float(self._offset))
        self._previous_target = self._initial_target.clone()

    def process_actions(self, actions: torch.Tensor):
        """Scale the policy action, then limit target change per policy step."""
        super().process_actions(actions)
        delta = torch.clamp(
            self.processed_actions - self._previous_target,
            min=-self._max_position_delta,
            max=self._max_position_delta,
        )
        self._processed_actions.copy_(self._previous_target + delta)
        self._previous_target.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_target[env_ids] = self._initial_target[env_ids]


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for joint position targets with a slew-rate limit."""

    class_type: type[ActionTerm] = RateLimitedJointPositionAction
    max_position_rate: float = 4.0
    """Maximum target position change in rad/s."""


class RateLimitedJointVelocityAction(JointVelocityAction):
    """Joint velocity action with a target-velocity slew-rate limit."""

    cfg: RateLimitedJointVelocityActionCfg

    def __init__(self, cfg: RateLimitedJointVelocityActionCfg, env):
        super().__init__(cfg, env)
        if cfg.max_velocity_rate <= 0.0:
            raise ValueError("max_velocity_rate must be positive.")

        self._max_velocity_delta = float(cfg.max_velocity_rate) * env.step_dt
        self._initial_target = torch.zeros_like(self.processed_actions)
        if isinstance(self._offset, torch.Tensor):
            self._initial_target.copy_(self._offset)
        else:
            self._initial_target.fill_(float(self._offset))
        self._previous_target = self._initial_target.clone()
        self._target_delta = torch.zeros_like(self.processed_actions)

    @property
    def target_delta(self) -> torch.Tensor:
        """Change of the final velocity target in the latest control step (rad/s)."""
        return self._target_delta

    def process_actions(self, actions: torch.Tensor):
        """Scale the policy action, then limit target change per policy step."""
        super().process_actions(actions)
        delta = torch.clamp(
            self.processed_actions - self._previous_target,
            min=-self._max_velocity_delta,
            max=self._max_velocity_delta,
        )
        self._processed_actions.copy_(self._previous_target + delta)
        self._target_delta.copy_(self._processed_actions - self._previous_target)
        self._previous_target.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_target[env_ids] = self._initial_target[env_ids]
        self._target_delta[env_ids] = 0.0


@configclass
class RateLimitedJointVelocityActionCfg(JointVelocityActionCfg):
    """Configuration for velocity actions with a target slew-rate limit."""

    class_type: type[ActionTerm] = RateLimitedJointVelocityAction
    max_velocity_rate: float = 200.0
    """Maximum target velocity change in rad/s per second (rad/s²)."""
