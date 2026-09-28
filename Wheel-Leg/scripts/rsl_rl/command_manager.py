"""Interactive command state for the Isaac Sim play script."""

from __future__ import annotations

from dataclasses import dataclass


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class CommandState:
    """Human-facing command state; policy actions are never modified here."""

    vx_cmd: float = 0.0
    yaw_rate_cmd: float = 0.0
    base_height_cmd: float = 0.32
    vx_target: float = 0.0
    yaw_target: float = 0.0
    v_set: float = 0.8
    w_set: float = 1.0
    default_height: float = 0.32
    emergency_stop: bool = False
    reset_requested: bool = False


class CommandManager:
    """Maps keyboard intent to policy commands with optional slew-rate limiting."""

    def __init__(
        self,
        default_height: float = 0.32,
        height_min: float = 0.25,
        height_max: float = 0.50,
        velocity_step: float = 0.05,
        yaw_step: float = 0.05,
        height_step: float = 0.005,
        max_accel: float = 2.0,
        max_yaw_accel: float = 4.0,
        smooth: bool = True,
        velocity_max: float = 1.5,
        yaw_max: float = 5.0,
    ) -> None:
        self.velocity_max = velocity_max
        self.yaw_max = yaw_max
        self.height_min = height_min
        self.height_max = height_max
        self.velocity_step = velocity_step
        self.yaw_step = yaw_step
        self.height_step = height_step
        self.max_accel = max_accel
        self.max_yaw_accel = max_yaw_accel
        self.smooth = smooth
        self.state = CommandState(default_height=default_height, base_height_cmd=default_height)
        self.update_settings()

    def update_settings(self, v_delta: float = 0.0, w_delta: float = 0.0) -> None:
        self.state.v_set = clamp(self.state.v_set + v_delta, min(0.1, self.velocity_max), self.velocity_max)
        self.state.w_set = clamp(self.state.w_set + w_delta, min(0.1, self.yaw_max), self.yaw_max)

    def update_height(self, delta: float) -> None:
        self.state.base_height_cmd = clamp(
            self.state.base_height_cmd + delta, self.height_min, self.height_max
        )

    def reset_height(self) -> None:
        self.state.base_height_cmd = self.state.default_height

    def set_pressed_targets(self, pressed: set[str]) -> None:
        forward = "W" in pressed
        backward = "S" in pressed
        left = "A" in pressed
        right = "D" in pressed
        self.state.vx_target = 0.0 if forward == backward else (self.state.v_set if forward else -self.state.v_set)
        self.state.yaw_target = 0.0 if left == right else (self.state.w_set if left else -self.state.w_set)
        self.state.emergency_stop = "SPACE" in pressed
        if self.state.emergency_stop:
            self.state.vx_target = 0.0
            self.state.yaw_target = 0.0

    def advance(self, dt: float, pressed: set[str]) -> CommandState:
        self.set_pressed_targets(pressed)
        if self.state.emergency_stop:
            self.state.vx_cmd = 0.0
            self.state.yaw_rate_cmd = 0.0
        elif self.smooth:
            self.state.vx_cmd = self._slew(self.state.vx_cmd, self.state.vx_target, self.max_accel * dt)
            self.state.yaw_rate_cmd = self._slew(
                self.state.yaw_rate_cmd, self.state.yaw_target, self.max_yaw_accel * dt
            )
        else:
            self.state.vx_cmd = self.state.vx_target
            self.state.yaw_rate_cmd = self.state.yaw_target
        return self.state

    @staticmethod
    def _slew(current: float, target: float, maximum_delta: float) -> float:
        delta = clamp(target - current, -maximum_delta, maximum_delta)
        return current + delta
