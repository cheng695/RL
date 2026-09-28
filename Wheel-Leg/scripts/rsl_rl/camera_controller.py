"""Third-person camera follow controller."""

from __future__ import annotations

import math


def yaw_from_quat_wxyz(quat) -> float:
    w, x, y, z = [float(value) for value in quat]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class FollowCamera:
    def __init__(self, env, distance: float = 3.0, height: float = 1.6, lookahead: float = 1.2) -> None:
        self.env = env
        self.distance = distance
        self.height = height
        self.lookahead = lookahead
        self.warning_printed = False

    def update(self) -> None:
        try:
            robot = self.env.unwrapped.scene["robot"]
            position = robot.data.root_pos_w[0].detach().cpu()
            yaw = yaw_from_quat_wxyz(robot.data.root_quat_w[0].detach().cpu())
            forward = (math.cos(yaw), math.sin(yaw))
            eye = (
                float(position[0]) - self.distance * forward[0],
                float(position[1]) - self.distance * forward[1],
                float(position[2]) + self.height,
            )
            target = (
                float(position[0]) + self.lookahead * forward[0],
                float(position[1]) + self.lookahead * forward[1],
                float(position[2]) + 0.35,
            )
            self.env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        except Exception as exc:
            if not self.warning_printed:
                print(f"[WARN] Follow camera disabled: {exc}")
                self.warning_printed = True
