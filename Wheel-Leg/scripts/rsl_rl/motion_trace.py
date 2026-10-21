"""Control-step traces for env0. Implicit-actuator torque is an estimate."""

import csv
import json
import math
from pathlib import Path


class MotionTrace:
    def __init__(self, env, path, checkpoint, leg_names, wheel_names):
        self.env = env.unwrapped
        self.robot = self.env.scene["robot"]
        self.names = list(leg_names) + list(wheel_names)
        self.ids, names = self.robot.find_joints(self.names, preserve_order=True)
        if list(names) != self.names:
            raise ValueError(f"Motion trace joint order mismatch: {names}")
        self.height_ids, _ = self.robot.find_bodies(
            self.env.command_manager.get_term("base_height").cfg.body_name
        )
        self.index = 0
        self.episode = 0
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("x", newline="", encoding="utf-8", buffering=1)
        fields = [
            "step", "time_s", "episode", "done", "state_valid",
            "vx_cmd_mps", "yaw_cmd_rad_s", "height_cmd_m",
            "vx_actual_mps", "vy_actual_mps", "yaw_actual_rad_s",
            "roll_deg", "pitch_deg", "height_actual_m",
        ]
        self.suffixes = [
            "policy_raw", "action_clipped", "q_target_rad", "q_actual_rad",
            "omega_target_rad_s", "omega_actual_rad_s", "tau_estimate_nm",
            "effort_limit_nm", "estimated_effort_at_limit",
        ]
        fields += [f"{name}/{suffix}" for name in self.names for suffix in self.suffixes]
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.path.with_suffix(".json").write_text(json.dumps({
            "checkpoint": str(checkpoint),
            "robot_asset_path": str(self.env.cfg.scene.robot.spawn.asset_path),
            "base_com_randomization_enabled": self.env.cfg.events.base_com is not None,
            "height_reference": "local_ground" if getattr(self.env.cfg, "terrain_relative_height", False) else "world_z",
            "dt": self.env.step_dt,
            "joints_in_action_order": self.names,
            "sampling": "One sample after each control step; not a physics-substep trace.",
            "torque": "robot.data.applied_torque: implicit-actuator estimate, not measured PhysX torque.",
            "done": "Auto-reset may already have replaced state. Physical/action-processed fields are blank when done.",
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    def write(self, raw_action, command, height, done):
        """raw_action and command must be copied before env.step."""
        terminal = bool(done)
        row = dict(step=self.index, time_s=(self.index + 1) * self.env.step_dt,
                   episode=self.episode, done=int(terminal), state_valid=int(not terminal),
                   vx_cmd_mps=float(command[0]), yaw_cmd_rad_s=float(command[2]),
                   height_cmd_m=float(height))
        for name, value in zip(self.names, raw_action):
            row[f"{name}/policy_raw"] = float(value)
        if not terminal:
            data = self.robot.data
            w, x, y, z = data.root_quat_w[0].detach().cpu().tolist()
            row.update(
                vx_actual_mps=float(data.root_lin_vel_b[0, 0]),
                vy_actual_mps=float(data.root_lin_vel_b[0, 1]),
                yaw_actual_rad_s=float(data.root_ang_vel_b[0, 2]),
                roll_deg=math.degrees(math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))),
                pitch_deg=math.degrees(math.asin(max(-1., min(1., 2 * (w*y - z*x))))),
                height_actual_m=float(data.body_pos_w[0, self.height_ids[0], 2]),
            )
            if getattr(self.env.cfg, "terrain_relative_height", False):
                from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import local_ground_height
                row["height_actual_m"] -= float(local_ground_height(self.env)[0])
            # Action terms are explicitly selected to avoid relying on all-action ordering.
            manager = self.env.action_manager
            clipped = (
                manager.get_term("leg_joint_pos").raw_actions[0].detach().cpu().tolist()
                + manager.get_term("wheel_joint_vel").raw_actions[0].detach().cpu().tolist()
            )
            for name, joint_id, action in zip(self.names, self.ids, clipped):
                torque = float(data.applied_torque[0, joint_id])
                limit = abs(float(data.joint_effort_limits[0, joint_id]))
                values = [
                    action, float(data.joint_pos_target[0, joint_id]),
                    float(data.joint_pos[0, joint_id]),
                    float(data.joint_vel_target[0, joint_id]),
                    float(data.joint_vel[0, joint_id]), torque, limit,
                    int(limit > 0 and abs(torque) >= 0.99 * limit),
                ]
                row.update({f"{name}/{suffix}": value
                            for suffix, value in zip(self.suffixes[1:], values)})
        self.writer.writerow(row)
        self.index += 1
        if terminal:
            self.episode += 1

    def close(self):
        self.file.close()
