"""Open-loop wheel sign test for WheelLeg.

This script does not load a policy. It holds the legs at a fixed action and
applies several left/right wheel velocity action combinations, then reports
base-frame vx and yaw-rate. Use it to identify wheel sign conventions.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Open-loop wheel sign test for WheelLeg.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--wheel_action", type=float, default=0.5, help="Absolute wheel action magnitude in [-1, 1].")
parser.add_argument("--leg_action", type=float, default=0.35, help="Leg action magnitude for [+, +, -, -].")
parser.add_argument("--steps", type=int, default=360, help="Steps per wheel combination.")
parser.add_argument("--warmup_steps", type=int, default=90, help="Initial steps excluded from statistics.")
parser.add_argument("--output", type=str, default="wheel_open_loop.csv", help="Output CSV path.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run at approximately real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    dt = env.step_dt

    wheel = abs(args_cli.wheel_action)
    leg = abs(args_cli.leg_action)
    cases = [
        ("left+ right+", wheel, wheel),
        ("left- right-", -wheel, -wheel),
        ("left+ right-", wheel, -wheel),
        ("left- right+", -wheel, wheel),
    ]

    rows = []
    for label, left_wheel_action, right_wheel_action in cases:
        env.reset()
        actions = torch.zeros_like(env.action_manager.action)
        actions[:, 0] = leg
        actions[:, 1] = leg
        actions[:, 2] = -leg
        actions[:, 3] = -leg
        actions[:, 4] = left_wheel_action
        actions[:, 5] = right_wheel_action

        vx_samples = []
        vy_samples = []
        yaw_samples = []
        pitch_samples = []
        done_count = 0

        for step in range(args_cli.steps):
            start_time = time.time()
            _, _, terminated, truncated, _ = env.step(actions)
            done = terminated | truncated

            if step >= args_cli.warmup_steps:
                vx_samples.append(robot.data.root_lin_vel_b[:, 0].detach())
                vy_samples.append(robot.data.root_lin_vel_b[:, 1].detach())
                yaw_samples.append(robot.data.root_ang_vel_b[:, 2].detach())
                pitch_samples.append(robot.data.root_ang_vel_b[:, 1].detach())
                done_count += int(done.sum().item())

            if args_cli.real_time:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        vx = torch.cat(vx_samples)
        vy = torch.cat(vy_samples)
        yaw = torch.cat(yaw_samples)
        pitch = torch.cat(pitch_samples)
        row = {
            "case": label,
            "left_wheel_action": left_wheel_action,
            "right_wheel_action": right_wheel_action,
            "vx_mean_m_s": float(vx.mean().item()),
            "vx_abs_mean_m_s": float(torch.abs(vx).mean().item()),
            "vy_mean_m_s": float(vy.mean().item()),
            "yaw_mean_rad_s": float(yaw.mean().item()),
            "yaw_abs_mean_rad_s": float(torch.abs(yaw).mean().item()),
            "pitch_abs_mean_rad_s": float(torch.abs(pitch).mean().item()),
            "done_count": done_count,
            "samples": int(vx.numel()),
        }
        rows.append(row)
        print(
            f"{label:14s} | vx={row['vx_mean_m_s']: .3f} m/s | vy={row['vy_mean_m_s']: .3f} "
            f"| yaw={row['yaw_mean_rad_s']: .3f} rad/s | dones={done_count}"
        )

    env.close()

    with open(args_cli.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[INFO] Wrote {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()
