"""Measure fixed yaw-rate command tracking for a trained WheelLeg policy.

Run from the Wheel-Leg project root, for example:

    python scripts/rsl_rl/verify_yaw_capacity.py \
        --load_run 2026-08-21_17-23-23 --checkpoint model_2000.pt \
        --yaw_values 0,0.5,1,1.5,2,3,4,5 --num_envs 16

The script forces a constant vx/yaw/base_height command during rollout and
prints the achieved yaw-rate statistics for each tested command.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Verify yaw-rate tracking capacity with a trained checkpoint.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--yaw_values", type=str, default="0,0.5,1,1.5,2,3,4,5,6", help="Comma-separated yaw rates in rad/s.")
parser.add_argument("--vx", type=float, default=0.0, help="Fixed forward velocity command in m/s.")
parser.add_argument("--command_height", type=float, default=0.31, help="Fixed base-height command in m.")
parser.add_argument("--steps", type=int, default=600, help="Policy steps per yaw value.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Initial steps excluded from statistics.")
parser.add_argument("--both_directions", action="store_true", default=False, help="Also test negative yaw values.")
parser.add_argument("--output", type=str, default="yaw_capacity.csv", help="Output CSV path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401


def _parse_yaw_values(text: str, both_directions: bool) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if both_directions:
        signed_values = []
        for value in values:
            if value != 0.0:
                signed_values.append(-abs(value))
            signed_values.append(abs(value))
        values = signed_values
    return values


def _force_commands(env, vx: float, yaw: float, height: float) -> None:
    command_manager = env.unwrapped.command_manager

    base_velocity = command_manager.get_command("base_velocity")
    base_velocity[:, 0] = vx
    base_velocity[:, 1] = 0.0
    base_velocity[:, 2] = yaw

    base_height = command_manager.get_command("base_height")
    base_height[:, 0] = height
    if base_height.shape[1] > 1:
        base_height[:, 1] = 0.0


def _get_obs(env):
    obs = env.get_observations()
    if isinstance(obs, dict):
        return obs["policy"] if "policy" in obs else next(iter(obs.values()))
    return obs


def main() -> None:
    installed_version = metadata.version("rsl-rl-lib")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_height.height_range = (args_cli.command_height, args_cli.command_height)
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint_path = Path(args_cli.checkpoint)
        if checkpoint_path.is_absolute():
            resume_path = str(checkpoint_path)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    yaw_values = _parse_yaw_values(args_cli.yaw_values, args_cli.both_directions)
    rows = []

    for yaw_cmd in yaw_values:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if obs is None:
            obs = _get_obs(env)
        _force_commands(env, args_cli.vx, yaw_cmd, args_cli.command_height)
        obs = _get_obs(env)

        actual_samples = []
        error_samples = []
        abs_error_samples = []
        done_count = 0
        height_samples = []
        roll_pitch_samples = []
        left_wheel_action_samples = []
        right_wheel_action_samples = []
        wheel_action_diff_samples = []

        for step in range(args_cli.steps):
            _force_commands(env, args_cli.vx, yaw_cmd, args_cli.command_height)
            obs = _get_obs(env)
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)
            _force_commands(env, args_cli.vx, yaw_cmd, args_cli.command_height)

            if step < args_cli.warmup_steps:
                continue

            actual_yaw = robot.data.root_ang_vel_b[:, 2].detach()
            error = actual_yaw - yaw_cmd
            actual_samples.append(actual_yaw)
            error_samples.append(error)
            abs_error_samples.append(torch.abs(error))
            height_samples.append(robot.data.root_pos_w[:, 2].detach())
            roll_pitch_samples.append(torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1).detach())
            left_wheel_action = actions[:, 4].detach()
            right_wheel_action = actions[:, 5].detach()
            left_wheel_action_samples.append(left_wheel_action)
            right_wheel_action_samples.append(right_wheel_action)
            wheel_action_diff_samples.append(right_wheel_action - left_wheel_action)
            done_count += int(dones.sum().item())

        actual = torch.cat(actual_samples)
        error = torch.cat(error_samples)
        abs_error = torch.cat(abs_error_samples)
        height = torch.cat(height_samples)
        roll_pitch = torch.cat(roll_pitch_samples)
        left_wheel_action = torch.cat(left_wheel_action_samples)
        right_wheel_action = torch.cat(right_wheel_action_samples)
        wheel_action_diff = torch.cat(wheel_action_diff_samples)

        row = {
            "yaw_cmd_rad_s": yaw_cmd,
            "actual_yaw_mean_rad_s": float(actual.mean().item()),
            "actual_yaw_std_rad_s": float(actual.std(unbiased=False).item()),
            "actual_yaw_abs_p95_rad_s": float(torch.quantile(torch.abs(actual), 0.95).item()),
            "yaw_error_mean_rad_s": float(error.mean().item()),
            "yaw_error_abs_mean_rad_s": float(abs_error.mean().item()),
            "yaw_error_rms_rad_s": float(torch.sqrt(torch.mean(torch.square(error))).item()),
            "tracking_ratio": float(actual.mean().item() / yaw_cmd) if abs(yaw_cmd) > 1.0e-6 else 1.0,
            "left_wheel_action_mean": float(left_wheel_action.mean().item()),
            "right_wheel_action_mean": float(right_wheel_action.mean().item()),
            "wheel_action_diff_mean": float(wheel_action_diff.mean().item()),
            "wheel_action_diff_abs_mean": float(torch.abs(wheel_action_diff).mean().item()),
            "base_height_mean_m": float(height.mean().item()),
            "roll_pitch_l2_mean": float(roll_pitch.mean().item()),
            "done_count": done_count,
            "samples": int(actual.numel()),
        }
        rows.append(row)
        print(
            f"yaw_cmd={yaw_cmd: .2f} rad/s | actual_mean={row['actual_yaw_mean_rad_s']: .2f} "
            f"| rms_err={row['yaw_error_rms_rad_s']: .2f} | abs_err={row['yaw_error_abs_mean_rad_s']: .2f} "
            f"| ratio={row['tracking_ratio']: .2f} | wheel_diff={row['wheel_action_diff_mean']: .2f} "
            f"| dones={done_count}"
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
