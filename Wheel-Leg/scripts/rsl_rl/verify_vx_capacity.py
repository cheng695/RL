"""Measure fixed vx command tracking for a trained WheelLeg policy.

Run from the Wheel-Leg project root, for example:

    python scripts/rsl_rl/verify_vx_capacity.py \
        --load_run 2026-08-21_20-19-42 --checkpoint model_3000.pt \
        --vx_values -2,-1,-0.5,0,0.5,1,2 --num_envs 16

The script forces a constant vx/yaw/base_height command and prints achieved
forward velocity, yaw drift, and wheel-action statistics for each vx command.
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

parser = argparse.ArgumentParser(description="Verify vx tracking capacity with a trained checkpoint.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--vx_values", type=str, default="-2,-1,-0.5,0,0.5,1,2", help="Comma-separated vx commands in m/s.")
parser.add_argument("--yaw", type=float, default=0.0, help="Fixed yaw-rate command in rad/s.")
parser.add_argument("--command_height", type=float, default=0.31, help="Fixed base-height command in m.")
parser.add_argument("--steps", type=int, default=600, help="Policy steps per vx value.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Initial steps excluded from statistics.")
parser.add_argument("--output", type=str, default="vx_capacity.csv", help="Output CSV path.")
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


def _parse_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


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
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.yaw, args_cli.yaw)
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
    rows = []

    for vx_cmd in _parse_values(args_cli.vx_values):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if obs is None:
            obs = _get_obs(env)
        _force_commands(env, vx_cmd, args_cli.yaw, args_cli.command_height)
        obs = _get_obs(env)

        vx_samples = []
        yaw_samples = []
        left_wheel_action_samples = []
        right_wheel_action_samples = []
        done_count = 0

        for step in range(args_cli.steps):
            _force_commands(env, vx_cmd, args_cli.yaw, args_cli.command_height)
            obs = _get_obs(env)
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            _force_commands(env, vx_cmd, args_cli.yaw, args_cli.command_height)
            if hasattr(policy, "reset"):
                policy.reset(dones)

            if step < args_cli.warmup_steps:
                continue

            vx_samples.append(robot.data.root_lin_vel_b[:, 0].detach())
            yaw_samples.append(robot.data.root_ang_vel_b[:, 2].detach())
            left_wheel_action_samples.append(actions[:, 4].detach())
            right_wheel_action_samples.append(actions[:, 5].detach())
            done_count += int(dones.sum().item())

        actual_vx = torch.cat(vx_samples)
        actual_yaw = torch.cat(yaw_samples)
        left_wheel_action = torch.cat(left_wheel_action_samples)
        right_wheel_action = torch.cat(right_wheel_action_samples)
        wheel_sum = left_wheel_action + right_wheel_action
        wheel_diff = right_wheel_action - left_wheel_action
        vx_error = actual_vx - vx_cmd

        row = {
            "vx_cmd_m_s": vx_cmd,
            "actual_vx_mean_m_s": float(actual_vx.mean().item()),
            "actual_vx_std_m_s": float(actual_vx.std(unbiased=False).item()),
            "vx_error_rms_m_s": float(torch.sqrt(torch.mean(torch.square(vx_error))).item()),
            "actual_yaw_mean_rad_s": float(actual_yaw.mean().item()),
            "actual_yaw_abs_mean_rad_s": float(torch.abs(actual_yaw).mean().item()),
            "left_wheel_action_mean": float(left_wheel_action.mean().item()),
            "right_wheel_action_mean": float(right_wheel_action.mean().item()),
            "wheel_sum_mean": float(wheel_sum.mean().item()),
            "wheel_diff_mean": float(wheel_diff.mean().item()),
            "done_count": done_count,
            "samples": int(actual_vx.numel()),
        }
        rows.append(row)
        print(
            f"vx_cmd={vx_cmd: .2f} m/s | actual_vx={row['actual_vx_mean_m_s']: .2f} "
            f"| rms_err={row['vx_error_rms_m_s']: .2f} | yaw={row['actual_yaw_mean_rad_s']: .2f} "
            f"| wheel_sum={row['wheel_sum_mean']: .2f} | wheel_diff={row['wheel_diff_mean']: .2f} "
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
