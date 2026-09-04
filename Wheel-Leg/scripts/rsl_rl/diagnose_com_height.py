"""Diagnose whole-robot COM offset relative to the wheel axis at fixed heights.

The script loads a trained policy, forces zero vx/yaw and a fixed base-height
command, then records whether high posture moves the robot COM behind the wheel
axis.  It does not change the environment observation, action, reward, or PPO
configuration.
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

parser = argparse.ArgumentParser(description="Diagnose COM-vs-wheel-axis offset for WheelLeg policies.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--height_values",
    type=str,
    default="0.27,0.31,0.35,0.40,0.45",
    help="Comma-separated fixed base-height commands in m.",
)
parser.add_argument("--steps", type=int, default=600, help="Policy steps per height value.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Initial steps excluded from statistics.")
parser.add_argument("--output", type=str, default="com_height_diagnosis", help="Output prefix for csv/png files.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_apply_inverse
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401


def _parse_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _get_obs(env):
    obs = env.get_observations()
    if isinstance(obs, dict):
        return obs["policy"] if "policy" in obs else next(iter(obs.values()))
    return obs


def _force_commands(env, height: float) -> None:
    command_manager = env.unwrapped.command_manager
    base_velocity = command_manager.get_command("base_velocity")
    base_velocity[:, 0] = 0.0
    base_velocity[:, 1] = 0.0
    base_velocity[:, 2] = 0.0

    base_height = command_manager.get_command("base_height")
    base_height[:, 0] = height
    if base_height.shape[1] > 1:
        base_height[:, 1] = 0.0


def _mean_std(values: torch.Tensor) -> tuple[float, float]:
    return float(values.mean().item()), float(values.std(unbiased=False).item())


def _whole_robot_com_w(robot) -> torch.Tensor:
    masses = robot.data.default_mass
    body_com_pos_w = robot.data.body_com_pos_w
    if masses is None:
        raise RuntimeError("robot.data.default_mass is not available; cannot compute whole-robot COM.")
    masses = masses.to(device=body_com_pos_w.device, dtype=body_com_pos_w.dtype)
    weights = masses.unsqueeze(-1)
    return torch.sum(body_com_pos_w * weights, dim=1) / torch.sum(weights, dim=1)


def main() -> None:
    installed_version = metadata.version("rsl-rl-lib")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
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
    wheel_body_ids = robot.find_bodies(["Left_Wheel_link", "Right_Wheel_link"], preserve_order=True)[0]
    wheel_joint_ids = robot.find_joints(["Left_Wheel_joint", "Right_Wheel_joint"], preserve_order=True)[0]
    dt = env.unwrapped.step_dt

    summary_rows = []
    trace_rows = []

    for height_cmd in _parse_values(args_cli.height_values):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if obs is None:
            obs = _get_obs(env)
        _force_commands(env, height_cmd)
        obs = _get_obs(env)

        previous_wheel_action = None
        samples: dict[str, list[torch.Tensor]] = {
            "height": [],
            "height_error": [],
            "com_minus_wheel_x_w": [],
            "com_minus_wheel_x_b": [],
            "root_x_minus_wheel_x_w": [],
            "root_vx_b": [],
            "root_vy_b": [],
            "root_yaw_rate_b": [],
            "wheel_action_left": [],
            "wheel_action_right": [],
            "wheel_action_rate_l2": [],
            "wheel_torque_left": [],
            "wheel_torque_right": [],
            "wheel_torque_abs_max": [],
        }
        done_count = 0

        for step in range(args_cli.steps):
            _force_commands(env, height_cmd)
            obs = _get_obs(env)
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            _force_commands(env, height_cmd)
            if hasattr(policy, "reset"):
                policy.reset(dones)
            done_count += int(dones.sum().item())

            wheel_action = actions[:, 4:6].detach()
            if previous_wheel_action is None:
                wheel_action_rate_l2 = torch.zeros(actions.shape[0], device=actions.device)
            else:
                wheel_action_rate_l2 = torch.sum(torch.square(wheel_action - previous_wheel_action), dim=1)
            previous_wheel_action = wheel_action.clone()

            if step < args_cli.warmup_steps:
                continue

            com_pos_w = _whole_robot_com_w(robot)
            wheel_center_w = torch.mean(robot.data.body_pos_w[:, wheel_body_ids, :], dim=1)
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            com_minus_wheel_b = quat_apply_inverse(root_quat_w, com_pos_w - wheel_center_w)
            torque = robot.data.applied_torque[:, wheel_joint_ids].detach()

            values = {
                "height": robot.data.root_pos_w[:, 2].detach(),
                "height_error": (robot.data.root_pos_w[:, 2] - height_cmd).detach(),
                "com_minus_wheel_x_w": (com_pos_w[:, 0] - wheel_center_w[:, 0]).detach(),
                "com_minus_wheel_x_b": com_minus_wheel_b[:, 0].detach(),
                "root_x_minus_wheel_x_w": (root_pos_w[:, 0] - wheel_center_w[:, 0]).detach(),
                "root_vx_b": robot.data.root_lin_vel_b[:, 0].detach(),
                "root_vy_b": robot.data.root_lin_vel_b[:, 1].detach(),
                "root_yaw_rate_b": robot.data.root_ang_vel_b[:, 2].detach(),
                "wheel_action_left": wheel_action[:, 0],
                "wheel_action_right": wheel_action[:, 1],
                "wheel_action_rate_l2": wheel_action_rate_l2.detach(),
                "wheel_torque_left": torque[:, 0],
                "wheel_torque_right": torque[:, 1],
                "wheel_torque_abs_max": torch.max(torch.abs(torque), dim=1).values,
            }
            for name, value in values.items():
                samples[name].append(value)

            env0 = {name: float(value[0].item()) for name, value in values.items()}
            env0.update({"height_cmd": height_cmd, "step": step, "t": round(step * dt, 5), "done": bool(dones[0])})
            trace_rows.append(env0)

        row = {"height_cmd": height_cmd, "done_count": done_count}
        for name, tensors in samples.items():
            if not tensors:
                continue
            stacked = torch.cat(tensors)
            mean, std = _mean_std(stacked)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
            row[f"{name}_abs_mean"] = float(torch.abs(stacked).mean().item())
        summary_rows.append(row)

        print(
            f"h_cmd={height_cmd:.3f} | h={row['height_mean']:.3f} "
            f"| COMx-wheelx_b={row['com_minus_wheel_x_b_mean']:+.4f} m "
            f"| vx={row['root_vx_b_mean']:+.3f} m/s "
            f"| wheel_rate_l2={row['wheel_action_rate_l2_mean']:.4f} "
            f"| tau=({row['wheel_torque_left_mean']:+.2f},{row['wheel_torque_right_mean']:+.2f}) Nm "
            f"| tau_abs={row['wheel_torque_abs_max_mean']:.2f} Nm "
            f"| dones={done_count}"
        )

    env.close()

    summary_path = f"{args_cli.output}_summary.csv"
    trace_path = f"{args_cli.output}_trace.csv"
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(trace_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(trace_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trace_rows)
    print(f"\n[INFO] Wrote {summary_path}")
    print(f"[INFO] Wrote {trace_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipped PNG plot.")
        return

    height_cmds = [row["height_cmd"] for row in summary_rows]
    com_x_b = [row["com_minus_wheel_x_b_mean"] for row in summary_rows]
    com_x_b_std = [row["com_minus_wheel_x_b_std"] for row in summary_rows]
    vx_b = [row["root_vx_b_mean"] for row in summary_rows]
    vx_b_std = [row["root_vx_b_std"] for row in summary_rows]
    wheel_rate = [row["wheel_action_rate_l2_mean"] for row in summary_rows]
    torque_abs = [row["wheel_torque_abs_max_mean"] for row in summary_rows]
    torque_left = [row["wheel_torque_left_mean"] for row in summary_rows]
    torque_right = [row["wheel_torque_right_mean"] for row in summary_rows]
    height_error_abs = [row["height_error_abs_mean"] for row in summary_rows]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    axes[0].errorbar(height_cmds, com_x_b, yerr=com_x_b_std, marker="o", capsize=3)
    axes[0].axhline(0.0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("COM_x - wheel_x [m]\n(base frame)")
    axes[0].set_title("COM offset vs commanded height")

    axes[1].errorbar(height_cmds, vx_b, yerr=vx_b_std, marker="o", capsize=3, color="tab:orange")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("base vx [m/s]")
    axes[1].set_title("Drift velocity vs commanded height")

    axes[2].plot(height_cmds, height_error_abs, marker="o", color="tab:green", label="|height error|")
    axes[2].set_ylabel("|h - h_cmd| [m]")
    axes[2].set_title("Height tracking residual")

    axes[3].plot(height_cmds, wheel_rate, marker="o", label="wheel action-rate L2", color="tab:red")
    axes[3].plot(height_cmds, torque_abs, marker="s", label="wheel torque abs max [Nm]", color="tab:purple")
    axes[3].set_ylabel("wheel diagnostics")
    axes[3].set_xlabel("commanded base height [m]")
    axes[3].set_title("Wheel chattering / torque usage")
    axes[3].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    png_path = f"{args_cli.output}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[INFO] Wrote {png_path}")

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes2[0].plot(height_cmds, torque_left, marker="o", label="left wheel torque mean")
    axes2[0].plot(height_cmds, torque_right, marker="o", label="right wheel torque mean")
    axes2[0].axhline(5.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[0].axhline(-5.0, color="gray", linewidth=0.8, linestyle="--")
    axes2[0].set_ylabel("torque [Nm]")
    axes2[0].set_title("Wheel torque mean vs commanded height")
    axes2[0].legend()
    axes2[1].plot(height_cmds, torque_abs, marker="s", color="tab:purple", label="max |wheel torque|")
    axes2[1].axhline(5.0, color="gray", linewidth=0.8, linestyle="--", label="effort limit 5 Nm")
    axes2[1].set_ylabel("|torque| [Nm]")
    axes2[1].set_xlabel("commanded base height [m]")
    axes2[1].set_title("Wheel torque saturation check")
    axes2[1].legend()
    for axis in axes2:
        axis.grid(True, alpha=0.3)
    fig2.tight_layout()
    torque_png_path = f"{args_cli.output}_wheel_torque.png"
    fig2.savefig(torque_png_path, dpi=150)
    print(f"[INFO] Wrote {torque_png_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
