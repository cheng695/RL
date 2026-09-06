"""Play a trained UZ-05 RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = PROJECT_ROOT / "source" / "my_robot_lab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a trained UZ-05 RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="MyRobot-Velocity-Flat-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Try to run at real-time speed.")
parser.add_argument("--command_vx", type=float, default=None, help="Override forward velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Override yaw-rate command in rad/s.")
parser.add_argument("--print_state", action="store_true", default=False, help="Print env0 state while playing.")
parser.add_argument("--print_interval_s", type=float, default=0.5, help="Seconds between state prints.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaacsim.core.utils.extensions import enable_extension
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import my_robot_lab  # noqa: F401

enable_extension("isaacsim.asset.importer.mjcf")


def _force_velocity_command(env, vx: float | None, yaw: float | None) -> None:
    command = env.unwrapped.command_manager.get_command("base_velocity")
    if vx is not None:
        command[:, 0] = vx
    command[:, 1] = 0.0
    if yaw is not None:
        command[:, 2] = yaw


def _print_state(env, step_count: int) -> None:
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")[0]
    root_pos_w = robot.data.root_pos_w[0]
    root_lin_vel_b = robot.data.root_lin_vel_b[0]
    root_ang_vel_b = robot.data.root_ang_vel_b[0]
    projected_gravity_b = robot.data.projected_gravity_b[0]
    tilt = torch.acos(torch.clamp(-projected_gravity_b[2], -1.0, 1.0))
    print(
        f"[step {step_count:06d}] "
        f"cmd=({command[0].item():+.2f}, {command[1].item():+.2f}, {command[2].item():+.2f}) "
        f"pos_z={root_pos_w[2].item():.3f} "
        f"vel_b=({root_lin_vel_b[0].item():+.2f}, {root_lin_vel_b[1].item():+.2f}, {root_lin_vel_b[2].item():+.2f}) "
        f"yaw_rate={root_ang_vel_b[2].item():+.2f} "
        f"tilt={tilt.item():.3f}"
    )


def main() -> None:
    installed_version = metadata.version("rsl-rl-lib")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    step_count = 0
    next_print_time = 0.0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            _force_velocity_command(env, args_cli.command_vx, args_cli.command_yaw)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        if args_cli.print_state and time.time() >= next_print_time:
            _print_state(env, step_count)
            next_print_time = time.time() + args_cli.print_interval_s

        step_count += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
