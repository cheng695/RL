"""Quick environment sanity check for the UZ-05 Isaac Lab task."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_SOURCE = PROJECT_ROOT / "source" / "my_robot_lab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a short UZ-05 environment sanity check.")
parser.add_argument("--task", type=str, default="MyRobot-Velocity-Flat-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps.")
parser.add_argument("--random_actions", action="store_true", default=False, help="Use random actions instead of zeros.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaacsim.core.utils.extensions import enable_extension

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import my_robot_lab  # noqa: F401

enable_extension("isaacsim.asset.importer.mjcf")


def _term_names(manager) -> list[str]:
    names = getattr(manager, "active_terms", None)
    if names is None:
        names = getattr(manager, "terms", [])
    return list(names)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]

    print(f"[INFO] task: {args_cli.task}")
    print(f"[INFO] num_envs: {env.unwrapped.num_envs}")
    print(f"[INFO] step_dt: {env.unwrapped.step_dt}")
    print(f"[INFO] action_space: {env.action_space}")
    print(f"[INFO] observation type: {type(obs).__name__}")
    print(f"[INFO] action terms: {_term_names(env.unwrapped.action_manager)}")
    print(f"[INFO] command terms: {_term_names(env.unwrapped.command_manager)}")
    print(f"[INFO] reward terms: {_term_names(env.unwrapped.reward_manager)}")
    print(f"[INFO] termination terms: {_term_names(env.unwrapped.termination_manager)}")
    print(f"[INFO] body names: {robot.body_names}")
    print(f"[INFO] joint names: {robot.joint_names}")

    action_dim = env.unwrapped.action_manager.total_action_dim
    for step in range(args_cli.steps):
        if args_cli.random_actions:
            actions = 2.0 * torch.rand((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device) - 1.0
        else:
            actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)

        _, rewards, terminated, truncated, _ = env.step(actions)
        if step % 20 == 0 or step == args_cli.steps - 1:
            root_z = robot.data.root_pos_w[:, 2]
            tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
            done = torch.logical_or(terminated, truncated)
            print(
                f"[step {step:04d}] "
                f"reward_mean={rewards.mean().item():+.3f} "
                f"z_mean={root_z.mean().item():.3f} "
                f"tilt_max={tilt.max().item():.3f} "
                f"done_count={int(done.sum().item())}"
            )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
