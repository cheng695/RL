"""Train UZ-05 velocity tasks with RSL-RL."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = PROJECT_ROOT / "source" / "my_robot_lab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Train a UZ-05 velocity task with RSL-RL.")
parser.add_argument("--task", type=str, default="MyRobot-Velocity-Flat-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed. Use -1 for a random seed.")
parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO training iterations.")
parser.add_argument(
    "--reset_noise_std", type=float, default=None,
    help="Reset policy exploration std after loading a checkpoint (e.g. 0.5).",
)
parser.add_argument(
    "--height_only", action="store_true",
    help="Train fixed-command height tracking with zero velocity commands and no COM/push randomization.",
)
parser.add_argument("--video", action="store_true", help="Record training videos periodically.")
parser.add_argument(
    "--video_interval", type=int, default=500,
    help="Record one video every N PPO iterations (requires --video).",
)
parser.add_argument(
    "--video_length", type=int, default=500,
    help="Length of each training video in simulation steps.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.reset_noise_std is not None and (
    not math.isfinite(args_cli.reset_noise_std) or args_cli.reset_noise_std <= 0
):
    parser.error("--reset_noise_std must be finite and positive")
if args_cli.video and args_cli.video_interval <= 0:
    parser.error("--video_interval must be positive")
if args_cli.video and args_cli.video_length <= 0:
    parser.error("--video_length must be positive")
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.mjcf")

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import my_robot_lab  # noqa: F401

def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    installed_version = metadata.version("rsl-rl-lib")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    if args_cli.height_only:
        velocity_cfg = env_cfg.commands.base_velocity
        velocity_cfg.rel_standing_envs = 1.0
        velocity_cfg.heading_command = False
        velocity_cfg.ranges.lin_vel_x = (0.0, 0.0)
        velocity_cfg.ranges.lin_vel_y = (0.0, 0.0)
        velocity_cfg.ranges.ang_vel_z = (0.0, 0.0)
        velocity_cfg.straight_command_prob = 1.0
        velocity_cfg.turn_command_prob = 0.0
        velocity_cfg.mixed_command_prob = 0.0
        velocity_cfg.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_height.height_range = (0.25, 0.35)
        env_cfg.commands.base_height.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_height.endpoint_switch_prob = 0.0
        env_cfg.events.add_base_mass = None
        env_cfg.events.base_com = None
        env_cfg.events.base_external_force_torque = None
        env_cfg.events.push_robot = None

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    resume_path = None
    if agent_cfg.resume:
        if os.path.isabs(agent_cfg.load_checkpoint) or os.path.isfile(agent_cfg.load_checkpoint):
            if not os.path.isfile(agent_cfg.load_checkpoint):
                raise FileNotFoundError(agent_cfg.load_checkpoint)
            resume_path = os.path.abspath(agent_cfg.load_checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if resume_path is not None:
        from checkpoint_mass_settings import restore_checkpoint_mass_settings
        for message in restore_checkpoint_mass_settings(env_cfg, resume_path):
            print(message)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        # RecordVideo counts env.step() calls, while the user-facing interval is
        # expressed in PPO iterations. One iteration contains num_steps_per_env
        # rollout steps, so convert once after the runner config is available.
        video_interval_steps = args_cli.video_interval * agent_cfg.num_steps_per_env
        video_folder = os.path.join(log_dir, "videos", "train")
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step % video_interval_steps == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording videos every {args_cli.video_interval} PPO iterations.")
        print(f"[INFO] Video folder: {video_folder}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    from weighted_metrics import install_weighted_metrics

    install_weighted_metrics(runner.logger)
    runner.add_git_repo_to_log(__file__)

    if resume_path is not None:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
        if getattr(env_cfg, "terrain_relative_height", False):
            runner.alg.learning_rate = agent_cfg.algorithm.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = runner.alg.learning_rate
    if args_cli.reset_noise_std is not None:
        policy = runner.alg.get_policy()
        noise_parameters = [
            (name, parameter) for name, parameter in policy.named_parameters()
            if name.split(".")[-1] in ("std_param", "log_std_param")
        ]
        if not noise_parameters:
            raise RuntimeError("Policy has no supported state-independent noise parameter")
        with torch.no_grad():
            for name, parameter in noise_parameters:
                value = args_cli.reset_noise_std
                parameter.fill_(math.log(value) if name.endswith("log_std_param") else value)
                runner.alg.optimizer.state.pop(parameter, None)
        print(f"[INFO] Reset policy exploration std to {args_cli.reset_noise_std}")

    env.unwrapped.command_manager.get_term("base_velocity").cfg.action_limit = agent_cfg.clip_actions or float("inf")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(os.path.join(log_dir, "params", "launch.yaml"), {
        "reset_noise_std": args_cli.reset_noise_std,
        "video": args_cli.video,
        "video_interval": args_cli.video_interval,
        "video_length": args_cli.video_length,
        "height_only": args_cli.height_only,
    })

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"[INFO] Training time: {time.time() - start_time:.2f} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
