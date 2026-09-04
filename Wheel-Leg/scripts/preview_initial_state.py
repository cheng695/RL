"""Open Isaac Sim and show the WheelLeg robot immediately after spawning."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview the WheelLeg spawned USD state.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
parser.add_argument("--reset", action="store_true", help="Run env.reset() before previewing.")
parser.add_argument("--zero_action_step", action="store_true", help="Step the environment with zero actions.")
parser.add_argument("--random_action_step", action="store_true", help="Step the environment with small random actions.")
parser.add_argument("--random_action_std", type=float, default=0.1, help="Standard deviation for random actions.")
parser.add_argument(
    "--fixed_action_step",
    action="store_true",
    help="Step the environment with a fixed 6D action.",
)
parser.add_argument(
    "--fixed_action",
    type=float,
    nargs=6,
    default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    metavar=("LF", "LR", "RF", "RR", "LW", "RW"),
    help="Fixed action values for 4 leg phase actions and 2 wheel velocity actions.",
)
parser.add_argument(
    "--direct_leg_position_action",
    action="store_true",
    help="Preview-only: interpret the first 4 fixed action values as direct leg joint position targets.",
)
parser.add_argument(
    "--direct_leg_position_mode",
    choices=("relative", "absolute"),
    default="relative",
    help="Use fixed leg values as default-relative rad offsets or absolute rad targets.",
)
parser.add_argument("--root_height", type=float, default=None, help="Override initial root/base spawn height in meters.")
parser.add_argument(
    "--hold_root",
    action="store_true",
    help="Preview-only: keep the robot root pose fixed in the air and zero root velocity every frame.",
)
parser.add_argument(
    "--disable_safety_terminations",
    action="store_true",
    help="Preview-only: disable height/velocity/joint safety terminations so suspended tests do not auto-reset.",
)
parser.add_argument("--print_interval", type=int, default=60, help="Print height diagnostics every N frames.")
parser.add_argument("--max_frames", type=int, default=None, help="Exit after this many rendered frames.")
parser.add_argument("--real_time", action="store_true", help="Throttle rendering to the environment step time.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import mdp
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401
from envs.wheel_leg.wheel_leg_env_cfg import LEG_JOINTS


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.root_height is not None:
        root_pos = list(env_cfg.scene.robot.init_state.pos)
        root_pos[2] = args_cli.root_height
        env_cfg.scene.robot.init_state.pos = tuple(root_pos)
    if args_cli.direct_leg_position_action:
        env_cfg.actions.leg_joint_positions = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEG_JOINTS,
            scale=1.0,
            use_default_offset=args_cli.direct_leg_position_mode == "relative",
            preserve_order=True,
        )
    if args_cli.disable_safety_terminations:
        for term_name in (
            "base_contact",
            "low_base_height",
            "leg_joint_deviation",
            "wheel_x_separation",
            "root_too_high",
            "root_linear_velocity_too_high",
            "root_angular_velocity_too_high",
            "joint_velocity_too_high",
            "root_state_non_finite",
        ):
            if hasattr(env_cfg.terminations, term_name):
                setattr(env_cfg.terminations, term_name, None)

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    step_enabled = args_cli.zero_action_step or args_cli.random_action_step or args_cli.fixed_action_step
    if args_cli.reset or step_enabled:
        env.reset()

    print("\nPreview is open. Close Isaac Sim to exit.")
    if args_cli.random_action_step:
        print(f"Running env.step() with random actions, std={args_cli.random_action_std}.")
    elif args_cli.fixed_action_step:
        print(f"Running env.step() with fixed action: {tuple(args_cli.fixed_action)}.")
        if args_cli.direct_leg_position_action:
            print(
                "Direct leg position preview is enabled. "
                f"First 4 action values are {args_cli.direct_leg_position_mode} leg joint targets in rad."
            )
    elif args_cli.zero_action_step:
        print("Running env.step() with zero actions.")
    elif args_cli.reset:
        print("Showing state after env.reset().")
    else:
        print("Showing spawned USD state without env.reset(). Use --reset if you want to preview reset events.")

    zero_actions = torch.zeros_like(env.action_manager.action)
    fixed_actions = torch.tensor(args_cli.fixed_action, device=zero_actions.device, dtype=zero_actions.dtype).repeat(
        zero_actions.shape[0], 1
    )
    robot = env.scene["robot"]
    held_root_pose = robot.data.root_state_w[:, :7].clone()
    if args_cli.root_height is not None:
        held_root_pose[:, 2] = args_cli.root_height
    zero_root_velocity = torch.zeros(zero_actions.shape[0], 6, device=zero_actions.device, dtype=zero_actions.dtype)
    generator = torch.Generator(device=zero_actions.device)
    generator.manual_seed(args_cli.seed)
    frame = 0
    dt = env.step_dt
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if args_cli.hold_root:
                robot.write_root_pose_to_sim(held_root_pose)
                robot.write_root_velocity_to_sim(zero_root_velocity)
            if args_cli.random_action_step:
                actions = torch.randn(
                    zero_actions.shape,
                    generator=generator,
                    device=zero_actions.device,
                    dtype=zero_actions.dtype,
                )
                actions = actions * args_cli.random_action_std
                env.step(actions)
            elif args_cli.fixed_action_step:
                env.step(fixed_actions)
            elif args_cli.zero_action_step:
                env.step(zero_actions)
            else:
                env.sim.render()
            if args_cli.hold_root:
                robot.write_root_pose_to_sim(held_root_pose)
                robot.write_root_velocity_to_sim(zero_root_velocity)

        frame += 1
        if step_enabled and args_cli.print_interval > 0 and (frame == 1 or frame % args_cli.print_interval == 0):
            root_height = robot.data.root_pos_w[:, 2]
            height_cmd = env.command_manager.get_command("base_height")[:, 0]
            height_error = torch.abs(height_cmd - root_height)
            print(
                f"[frame {frame}] "
                f"height={root_height.mean().item():.4f} "
                f"cmd={height_cmd.mean().item():.4f} "
                f"abs_error={height_error.mean().item():.4f}"
            )
        if args_cli.max_frames is not None and frame >= args_cli.max_frames:
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
