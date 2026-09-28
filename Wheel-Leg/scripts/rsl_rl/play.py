"""Play a trained UZ-05 RSL-RL checkpoint in the Isaac Sim window."""

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
parser.add_argument(
    "--terrain_column", type=int, default=None,
    help="Spawn on a chosen generator terrain column (zero-based); preserves the complete map.",
)
parser.add_argument("--terrain_level", type=int, default=None, help="Fixed generator row for terrain evaluation (zero-based).")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Try to run at real-time speed.")
parser.add_argument("--command_vx", type=float, default=None, help="Override forward velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Override yaw-rate command in rad/s.")
parser.add_argument("--command_height", type=float, default=None, help="Override target base height in meters.")
parser.add_argument("--print_state", action="store_true", default=False, help="Print env0 state while playing.")
parser.add_argument("--print_interval_s", type=float, default=0.5, help="Seconds between state prints.")
parser.add_argument("--no_keyboard", action="store_true", help="Disable WASD keyboard interaction.")
parser.add_argument("--no_hud", action="store_true", help="Disable the Isaac Sim HUD window.")
parser.add_argument("--no_follow_camera", action="store_true", help="Disable the third-person follow camera.")
parser.add_argument("--follow_camera", action="store_true", help="Opt in to the third-person follow camera.")
parser.add_argument("--auto_reset", action="store_true", default=True, help="Enable automatic episode resets (default).")
parser.add_argument("--no_auto_reset", dest="auto_reset", action="store_false", help="Disable automatic resets; use R manually.")
parser.add_argument("--no_command_smoothing", action="store_true", help="Disable velocity/yaw slew-rate limiting.")
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

from camera_controller import FollowCamera
from command_manager import CommandManager
from hud import RobotHUD
from keyboard_controller import KeyboardController


def _force_velocity_command(env, vx: float | None, yaw: float | None):
    """Apply CLI velocity overrides and the same wheel-workspace projection as training."""
    term = env.unwrapped.command_manager.get_term("base_velocity")
    command = term.command
    if vx is not None:
        command[:, 0] = vx
    command[:, 1] = 0.0
    if yaw is not None:
        command[:, 2] = yaw
    term._project_to_wheel_speed_limit(slice(None))
    if hasattr(term, "travel_sign"):
        sign = torch.sign(command[:, 0])
        changed = (sign != 0) & (sign != term.travel_sign)
        term.hold[changed] = 0
        term.travel_sign[:] = torch.where(sign != 0, sign, term.travel_sign)
    return command[0].detach().clone()


def _print_state(env, step_count: int) -> None:
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")[0]
    root_pos_w = robot.data.root_pos_w[0]
    root_lin_vel_b = robot.data.root_lin_vel_b[0]
    root_ang_vel_b = robot.data.root_ang_vel_b[0]
    projected_gravity_b = robot.data.projected_gravity_b[0]
    tilt = torch.acos(torch.clamp(-projected_gravity_b[2], -1.0, 1.0))
    print(
        f"[step {step_count:06d}] cmd=({command[0].item():+.2f}, {command[1].item():+.2f}, {command[2].item():+.2f}) "
        f"pos_z={root_pos_w[2].item():.3f} "
        f"vel_b=({root_lin_vel_b[0].item():+.2f}, {root_lin_vel_b[1].item():+.2f}, {root_lin_vel_b[2].item():+.2f}) "
        f"yaw_rate={root_ang_vel_b[2].item():+.2f} tilt={tilt.item():.3f}"
    )


def _euler_xyz_from_quat_wxyz(quat):
    """Return roll, pitch, yaw from Isaac Lab's (w, x, y, z) quaternion."""
    w, x, y, z = [float(value) for value in quat]
    roll = torch.atan2(torch.tensor(2.0 * (w * x + y * z)), torch.tensor(1.0 - 2.0 * (x * x + y * y)))
    pitch_arg = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = torch.asin(torch.tensor(pitch_arg))
    yaw = torch.atan2(torch.tensor(2.0 * (w * z + x * y)), torch.tensor(1.0 - 2.0 * (y * y + z * z)))
    return float(roll), float(pitch), float(yaw)


def _hud_text(env, command_state, effective_velocity, terrain_info: str) -> str:
    robot = env.unwrapped.scene["robot"]
    root = robot.data.root_pos_w[0]
    height_term = env.unwrapped.command_manager.get_term("base_height")
    body_ids, _ = robot.find_bodies(height_term.cfg.body_name)
    ground = height_term.reference_ground_height()[0]
    body_z = robot.data.body_pos_w[0, body_ids[0], 2]
    relative_height = body_z - ground
    linear = robot.data.root_lin_vel_b[0]
    angular = robot.data.root_ang_vel_b[0]
    roll, pitch, yaw = _euler_xyz_from_quat_wxyz(robot.data.root_quat_w[0].detach().cpu())
    return (
        "UZ-05 Interactive Play\n\n"
        "Command\n"
        f"target vx     {command_state.vx_target:+.2f} m/s\n"
        f"target yaw    {command_state.yaw_target:+.2f} rad/s\n"
        f"effective vx  {float(effective_velocity[0]):+.2f} m/s\n"
        f"effective yaw {float(effective_velocity[2]):+.2f} rad/s\n"
        f"height target  {float(height_term.command[0, 0]):.3f} m\n"
        f"ascent phase   {int(height_term.phase[0]) if hasattr(height_term, 'phase') else 0}\n"
        f"v_set/w_set    {command_state.v_set:.2f} / {command_state.w_set:.2f}\n\n"
        "Robot\n"
        f"vx / vy        {float(linear[0]):+.2f} / {float(linear[1]):+.2f} m/s\n"
        f"yaw rate       {float(angular[2]):+.2f} rad/s\n"
        f"roll / pitch   {roll * 57.2958:+.1f} / {pitch * 57.2958:+.1f} deg\n"
        f"height local   {float(relative_height):.3f} m\n"
        f"height error   {float(relative_height - height_term.command[0, 0]):+.3f} m\n"
        f"ground Z       {float(ground):.3f} m\n"
        f"base world Z   {float(body_z):.3f} m\n\n"
        f"Terrain\n{terrain_info}\n\n"
        "W/S move  A/D turn  Up/Down v  Left/Right yaw\n"
        "I/K height  H default  Space stop  R reset"
    )


def _terrain_info(env) -> str:
    try:
        terrain = env.unwrapped.scene.terrain
        terrain_type = int(terrain.terrain_types[0].item()) if terrain.terrain_types is not None else -1
        terrain_level = int(terrain.terrain_levels[0].item()) if terrain.terrain_levels is not None else -1
        return f"type index     {terrain_type}\nlevel          {terrain_level}"
    except Exception:
        return "unavailable"


@torch.inference_mode()
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
    if not args_cli.auto_reset:
        # ManagerBasedRLEnv resets when a termination term fires. Disable these
        # only on this play config; explicit env.reset() still works normally.
        for name in vars(env_cfg.terminations):
            if not name.startswith("_"):
                setattr(env_cfg.terminations, name, None)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if args_cli.terrain_column is not None or args_cli.terrain_level is not None:
        terrain = env.unwrapped.scene.terrain
        if terrain.terrain_origins is None:
            raise ValueError("--terrain_column requires a generated terrain task")
        columns = terrain.terrain_origins.shape[1]
        if args_cli.terrain_column is not None and not 0 <= args_cli.terrain_column < columns:
            raise ValueError(f"--terrain_column must be between 0 and {columns - 1}")
        if args_cli.terrain_column is not None:
            terrain.terrain_types[:] = args_cli.terrain_column
        if args_cli.terrain_level is not None:
            if not 0 <= args_cli.terrain_level < terrain.terrain_origins.shape[0]:
                raise ValueError("--terrain_level outside generated rows")
            terrain.terrain_levels[:] = args_cli.terrain_level
        ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
        no_move = torch.zeros_like(ids, dtype=torch.bool)
        terrain.update_env_origins(ids, no_move, no_move)
        env.reset()
        print(f"[INFO] Spawn terrain column: {args_cli.terrain_column}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    step_count = 0
    next_print_time = 0.0

    default_height = float(env.unwrapped.command_manager.get_command("base_height")[0, 0].item())
    commands = CommandManager(
        default_height=default_height,
        height_min=env_cfg.commands.base_height.height_range[0],
        height_max=env_cfg.commands.base_height.height_range[1],
        smooth=not args_cli.no_command_smoothing,
        velocity_max=1.0,
        yaw_max=5.0,
    )
    commands.state.v_set = min(0.3, max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x))
    if args_cli.command_vx is not None:
        commands.state.v_set = max(0.1, abs(args_cli.command_vx))
    if args_cli.command_yaw is not None:
        commands.state.w_set = max(0.1, abs(args_cli.command_yaw))
    commands.update_settings()
    print(f"[INFO] Keyboard speed limits: vx={commands.velocity_max:.2f} m/s, yaw={commands.yaw_max:.2f} rad/s")
    for name, value, maximum in (("command_vx", args_cli.command_vx,
                                  max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x)),
                                 ("command_yaw", args_cli.command_yaw,
                                  max(abs(v) for v in env_cfg.commands.base_velocity.ranges.ang_vel_z))):
        if value is not None and abs(value) > maximum:
            print(f"[WARNING] Explicit --{name}={value} exceeds task training range +/-{maximum}; stability is unvalidated.")
    keyboard = None if args_cli.no_keyboard else KeyboardController(commands)
    hud = None if args_cli.no_hud else RobotHUD()
    follow_camera = FollowCamera(env) if args_cli.follow_camera and not args_cli.no_follow_camera else None
    print(f"[INFO] Automatic reset: {args_cli.auto_reset}. Press R to reset manually.")
    print("[INFO] Keyboard: W/S move, A/D turn, arrows tune speeds, I/K height, H default, Space stop, R reset.")

    print("[INFO] Play started in the Isaac Sim window.")
    while simulation_app.is_running():
        start_time = time.time()
        if commands.state.reset_requested:
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            commands.state.reset_requested = False
            commands.state.vx_cmd = 0.0
            commands.state.yaw_rate_cmd = 0.0
            if keyboard is not None:
                keyboard.pressed.clear()
            if hasattr(policy, "reset"):
                policy.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device))
            step_count = 0

        with torch.inference_mode():
            if keyboard is not None:
                command_state = commands.advance(dt, keyboard.pressed)
            else:
                command_state = commands.advance(dt, set())
            vx = args_cli.command_vx if args_cli.command_vx is not None else command_state.vx_cmd
            yaw = args_cli.command_yaw if args_cli.command_yaw is not None else command_state.yaw_rate_cmd
            if args_cli.command_vx is not None:
                command_state.vx_target = vx
            if args_cli.command_yaw is not None:
                command_state.yaw_target = yaw
            if command_state.emergency_stop:
                vx = yaw = 0.0
                command_state.vx_target = command_state.yaw_target = 0.0
            effective_velocity = _force_velocity_command(env, vx, yaw)
            height = args_cli.command_height if args_cli.command_height is not None else command_state.base_height_cmd
            if args_cli.command_height is not None:
                command_state.base_height_cmd = height
            env.unwrapped.command_manager.get_command("base_height")[:, 0] = height
            height_term = env.unwrapped.command_manager.get_term("base_height")
            if hasattr(height_term, "apply_assist"):
                height_term.nominal[:, 0] = height
                height_term.apply_assist()
            obs = env.get_observations()
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        if follow_camera is not None and step_count % 2 == 0:
            follow_camera.update()
        if hud is not None and step_count % 2 == 0:
            hud.update(_hud_text(env, command_state, effective_velocity, _terrain_info(env)))

        if args_cli.print_state and time.time() >= next_print_time:
            _print_state(env, step_count)
            next_print_time = time.time() + args_cli.print_interval_s

        step_count += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    if keyboard is not None:
        keyboard.close()
    if hud is not None:
        hud.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
