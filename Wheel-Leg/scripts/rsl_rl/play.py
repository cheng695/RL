"""Play a trained UZ-05 RSL-RL checkpoint in the Isaac Sim window."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import os
import math
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
parser.add_argument("--terrain", choices=("mixed", "flat", "rough", "undulating", "slope_up", "slope_down", "stairs_up", "stairs_down"), default=None, help="Gentle-Play terrain selection; stairs are reserved.")
parser.add_argument("--terrain_amplitude", type=float, default=None, help="Rough/undulating height amplitude in meters (multiples of 0.0025).")
parser.add_argument("--terrain_slope_deg", type=float, default=None, help="Positive slope magnitude in degrees.")
parser.add_argument("--terrain_seed", type=int, default=None, help="Geometry seed, independent of --seed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Try to run at real-time speed.")
parser.add_argument("--command_vx", type=float, default=None, help="Override forward velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Override yaw-rate command in rad/s.")
parser.add_argument("--command_height", type=float, default=None, help="Override target base height in meters.")
parser.add_argument("--print_state", action="store_true", default=False, help="Print env0 state while playing.")
parser.add_argument("--print_interval_s", type=float, default=0.5, help="Seconds between state prints.")
parser.add_argument("--record_joint", type=str, default=None, help="Record one joint's target, state, and estimated torque.")
parser.add_argument("--joint_trace_csv", type=Path, default=None, help="CSV output path for --record_joint.")
parser.add_argument("--motion_trace_csv", type=Path, default=None,
                    help="Record env0 wheels, legs, body state and estimated torque to a new CSV.")
parser.add_argument("--motion_trace_steps", type=int, default=1000,
                    help="Stop after this many control steps when --motion_trace_csv is used.")
parser.add_argument("--no_keyboard", action="store_true", help="Disable WASD keyboard interaction.")
parser.add_argument("--no_hud", action="store_true", help="Disable the Isaac Sim HUD window.")
parser.add_argument("--no_follow_camera", action="store_true", help="Disable the third-person follow camera.")
parser.add_argument("--follow_camera", action="store_true", help="Opt in to the third-person follow camera.")
parser.add_argument(
    "--auto_reset", action="store_true", default=None,
    help="Enable all configured resets, including timeouts; default resets on failures only.",
)
parser.add_argument(
    "--no_auto_reset", dest="auto_reset", action="store_false",
    help="Disable all automatic resets; use R manually.",
)
parser.add_argument("--no_command_smoothing", action="store_true", help="Disable velocity/yaw slew-rate limiting.")
parser.add_argument("--eval_terrain", action="store_true", help="Stratified gentle-terrain evaluation, then exit.")
parser.add_argument("--terrain_eval_trials", type=int, default=100, help="Resets per terrain type and command case.")
parser.add_argument("--eval_tracking", action="store_true", help="Run five fixed-command cases and save CSV/JSON, then exit.")
parser.add_argument("--eval_trials", type=int, default=3, help="Independent resets per evaluation case.")
parser.add_argument("--eval_settle_s", type=float, default=2.0, help="Settling time excluded from measurements.")
parser.add_argument("--eval_measure_s", type=float, default=8.0, help="Measurement time per trial.")
parser.add_argument("--eval_output", type=Path, default=None, help="New directory for evaluation results.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.terrain in ("stairs_up", "stairs_down"):
    parser.error("上台阶/下台阶选项已预留，当前尚未启用。")
if args_cli.terrain_amplitude is not None:
    value = args_cli.terrain_amplitude
    if args_cli.terrain not in ("rough", "undulating") or not math.isfinite(value) or value <= 0 or not math.isclose(value/.0025, round(value/.0025), abs_tol=1.e-6):
        parser.error("--terrain_amplitude requires rough/undulating and a positive multiple of 0.0025 m")
if args_cli.terrain_slope_deg is not None:
    if args_cli.terrain not in ("slope_up", "slope_down") or not math.isfinite(args_cli.terrain_slope_deg) or not 0 < args_cli.terrain_slope_deg <= 17:
        parser.error("--terrain_slope_deg requires slope_up/slope_down and 0 < degrees <= 17")
if args_cli.terrain_seed is not None and args_cli.terrain_seed < 0:
    parser.error("--terrain_seed must be nonnegative")
if args_cli.eval_terrain and (args_cli.terrain not in (None, "mixed") or args_cli.terrain_seed is not None):
    parser.error("--eval_terrain uses the complete stage terrain mixture; set UZ05_TERRAIN_SEED for evaluation geometry")
if args_cli.motion_trace_csv is not None:
    if args_cli.motion_trace_steps < 1:
        parser.error("--motion_trace_steps must be positive")
    if args_cli.eval_tracking or args_cli.eval_terrain:
        parser.error("--motion_trace_csv uses ordinary play; do not combine with --eval_tracking")
if args_cli.eval_tracking or args_cli.eval_terrain:
    if args_cli.num_envs != 1:
        parser.error("Evaluation requires --num_envs 1")
    if args_cli.eval_trials < 1 or not math.isfinite(args_cli.eval_settle_s) or args_cli.eval_settle_s < 0:
        parser.error("Evaluation requires positive trials and finite, nonnegative settling time")
    if not math.isfinite(args_cli.eval_measure_s) or args_cli.eval_measure_s <= 0:
        parser.error("--eval_measure_s must be finite and positive")
    if any(value is not None for value in (args_cli.command_vx, args_cli.command_yaw, args_cli.command_height)):
        parser.error("Evaluation defines its own commands; omit --command_* overrides")

if args_cli.eval_tracking and args_cli.eval_terrain:
    parser.error("Choose one evaluation mode")
if args_cli.eval_terrain and args_cli.terrain_eval_trials < 1:
    parser.error("--terrain_eval_trials must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.mjcf")

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import my_robot_lab  # noqa: F401

from camera_controller import FollowCamera
from command_manager import CommandManager
from hud import RobotHUD
from keyboard_controller import KeyboardController
from motion_trace import MotionTrace
from checkpoint_mass_settings import restore_checkpoint_mass_settings
from my_robot_lab.assets.robots.my_robot_ import LEG_JOINTS, WHEEL_JOINTS


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


def _hud_text(env, command_state, effective_velocity) -> str:
    robot = env.unwrapped.scene["robot"]
    height_term = env.unwrapped.command_manager.get_term("base_height")
    body_ids, _ = robot.find_bodies(height_term.cfg.body_name)
    from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import local_ground_height
    ground = local_ground_height(env.unwrapped)[0]
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
        f"v_set/w_set    {command_state.v_set:.2f} / {command_state.w_set:.2f}\n\n"
        "Robot\n"
        f"vx / vy        {float(linear[0]):+.2f} / {float(linear[1]):+.2f} m/s\n"
        f"yaw rate       {float(angular[2]):+.2f} rad/s\n"
        f"roll / pitch   {roll * 57.2958:+.1f} / {pitch * 57.2958:+.1f} deg\n"
        f"height local   {float(relative_height):.3f} m\n"
        f"height error   {float(relative_height - height_term.command[0, 0]):+.3f} m\n"
        f"ground Z       {float(ground):.3f} m\n"
        f"base world Z   {float(body_z):.3f} m\n\n"
        "W/S move  A/D turn  Up/Down v  Left/Right yaw\n"
        "I/K height  H default  Space stop  R reset"
    )


@torch.inference_mode()
def main() -> None:
    installed_version = metadata.version("rsl-rl-lib")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    if args_cli.terrain is not None or args_cli.terrain_seed is not None:
        if not hasattr(env_cfg, "select_terrain"):
            raise ValueError("--terrain options require --task MyRobot-Velocity-Gentle-Play-v0")
        env_cfg.select_terrain(args_cli.terrain or "mixed", args_cli.terrain_amplitude,
                               args_cli.terrain_slope_deg, args_cli.terrain_seed)
        print(f"[INFO] Play terrain: {args_cli.terrain or 'mixed'}, geometry seed={env_cfg.scene.terrain.terrain_generator.seed}")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    env_cfg.seed = agent_cfg.seed
    if args_cli.eval_tracking or args_cli.eval_terrain:
        # Preserve failure terminations, but allow each trial its full duration.
        env_cfg.episode_length_s = args_cli.eval_settle_s + args_cli.eval_measure_s + (3.0 if args_cli.eval_terrain else 1.0)
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_height.resampling_time_range = (1.0e6, 1.0e6)
    else:
        # Interactive default: retain failure resets and remove time limits.
        # Explicit --no_auto_reset disables all terminations for manual inspection.
        for name in vars(env_cfg.terminations):
            term = getattr(env_cfg.terminations, name)
            if not name.startswith("_") and (args_cli.auto_reset is False or (
                args_cli.auto_reset is None and getattr(term, "time_out", False)
            )):
                setattr(env_cfg.terminations, name, None)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    for message in restore_checkpoint_mass_settings(env_cfg, resume_path):
        print(message)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.eval_tracking or args_cli.eval_terrain:
        from datetime import datetime
        from isaaclab.utils.io import dump_yaml
        from tracking_evaluation import run_tracking_evaluation

        output = args_cli.eval_output or Path("logs/tracking_eval") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = output.expanduser().resolve()
        try:
            output.mkdir(parents=True, exist_ok=False)
            dump_yaml(str(output / "env.yaml"), env_cfg)
            dump_yaml(str(output / "agent.yaml"), agent_cfg)
            if args_cli.eval_terrain:
                from terrain_evaluation import run_terrain_evaluation
                run_terrain_evaluation(env, policy, simulation_app, _force_velocity_command,
                                       args_cli, resume_path, output)
            else:
                run_tracking_evaluation(env, policy, simulation_app, _force_velocity_command,
                                        args_cli, resume_path, output)
        finally:
            env.close()
        return

    trace_file = None
    trace_writer = None
    trace_joint_id = None
    trace_joint_name = None
    trace_sample = 0
    if args_cli.record_joint is not None:
        robot = env.unwrapped.scene["robot"]
        joint_ids, joint_names = robot.find_joints(args_cli.record_joint, preserve_order=True)
        if len(joint_ids) != 1:
            raise ValueError(
                f"--record_joint must match exactly one joint; got {joint_names} for {args_cli.record_joint!r}."
            )
        trace_joint_id = joint_ids[0]
        trace_joint_name = joint_names[0]
        trace_path = args_cli.joint_trace_csv
        if trace_path is None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            trace_path = Path("logs") / "joint_traces" / f"{trace_joint_name}_{timestamp}.csv"
        trace_path = trace_path.expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", newline="", encoding="utf-8")
        trace_writer = csv.writer(trace_file)
        trace_writer.writerow([
            "time_s", "joint_name", "q_target_rad", "q_actual_rad", "qdot_actual_rad_s",
            "tau_pd_estimate_nm", "done",
        ])
        print(f"[INFO] Recording joint trace for {trace_joint_name}: {trace_path}")
        print("[INFO] Torque column is a clipped PD estimate from configured stiffness/damping, not measured motor torque.")

    motion_trace = None
    if args_cli.motion_trace_csv is not None:
        import atexit
        motion_trace = MotionTrace(env, args_cli.motion_trace_csv, resume_path, LEG_JOINTS, WHEEL_JOINTS)
        atexit.register(motion_trace.close)
        print(f"[INFO] Motion trace: {motion_trace.path} (implicit torque estimates, env0)")

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
        velocity_max=max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x),
        yaw_max=max(abs(v) for v in env_cfg.commands.base_velocity.ranges.ang_vel_z),
    )
    commands.state.v_set = min(0.3, commands.velocity_max)
    if args_cli.command_vx is not None:
        commands.state.v_set = max(0.1, abs(args_cli.command_vx))
    if args_cli.command_yaw is not None:
        commands.state.w_set = max(0.1, abs(args_cli.command_yaw))
    commands.update_settings()
    print(
        f"[INFO] Keyboard speed limits: vx={commands.velocity_max:.2f} m/s, "
        f"yaw={commands.yaw_max:.2f} rad/s (configured task command ranges)."
    )
    for name, value, maximum in (("command_vx", args_cli.command_vx,
                                  max(abs(v) for v in env_cfg.commands.base_velocity.ranges.lin_vel_x)),
                                 ("command_yaw", args_cli.command_yaw,
                                  max(abs(v) for v in env_cfg.commands.base_velocity.ranges.ang_vel_z))):
        if value is not None and abs(value) > maximum:
            print(f"[WARNING] Explicit --{name}={value} exceeds configured play range +/-{maximum}; stability is unvalidated.")
    keyboard = None if args_cli.no_keyboard else KeyboardController(commands)
    hud = None if args_cli.no_hud else RobotHUD()
    follow_camera = FollowCamera(env) if args_cli.follow_camera and not args_cli.no_follow_camera else None
    reset_mode = ("all configured terminations, including timeouts" if args_cli.auto_reset is True
                  else "manual only" if args_cli.auto_reset is False
                  else "failures only; timeouts disabled")
    print(f"[INFO] Automatic reset: {reset_mode}. Press R to reset manually.")
    print("[INFO] Keyboard: W/S move, A/D turn, arrows tune speeds, I/K height, H default, Space stop, R reset.")

    print("[INFO] Play started in the Isaac Sim window.")
    while simulation_app.is_running():
        start_time = time.time()
        if commands.state.reset_requested:
            if motion_trace is not None:
                motion_trace.episode += 1
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
            if motion_trace is not None:
                raw_action_snapshot = actions[0].detach().cpu().tolist()
                if len(raw_action_snapshot) != len(LEG_JOINTS) + len(WHEEL_JOINTS):
                    raise ValueError("Motion trace expects four leg actions followed by two wheel actions.")
                trace_command = effective_velocity.detach().cpu().tolist()
                trace_height = float(height_term.command[0, 0])
            obs, _, dones, _ = env.step(actions)
            if motion_trace is not None:
                motion_trace.write(raw_action_snapshot, trace_command, trace_height, dones[0].item())
            if trace_writer is not None:
                robot = env.unwrapped.scene["robot"]
                joint_id = trace_joint_id
                q_target = robot.data.joint_pos_target[0, joint_id]
                q_actual = robot.data.joint_pos[0, joint_id]
                qdot_target = robot.data.joint_vel_target[0, joint_id]
                qdot_actual = robot.data.joint_vel[0, joint_id]
                kp = robot.data.joint_stiffness[0, joint_id]
                kd = robot.data.joint_damping[0, joint_id]
                effort_limit = robot.data.joint_effort_limits[0, joint_id].abs()
                tau_estimate = torch.clamp(
                    kp * (q_target - q_actual) + kd * (qdot_target - qdot_actual),
                    min=-effort_limit,
                    max=effort_limit,
                )
                trace_writer.writerow([
                    f"{trace_sample * dt:.6f}",
                    trace_joint_name,
                    f"{q_target.item():.8f}",
                    f"{q_actual.item():.8f}",
                    f"{qdot_actual.item():.8f}",
                    f"{tau_estimate.item():.8f}",
                    int(dones[0].item()),
                ])
                trace_sample += 1
            if hasattr(policy, "reset"):
                policy.reset(dones)

        if follow_camera is not None and step_count % 2 == 0:
            follow_camera.update()
        if hud is not None and step_count % 2 == 0:
            hud.update(_hud_text(env, command_state, effective_velocity))

        if args_cli.print_state and time.time() >= next_print_time:
            _print_state(env, step_count)
            next_print_time = time.time() + args_cli.print_interval_s

        step_count += 1
        if motion_trace is not None and motion_trace.index >= args_cli.motion_trace_steps:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    if keyboard is not None:
        keyboard.close()
    if hud is not None:
        hud.close()
    if trace_file is not None:
        trace_file.close()
        print(f"[INFO] Joint trace saved: {trace_path}")
    if motion_trace is not None:
        motion_trace.close()
        print(f"[INFO] Motion trace saved: {motion_trace.path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
