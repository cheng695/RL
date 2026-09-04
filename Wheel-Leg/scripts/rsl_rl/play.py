"""Play a trained WheelLeg RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a WheelLeg RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Try to run at real time speed.")
parser.add_argument("--scripted_commands", action="store_true", default=False, help="Play a fixed vx/yaw/height command sequence.")
parser.add_argument("--scripted_segment_s", type=float, default=5.0, help="Duration of each scripted command segment.")
parser.add_argument("--scripted_ramp_s", type=float, default=1.0, help="Ramp duration between scripted command segments.")
parser.add_argument("--scripted_vx", type=float, default=None, help="Override scripted forward/backward speed magnitude in m/s.")
parser.add_argument("--scripted_yaw", type=float, default=None, help="Override scripted yaw-rate magnitude in rad/s.")
parser.add_argument("--command_vx", type=float, default=None, help="Force a fixed vx command during play, in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Force a fixed yaw-rate command during play, in rad/s.")
parser.add_argument("--command_height", type=float, default=None, help="Force a fixed base-height command during play, in m.")
parser.add_argument(
    "--scripted_heights",
    type=str,
    default=None,
    help="Comma-separated height-only command sequence. Forces vx=0 and yaw=0 for each height.",
)
parser.add_argument("--print_state", action="store_true", default=False, help="Print env0 state diagnostics during play.")
parser.add_argument("--print_interval_s", type=float, default=0.5, help="Seconds between --print_state diagnostics.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
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


def _force_commands(env, vx: float | None, yaw: float | None, height: float | None) -> None:
    command_manager = env.unwrapped.command_manager

    base_velocity = command_manager.get_command("base_velocity")
    if vx is not None:
        base_velocity[:, 0] = vx
    base_velocity[:, 1] = 0.0
    if yaw is not None:
        base_velocity[:, 2] = yaw

    if height is not None:
        base_height = command_manager.get_command("base_height")
        base_height[:, 0] = height
        if base_height.shape[1] > 1:
            base_height[:, 1] = 0.0


def _parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _scripted_height_sequence(height_values: str) -> list[tuple[str, float, float, float]]:
    return [(f"h={height:.2f}, zero vx/yaw", 0.0, 0.0, height) for height in _parse_float_list(height_values)]


def _scripted_command_sequence(env_cfg, vx_override: float | None, yaw_override: float | None) -> list[tuple[str, float, float, float]]:
    cfg_vx_max = float(env_cfg.commands.base_velocity.ranges.lin_vel_x[1])
    cfg_vx_min = float(env_cfg.commands.base_velocity.ranges.lin_vel_x[0])
    cfg_yaw_max = float(env_cfg.commands.base_velocity.ranges.ang_vel_z[1])
    cfg_yaw_min = float(env_cfg.commands.base_velocity.ranges.ang_vel_z[0])
    vx_max = cfg_vx_max if vx_override is None else min(abs(vx_override), abs(cfg_vx_max))
    vx_min = cfg_vx_min if vx_override is None else -min(abs(vx_override), abs(cfg_vx_min))
    yaw_max = cfg_yaw_max if yaw_override is None else min(abs(yaw_override), abs(cfg_yaw_max))
    yaw_min = cfg_yaw_min if yaw_override is None else -min(abs(yaw_override), abs(cfg_yaw_min))
    return [
        ("h=0.27, forward max vx", vx_max, 0.0, 0.27),
        ("h=0.27, max yaw", 0.0, yaw_max, 0.27),
        ("h=0.35, backward max vx", vx_min, 0.0, 0.35),
        ("h=0.35, reverse max yaw", 0.0, yaw_min, 0.35),
        ("h=0.45, forward max vx", vx_max, 0.0, 0.45),
        ("h=0.45, max yaw", 0.0, yaw_max, 0.45),
    ]


def _interpolate_scripted_command(
    sequence: list[tuple[str, float, float, float]],
    segment_index: int,
    step_in_segment: int,
    ramp_steps: int,
) -> tuple[float, float, float]:
    _, target_vx, target_yaw, target_height = sequence[segment_index]
    if ramp_steps <= 0 or step_in_segment >= ramp_steps:
        return target_vx, target_yaw, target_height

    previous_index = (segment_index - 1) % len(sequence)
    _, previous_vx, previous_yaw, previous_height = sequence[previous_index]
    alpha = float(step_in_segment + 1) / float(ramp_steps)
    vx = previous_vx + alpha * (target_vx - previous_vx)
    yaw = previous_yaw + alpha * (target_yaw - previous_yaw)
    height = previous_height + alpha * (target_height - previous_height)
    return vx, yaw, height


def _as_scalar(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _whole_robot_com_w(robot) -> torch.Tensor:
    masses = robot.data.default_mass
    body_com_pos_w = robot.data.body_com_pos_w
    if masses is None:
        raise RuntimeError("robot.data.default_mass is not available; cannot compute whole-robot COM.")
    masses = masses.to(device=body_com_pos_w.device, dtype=body_com_pos_w.dtype)
    weights = masses.unsqueeze(-1)
    return torch.sum(body_com_pos_w * weights, dim=1) / torch.sum(weights, dim=1)


def _quat_pitch_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return pitch angle from a wxyz quaternion."""
    w, x, y, z = quat.unbind(dim=-1)
    sin_pitch = 2.0 * (w * y - z * x)
    return torch.asin(torch.clamp(sin_pitch, -1.0, 1.0))


def _quat_roll_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Return roll angle from a wxyz quaternion."""
    w, x, y, z = quat.unbind(dim=-1)
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    return torch.atan2(sin_roll, cos_roll)


def _wheel_forward_pair(raw_pair: torch.Tensor) -> torch.Tensor:
    """Normalize wheel sign so positive means chassis forward for both wheels."""
    return torch.stack((raw_pair[0], -raw_pair[1]))


def _common_diff(forward_pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    common = 0.5 * (forward_pair[0] + forward_pair[1])
    diff = 0.5 * (forward_pair[1] - forward_pair[0])
    return common, diff


def _wheel_normal_forces(env, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Best-effort wheel normal force from the wheel contact sensor."""
    nan_pair = torch.full((2,), float("nan"), device=device, dtype=dtype)
    try:
        sensors = env.unwrapped.scene.sensors
        sensor = sensors.get("wheel_contact") if hasattr(sensors, "get") else sensors["wheel_contact"]
    except Exception:
        return nan_pair
    if sensor is None:
        return nan_pair

    forces_w = getattr(sensor.data, "net_forces_w", None)
    if forces_w is not None:
        latest_forces = forces_w[0]
    else:
        forces_w_history = getattr(sensor.data, "net_forces_w_history", None)
        if forces_w_history is None:
            return nan_pair
        latest_forces = forces_w_history[0, -1]

    if latest_forces.shape[0] < 2:
        return nan_pair
    return latest_forces[:2, 2].to(device=device, dtype=dtype)


def _resolve_cfg_value(env_cfg, name: str, default: float | tuple[float, float, float]) -> float | tuple[float, float, float]:
    return getattr(env_cfg, name, default)


def _print_play_state(env, actions: torch.Tensor, env_cfg, step_count: int) -> None:
    robot = env.unwrapped.scene["robot"]
    command_manager = env.unwrapped.command_manager
    base_velocity = command_manager.get_command("base_velocity")
    base_height = command_manager.get_command("base_height")
    wheel_joint_ids = robot.find_joints(["Left_Wheel_joint", "Right_Wheel_joint"], preserve_order=True)[0]
    wheel_body_ids = robot.find_bodies(["Left_Wheel_link", "Right_Wheel_link"], preserve_order=True)[0]
    leg_body_ids = robot.find_bodies(["base_link", "Left_Wheel_link", "Right_Wheel_link"], preserve_order=True)[0]
    leg_joint_ids = robot.find_joints(
        ["Left_front_joint", "Left_rear_joint", "Right_front_joint", "Right_rear_joint"],
        preserve_order=True,
    )[0]

    root_pos_w = robot.data.root_pos_w[0]
    root_quat_w = robot.data.root_quat_w[0:1]
    root_lin_vel_b = robot.data.root_lin_vel_b[0]
    root_ang_vel_b = robot.data.root_ang_vel_b[0]
    projected_gravity_b = robot.data.projected_gravity_b[0]
    joint_vel = robot.data.joint_vel[0, wheel_joint_ids]
    leg_pos = robot.data.joint_pos[0, leg_joint_ids]
    leg_action = actions[0, 0:4].detach()
    if "leg_joint_positions" in env.unwrapped.action_manager.active_terms:
        leg_target = env.unwrapped.action_manager.get_term("leg_joint_positions").processed_actions[0].detach()
    else:
        leg_target = leg_action * _as_scalar(env_cfg.actions.leg_joint_positions.scale)
    torque = robot.data.applied_torque[0, wheel_joint_ids]
    torque_abs_max = torch.max(torch.abs(robot.data.applied_torque[:, wheel_joint_ids]))
    com_pos_w = _whole_robot_com_w(robot)
    wheel_center_w = torch.mean(robot.data.body_pos_w[:, wheel_body_ids, :], dim=1)
    com_minus_wheel_b = quat_apply_inverse(root_quat_w, com_pos_w[0:1] - wheel_center_w[0:1])[0]
    com_minus_wheel_x_w = com_pos_w[0, 0] - wheel_center_w[0, 0]

    wheel_action = actions[0, 4:6].detach()
    wheel_scale = _as_scalar(env_cfg.actions.wheel_joint_velocities.scale)
    wheel_radius = _as_scalar(_resolve_cfg_value(env_cfg, "WHEEL_RADIUS", 0.05))
    if "wheel_joint_velocities" in env.unwrapped.action_manager.active_terms:
        wheel_term = env.unwrapped.action_manager.get_term("wheel_joint_velocities")
        wheel_target_vel = wheel_term.processed_actions[0].detach()
        if hasattr(wheel_term, "shaped_actions"):
            wheel_shaped_action = wheel_term.shaped_actions[0].detach()
        else:
            wheel_shaped_action = wheel_target_vel / wheel_scale
    else:
        wheel_beta = _as_scalar(_resolve_cfg_value(env_cfg, "WHEEL_ACTION_CUBIC_BETA", 0.0))
        saturated_wheel_action = torch.clamp(torch.tanh(wheel_action) / math.tanh(1.0), -1.0, 1.0)
        wheel_shaped_action = (1.0 - wheel_beta) * saturated_wheel_action + wheel_beta * saturated_wheel_action**3
        wheel_target_vel = wheel_shaped_action * wheel_scale
    wheel_forward_action = _wheel_forward_pair(wheel_action)
    wheel_forward_shaped_action = _wheel_forward_pair(wheel_shaped_action)
    wheel_forward_target_vel = _wheel_forward_pair(wheel_target_vel)
    wheel_forward_vel = _wheel_forward_pair(joint_vel)
    wheel_action_common, wheel_action_diff = _common_diff(wheel_forward_action)
    wheel_shaped_common, wheel_shaped_diff = _common_diff(wheel_forward_shaped_action)
    wheel_target_common, wheel_target_diff = _common_diff(wheel_forward_target_vel)
    wheel_vel_common, wheel_vel_diff = _common_diff(wheel_forward_vel)
    vx_from_wheels = wheel_radius * wheel_vel_common
    wheel_slip_error = root_lin_vel_b[0] - vx_from_wheels
    wheel_vel_error = wheel_target_vel - joint_vel
    wheel_normal_force = _wheel_normal_forces(env, root_pos_w.device, root_pos_w.dtype)
    pitch = _quat_pitch_wxyz(root_quat_w[0])
    roll = _quat_roll_wxyz(root_quat_w[0])
    pitch_rate = root_ang_vel_b[1]
    env_origin = env.unwrapped.scene.env_origins[0]
    root_x_rel = root_pos_w[0] - env_origin[0]
    root_y_rel = root_pos_w[1] - env_origin[1]
    leg_left_sync = torch.square(leg_target[0] - leg_target[1])
    leg_right_sync = torch.square(leg_target[2] - leg_target[3])
    leg_front_mirror = torch.square(leg_target[0] + leg_target[2])
    leg_rear_mirror = torch.square(leg_target[1] + leg_target[3])
    leg_pattern_asymmetry = leg_left_sync + leg_right_sync + leg_front_mirror + leg_rear_mirror
    base_id, left_wheel_id, right_wheel_id = leg_body_ids
    base_pos_w = robot.data.body_pos_w[0:1, base_id]
    base_quat_w = robot.data.body_quat_w[0:1, base_id]
    left_wheel_pos_b = quat_apply_inverse(base_quat_w, robot.data.body_pos_w[0:1, left_wheel_id] - base_pos_w)[0]
    right_wheel_pos_b = quat_apply_inverse(base_quat_w, robot.data.body_pos_w[0:1, right_wheel_id] - base_pos_w)[0]
    wheel_x_b_l = left_wheel_pos_b[0]
    wheel_x_b_r = right_wheel_pos_b[0]
    wheel_x_diff = wheel_x_b_l - wheel_x_b_r
    wheel_x_mid = 0.5 * (wheel_x_b_l + wheel_x_b_r)
    left_hip_offset_b = _resolve_cfg_value(env_cfg, "LEFT_HIP_OFFSET_B", (-0.0193914, -0.1717, -0.05))
    right_hip_offset_b = _resolve_cfg_value(env_cfg, "RIGHT_HIP_OFFSET_B", (-0.0193914, 0.1707, -0.05))
    min_target_length = _as_scalar(_resolve_cfg_value(env_cfg, "MIN_LEG_VERTICAL_TARGET", 0.07))
    max_target_length = _as_scalar(_resolve_cfg_value(env_cfg, "MAX_LEG_VERTICAL_TARGET", 0.35))
    left_vertical_support = float(left_hip_offset_b[2]) - left_wheel_pos_b[2]
    right_vertical_support = float(right_hip_offset_b[2]) - right_wheel_pos_b[2]
    hip_z = 0.5 * (float(left_hip_offset_b[2]) + float(right_hip_offset_b[2]))
    target_vertical_support = torch.clamp(
        base_height[0, 0] + hip_z - wheel_radius,
        min=min_target_length,
        max=max_target_length,
    )

    print(
        "[PLAY] "
        f"step={step_count:06d} "
        f"h_cmd={base_height[0, 0].item(): .3f} h={root_pos_w[2].item(): .3f} "
        f"root=({root_pos_w[0].item():+.3f},{root_pos_w[1].item():+.3f})m "
        f"root_rel=({root_x_rel.item():+.3f},{root_y_rel.item():+.3f})m "
        f"vx_cmd={base_velocity[0, 0].item(): .3f} yaw_cmd={base_velocity[0, 2].item(): .3f} "
        f"vx={root_lin_vel_b[0].item(): .3f} vx_wheel={vx_from_wheels.item(): .3f} "
        f"slip_err={wheel_slip_error.item():+.3f} vz={root_lin_vel_b[2].item(): .3f} "
        f"roll={roll.item():+.3f} pitch={pitch.item():+.3f} pitch_rate={pitch_rate.item():+.3f} "
        f"proj_g_x={projected_gravity_b[0].item():+.3f} yaw_rate={root_ang_vel_b[2].item(): .3f} "
        f"wheel_act_raw=({wheel_action[0].item(): .3f},{wheel_action[1].item(): .3f}) "
        f"wheel_act_common={wheel_action_common.item():+.3f} wheel_act_diff={wheel_action_diff.item():+.3f} "
        f"wheel_act_shaped=({wheel_shaped_action[0].item(): .4f},{wheel_shaped_action[1].item(): .4f}) "
        f"wheel_act_shaped_common={wheel_shaped_common.item():+.4f} "
        f"wheel_act_shaped_diff={wheel_shaped_diff.item():+.4f} "
        f"wheel_target_raw=({wheel_target_vel[0].item(): .2f},{wheel_target_vel[1].item(): .2f})rad/s "
        f"wheel_target_common={wheel_target_common.item():+.2f} wheel_target_diff={wheel_target_diff.item():+.2f} "
        f"wheel_vel_raw=({joint_vel[0].item(): .2f},{joint_vel[1].item(): .2f})rad/s "
        f"wheel_vel_common={wheel_vel_common.item():+.2f} wheel_vel_diff={wheel_vel_diff.item():+.2f} "
        f"wheel_vel_err_raw=({wheel_vel_error[0].item():+.2f},{wheel_vel_error[1].item():+.2f})rad/s "
        f"wheel_tau=({torque[0].item(): .2f},{torque[1].item(): .2f})Nm "
        f"wheel_tau_abs_max={torque_abs_max.item(): .2f}Nm "
        f"wheel_Fz=({wheel_normal_force[0].item():.1f},{wheel_normal_force[1].item():.1f})N "
        f"wheel_x_b=(L {wheel_x_b_l.item():+.4f},R {wheel_x_b_r.item():+.4f},"
        f"diff {wheel_x_diff.item():+.4f},mid {wheel_x_mid.item():+.4f})m "
        f"COMx-wheelx_b={com_minus_wheel_b[0].item():+.4f}m "
        f"COMy-wheely_b={com_minus_wheel_b[1].item():+.4f}m "
        f"COMx-wheelx_w={com_minus_wheel_x_w.item():+.4f}m "
        f"leg_action=[LF {leg_action[0].item():+.3f}, LR {leg_action[1].item():+.3f}, "
        f"RF {leg_action[2].item():+.3f}, RR {leg_action[3].item():+.3f}] "
        f"leg_target=[LF {leg_target[0].item():+.3f}, LR {leg_target[1].item():+.3f}, "
        f"RF {leg_target[2].item():+.3f}, RR {leg_target[3].item():+.3f}]rad "
        f"leg_joint_pos=[LF {leg_pos[0].item():+.3f}, LR {leg_pos[1].item():+.3f}, "
        f"RF {leg_pos[2].item():+.3f}, RR {leg_pos[3].item():+.3f}]rad "
        f"leg_pattern_err={leg_pattern_asymmetry.item():.4f} "
        f"vertical_support=(L {left_vertical_support.item():+.4f}, R {right_vertical_support.item():+.4f}, "
        f"target {target_vertical_support.item():+.4f})m"
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

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    use_fixed_commands = (
        args_cli.command_vx is not None or args_cli.command_yaw is not None or args_cli.command_height is not None
    )
    use_scripted_commands = (args_cli.scripted_commands or args_cli.scripted_heights is not None) and not use_fixed_commands
    if args_cli.scripted_heights is not None:
        scripted_sequence = _scripted_height_sequence(args_cli.scripted_heights)
    else:
        scripted_sequence = _scripted_command_sequence(env_cfg, args_cli.scripted_vx, args_cli.scripted_yaw)
    scripted_segment_steps = max(1, int(round(args_cli.scripted_segment_s / dt)))
    scripted_ramp_steps = max(0, int(round(args_cli.scripted_ramp_s / dt)))
    step_count = 0
    last_segment_index = -1
    print_interval_steps = max(1, int(round(args_cli.print_interval_s / dt)))
    if use_fixed_commands:
        print(
            "[COMMAND] fixed: "
            f"vx={args_cli.command_vx if args_cli.command_vx is not None else 'keep'} m/s, "
            f"yaw={args_cli.command_yaw if args_cli.command_yaw is not None else 'keep'} rad/s, "
            f"height={args_cli.command_height if args_cli.command_height is not None else 'keep'} m"
        )

    while simulation_app.is_running():
        start_time = time.time()

        if use_fixed_commands:
            _force_commands(env, args_cli.command_vx, args_cli.command_yaw, args_cli.command_height)
            obs = env.get_observations()
        elif use_scripted_commands:
            segment_index = (step_count // scripted_segment_steps) % len(scripted_sequence)
            step_in_segment = step_count % scripted_segment_steps
            label, target_vx_cmd, target_yaw_cmd, target_height_cmd = scripted_sequence[segment_index]
            vx_cmd, yaw_cmd, height_cmd = _interpolate_scripted_command(
                scripted_sequence, segment_index, step_in_segment, scripted_ramp_steps
            )
            if segment_index != last_segment_index:
                print(
                    f"[COMMAND] {label}: target_vx={target_vx_cmd:.3f} m/s, "
                    f"target_yaw={target_yaw_cmd:.3f} rad/s, target_height={target_height_cmd:.3f} m"
                )
                last_segment_index = segment_index
            _force_commands(env, vx_cmd, yaw_cmd, height_cmd)
            obs = env.get_observations()

        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        if use_fixed_commands:
            _force_commands(env, args_cli.command_vx, args_cli.command_yaw, args_cli.command_height)
        elif use_scripted_commands:
            _force_commands(env, vx_cmd, yaw_cmd, height_cmd)
        if hasattr(policy, "reset"):
            policy.reset(dones)
        step_count += 1
        if args_cli.print_state and step_count % print_interval_steps == 0:
            _print_play_state(env, actions, env_cfg, step_count)

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
