"""Calibrate virtual-leg geometry from the spawned WheelLeg USD.

The script samples leg joint targets, reads the resulting wheel-center poses in
the base frame, and fits a Fudan-style two-link virtual leg model:

    x = offset + l1*cos(theta1) + l2*cos(theta1 + theta2)
    y =          l1*sin(theta1) + l2*sin(theta1 + theta2)

where y is the downward vertical distance from the hip point to the wheel.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Calibrate WheelLeg virtual-leg geometry.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--samples", type=int, default=2048, help="Number of settled samples to collect.")
parser.add_argument("--preview_samples", type=int, default=0, help="Render the first N sampled poses slowly in GUI mode.")
parser.add_argument("--preview_dt", type=float, default=0.25, help="Seconds to hold each previewed sample.")
parser.add_argument("--linger", action="store_true", help="Keep the GUI window open after calibration until it is closed.")
parser.add_argument("--preview_only", action="store_true", help="Collect/preview samples and skip fitting.")
parser.add_argument(
    "--analysis_mode",
    choices=("fit_two_link", "point"),
    default="fit_two_link",
    help="Fit the Fudan two-link model or validate direct hip-to-wheel point geometry.",
)
parser.add_argument(
    "--collection_mode",
    choices=("kinematic", "pd"),
    default="kinematic",
    help="Collect samples by direct joint-state writes or by PD settling through actions.",
)
parser.add_argument("--settle_steps", type=int, default=25, help="Policy steps to settle each sampled action in pd mode.")
parser.add_argument("--joint_range", type=float, default=0.8, help="Random absolute leg target range, in rad.")
parser.add_argument(
    "--sample_mode",
    choices=("mirrored", "independent"),
    default="mirrored",
    help="Use mirrored left/right samples or fully independent samples.",
)
parser.add_argument("--root_height", type=float, default=1.0, help="Fixed suspended root height during sampling.")
parser.add_argument("--no_hold_root", action="store_true", help="Do not pin the root pose during sampling.")
parser.add_argument(
    "--keep_safety_terminations",
    action="store_true",
    help="Keep normal safety terminations. By default they are disabled for suspended calibration.",
)
parser.add_argument("--min_leg_vertical", type=float, default=0.03, help="Reject samples below this hip-to-wheel vertical span.")
parser.add_argument("--max_leg_vertical", type=float, default=0.45, help="Reject samples above this hip-to-wheel vertical span.")
parser.add_argument("--max_wheel_x", type=float, default=0.35, help="Reject samples with too large hip-to-wheel x offset.")
parser.add_argument("--fit_steps", type=int, default=2500, help="Adam steps per sign candidate.")
parser.add_argument("--lr", type=float, default=3.0e-2, help="Adam learning rate.")
parser.add_argument("--q0_reg", type=float, default=1.0e-4, help="Small regularization on fitted encoder zero offsets.")
parser.add_argument("--fit_q0", action="store_true", help="Fit encoder zero offsets instead of keeping q0 fixed at zero.")
parser.add_argument("--q0_limit", type=float, default=0.8, help="Absolute q0 bound in rad when --fit_q0 is enabled.")
parser.add_argument("--length_reg", type=float, default=0.0, help="Regularization toward initial l1/l2 guesses.")
parser.add_argument("--init_offset", type=float, default=0.0, help="Initial offset guess.")
parser.add_argument("--init_l1", type=float, default=0.175, help="Initial first-link length guess.")
parser.add_argument("--init_l2", type=float, default=0.208, help="Initial second-link length guess.")
parser.add_argument(
    "--fixed_signs",
    type=float,
    nargs=4,
    default=None,
    metavar=("LF", "LR", "RF", "RR"),
    help="Fit only one sign pattern, e.g. 1 1 -1 -1.",
)
parser.add_argument(
    "--search_joint_order",
    action="store_true",
    help="Also test front/rear swaps inside each side. This is slower but checks the assumed joint order.",
)
parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import mdp
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils.math import quat_apply_inverse

import envs.wheel_leg  # noqa: F401
from envs.wheel_leg.wheel_leg_env_cfg import (
    LEG_JOINTS,
    LEG_KINEMATICS_BODIES,
    LEFT_HIP_OFFSET_B,
    RIGHT_HIP_OFFSET_B,
    WHEEL_RADIUS_FOR_HEIGHT_REWARD,
)


def _softplus_inverse(value: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = torch.tensor(value, device=device, dtype=dtype)
    return torch.log(torch.expm1(tensor.clamp_min(1.0e-6)))


def _relative_wheel_xy(env, leg_joint_ids: list[int], body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return q, target [left_x, left_y_down, right_x, right_y_down], and base z."""
    robot = env.scene["robot"]
    base_id, left_wheel_id, right_wheel_id = body_ids

    q = robot.data.joint_pos[:, leg_joint_ids].detach()
    base_pos_w = robot.data.body_pos_w[:, base_id]
    base_quat_w = robot.data.body_quat_w[:, base_id]

    left_wheel_b = quat_apply_inverse(base_quat_w, robot.data.body_pos_w[:, left_wheel_id] - base_pos_w)
    right_wheel_b = quat_apply_inverse(base_quat_w, robot.data.body_pos_w[:, right_wheel_id] - base_pos_w)

    left_hip = torch.tensor(LEFT_HIP_OFFSET_B, device=q.device, dtype=q.dtype)
    right_hip = torch.tensor(RIGHT_HIP_OFFSET_B, device=q.device, dtype=q.dtype)

    left_x = left_wheel_b[:, 0] - left_hip[0]
    left_y_down = left_hip[2] - left_wheel_b[:, 2]
    right_x = right_wheel_b[:, 0] - right_hip[0]
    right_y_down = right_hip[2] - right_wheel_b[:, 2]
    target = torch.stack((left_x, left_y_down, right_x, right_y_down), dim=1).detach()
    return q, target, robot.data.root_pos_w[:, 2].detach()


def _collect_samples(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    leg_joint_ids = robot.find_joints(LEG_JOINTS, preserve_order=True)[0]
    body_ids = robot.find_bodies(LEG_KINEMATICS_BODIES, preserve_order=True)[0]

    env.reset()
    actions = torch.zeros_like(env.action_manager.action)
    held_root_pose = robot.data.root_state_w[:, :7].clone()
    if args_cli.root_height is not None:
        held_root_pose[:, 2] = args_cli.root_height
    zero_root_velocity = torch.zeros(actions.shape[0], 6, device=actions.device, dtype=actions.dtype)

    generator = torch.Generator(device=actions.device)
    generator.manual_seed(args_cli.seed)
    q_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    base_z_chunks: list[torch.Tensor] = []
    collected = 0
    previewed = 0

    while collected < args_cli.samples:
        leg_targets = _sample_leg_targets(actions.shape[0], actions.device, actions.dtype, generator)

        with torch.inference_mode():
            if args_cli.collection_mode == "kinematic":
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = torch.zeros_like(joint_pos)
                joint_pos[:, leg_joint_ids] = leg_targets
                robot.write_root_pose_to_sim(held_root_pose)
                robot.write_root_velocity_to_sim(zero_root_velocity)
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                _refresh_sim_kinematics(env, robot)
                if previewed < args_cli.preview_samples:
                    _render_preview(env, args_cli.preview_dt)
                    previewed += actions.shape[0]
            else:
                actions.zero_()
                actions[:, 0:4] = leg_targets
                for _ in range(args_cli.settle_steps):
                    if not args_cli.no_hold_root:
                        robot.write_root_pose_to_sim(held_root_pose)
                        robot.write_root_velocity_to_sim(zero_root_velocity)
                    env.step(actions)
                    if not args_cli.no_hold_root:
                        robot.write_root_pose_to_sim(held_root_pose)
                        robot.write_root_velocity_to_sim(zero_root_velocity)
                if previewed < args_cli.preview_samples:
                    _render_preview(env, args_cli.preview_dt)
                    previewed += actions.shape[0]

        q, target, base_z = _relative_wheel_xy(env, leg_joint_ids, body_ids)
        finite = torch.all(torch.isfinite(q), dim=1) & torch.all(torch.isfinite(target), dim=1)
        sane_left = (
            (torch.abs(target[:, 0]) < args_cli.max_wheel_x)
            & (target[:, 1] > args_cli.min_leg_vertical)
            & (target[:, 1] < args_cli.max_leg_vertical)
        )
        sane_right = (
            (torch.abs(target[:, 2]) < args_cli.max_wheel_x)
            & (target[:, 3] > args_cli.min_leg_vertical)
            & (target[:, 3] < args_cli.max_leg_vertical)
        )
        finite = finite & sane_left & sane_right
        q_chunks.append(q[finite].detach().cpu())
        target_chunks.append(target[finite].detach().cpu())
        base_z_chunks.append(base_z[finite].detach().cpu())
        collected += int(finite.sum().item())
        print(f"collected={min(collected, args_cli.samples)}/{args_cli.samples}", flush=True)

    q_all = torch.cat(q_chunks, dim=0)[: args_cli.samples]
    target_all = torch.cat(target_chunks, dim=0)[: args_cli.samples]
    base_z_all = torch.cat(base_z_chunks, dim=0)[: args_cli.samples]
    return q_all, target_all, base_z_all


def _sample_leg_targets(
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if args_cli.sample_mode == "mirrored":
        left_offsets = torch.rand(num_envs, 2, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0
        leg_targets = torch.empty(num_envs, 4, device=device, dtype=dtype)
        leg_targets[:, 0:2] = left_offsets * args_cli.joint_range
        leg_targets[:, 2:4] = -left_offsets * args_cli.joint_range
    else:
        leg_targets = torch.rand(num_envs, 4, generator=generator, device=device, dtype=dtype) * 2.0 - 1.0
        leg_targets = leg_targets * args_cli.joint_range
    return leg_targets


def _refresh_sim_kinematics(env, robot) -> None:
    """Refresh body poses after direct joint writes without relying on falling/contact dynamics."""
    if hasattr(env.sim, "forward"):
        env.sim.forward()
    else:
        env.sim.step(render=False)
    if hasattr(robot, "update"):
        robot.update(0.0)
    if hasattr(env.scene, "update"):
        env.scene.update(0.0)


def _render_preview(env, hold_seconds: float) -> None:
    start_time = time.time()
    while simulation_app.is_running() and time.time() - start_time < hold_seconds:
        env.sim.render()
        time.sleep(1.0 / 60.0)


def _forward_kinematics(
    q: torch.Tensor,
    signs: torch.Tensor,
    q0: torch.Tensor,
    offset: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
) -> torch.Tensor:
    lf_q, lr_q, rf_q, rr_q = (signs * (q - q0)).unbind(dim=1)
    theta1 = torch.stack((lf_q, rf_q), dim=1)
    theta2 = torch.stack((lr_q + math.pi / 2.0, rr_q + math.pi / 2.0), dim=1)
    theta12 = theta1 + theta2

    end_x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta12)
    end_y = l1 * torch.sin(theta1) + l2 * torch.sin(theta12)
    return torch.stack((end_x[:, 0], end_y[:, 0], end_x[:, 1], end_y[:, 1]), dim=1)


def _fit_one_candidate(
    q: torch.Tensor,
    target: torch.Tensor,
    signs_tuple: tuple[float, float, float, float],
    order_tuple: tuple[int, int, int, int],
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    q = q[:, order_tuple].to(device=device, dtype=dtype)
    target = target.to(device=device, dtype=dtype)
    signs = torch.tensor(signs_tuple, device=device, dtype=dtype)

    offset = torch.tensor(args_cli.init_offset, device=device, dtype=dtype, requires_grad=True)
    raw_l1 = _softplus_inverse(args_cli.init_l1, device, dtype).detach().requires_grad_(True)
    raw_l2 = _softplus_inverse(args_cli.init_l2, device, dtype).detach().requires_grad_(True)
    raw_q0 = torch.zeros(4, device=device, dtype=dtype, requires_grad=args_cli.fit_q0)

    params = [offset, raw_l1, raw_l2]
    if args_cli.fit_q0:
        params.append(raw_q0)
    optimizer = torch.optim.Adam(params, lr=args_cli.lr)
    for _ in range(args_cli.fit_steps):
        optimizer.zero_grad(set_to_none=True)
        l1 = torch.nn.functional.softplus(raw_l1) + 1.0e-5
        l2 = torch.nn.functional.softplus(raw_l2) + 1.0e-5
        if args_cli.fit_q0:
            q0 = args_cli.q0_limit * torch.tanh(raw_q0)
        else:
            q0 = raw_q0
        pred = _forward_kinematics(q, signs, q0, offset, l1, l2)
        mse = torch.mean((pred - target) ** 2)
        reg = args_cli.q0_reg * torch.mean(q0**2) if args_cli.fit_q0 else torch.zeros((), device=device)
        reg = reg + args_cli.length_reg * ((l1 - args_cli.init_l1) ** 2 + (l2 - args_cli.init_l2) ** 2)
        loss = mse + reg
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        l1 = torch.nn.functional.softplus(raw_l1) + 1.0e-5
        l2 = torch.nn.functional.softplus(raw_l2) + 1.0e-5
        if args_cli.fit_q0:
            q0 = args_cli.q0_limit * torch.tanh(raw_q0)
        else:
            q0 = raw_q0
        pred = _forward_kinematics(q, signs, q0, offset, l1, l2)
        err = pred - target
        rmse_total = torch.sqrt(torch.mean(err**2))
        rmse_by_axis = torch.sqrt(torch.mean(err**2, dim=0))
        theta_pred = torch.atan2(
            torch.stack((pred[:, 1], pred[:, 3]), dim=1),
            torch.stack((pred[:, 0], pred[:, 2]), dim=1),
        ) - math.pi / 2.0
        length_pred = torch.sqrt(
            torch.stack((pred[:, 0] ** 2 + pred[:, 1] ** 2, pred[:, 2] ** 2 + pred[:, 3] ** 2), dim=1)
        )

    return {
        "signs": list(signs_tuple),
        "joint_order_indices": list(order_tuple),
        "joint_order_names": [LEG_JOINTS[index] for index in order_tuple],
        "loss": float(torch.mean(err**2).item()),
        "rmse_total_m": float(rmse_total.item()),
        "rmse_left_x_m": float(rmse_by_axis[0].item()),
        "rmse_left_y_down_m": float(rmse_by_axis[1].item()),
        "rmse_right_x_m": float(rmse_by_axis[2].item()),
        "rmse_right_y_down_m": float(rmse_by_axis[3].item()),
        "offset_m": float(offset.detach().cpu().item()),
        "l1_m": float(l1.detach().cpu().item()),
        "l2_m": float(l2.detach().cpu().item()),
        "q0_rad": [float(value) for value in q0.detach().cpu().tolist()],
        "length_range_m": [
            float(length_pred.min().detach().cpu().item()),
            float(length_pred.max().detach().cpu().item()),
        ],
        "theta_range_rad": [
            float(theta_pred.min().detach().cpu().item()),
            float(theta_pred.max().detach().cpu().item()),
        ],
    }


def _fit_geometry(q: torch.Tensor, target: torch.Tensor) -> list[dict]:
    if args_cli.fixed_signs is not None:
        sign_patterns = [tuple(float(value) for value in args_cli.fixed_signs)]
    else:
        sign_patterns = list(itertools.product((-1.0, 1.0), repeat=4))
    if args_cli.search_joint_order:
        order_patterns = [
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (0, 1, 3, 2),
            (1, 0, 3, 2),
        ]
    else:
        order_patterns = [(0, 1, 2, 3)]

    results = []
    total = len(sign_patterns) * len(order_patterns)
    index = 0
    for order in order_patterns:
        for signs in sign_patterns:
            index += 1
            print(f"fitting candidate {index}/{total}: order={order} signs={signs}", flush=True)
            results.append(_fit_one_candidate(q, target, signs, order))

    results.sort(key=lambda item: item["loss"])
    return results


def _print_dataset_stats(q: torch.Tensor, target: torch.Tensor) -> None:
    print("\nCollected dataset stats")
    for index, name in enumerate(LEG_JOINTS):
        values = q[:, index]
        print(
            f"  q/{name:18s} min={values.min().item(): .4f} "
            f"mean={values.mean().item(): .4f} max={values.max().item(): .4f} rad"
        )
    target_names = ("left_x", "left_y_down", "right_x", "right_y_down")
    for index, name in enumerate(target_names):
        values = target[:, index]
        print(
            f"  target/{name:12s} min={values.min().item(): .4f} "
            f"mean={values.mean().item(): .4f} max={values.max().item(): .4f} m"
        )


def _summary(values: torch.Tensor) -> dict[str, float]:
    return {
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "rmse": float(torch.sqrt(torch.mean(values**2)).item()),
    }


def _print_point_geometry_validation(target: torch.Tensor, base_z: torch.Tensor) -> dict:
    left_x = target[:, 0]
    left_y_down = target[:, 1]
    right_x = target[:, 2]
    right_y_down = target[:, 3]

    left_l = torch.sqrt(left_x**2 + left_y_down**2).clamp_min(1.0e-6)
    right_l = torch.sqrt(right_x**2 + right_y_down**2).clamp_min(1.0e-6)
    left_theta = torch.atan2(left_x, left_y_down)
    right_theta = torch.atan2(right_x, right_y_down)

    left_x_recon = left_l * torch.sin(left_theta)
    left_y_recon = left_l * torch.cos(left_theta)
    right_x_recon = right_l * torch.sin(right_theta)
    right_y_recon = right_l * torch.cos(right_theta)
    reconstruction_error = torch.cat(
        (
            left_x_recon - left_x,
            left_y_recon - left_y_down,
            right_x_recon - right_x,
            right_y_recon - right_y_down,
        )
    )

    left_height_est = left_y_down - LEFT_HIP_OFFSET_B[2] + WHEEL_RADIUS_FOR_HEIGHT_REWARD
    right_height_est = right_y_down - RIGHT_HIP_OFFSET_B[2] + WHEEL_RADIUS_FOR_HEIGHT_REWARD
    mean_height_est = 0.5 * (left_height_est + right_height_est)

    validation = {
        "left_length_m": _summary(left_l),
        "right_length_m": _summary(right_l),
        "left_theta_rad": _summary(left_theta),
        "right_theta_rad": _summary(right_theta),
        "length_left_minus_right_m": _summary(left_l - right_l),
        "theta_left_minus_right_rad": _summary(left_theta - right_theta),
        "wheel_x_left_minus_right_m": _summary(left_x - right_x),
        "point_reconstruction_error_m": _summary(reconstruction_error),
        "estimated_base_height_left_m": _summary(left_height_est),
        "estimated_base_height_right_m": _summary(right_height_est),
        "estimated_base_height_mean_m": _summary(mean_height_est),
        "estimated_minus_actual_base_height_m": _summary(mean_height_est - base_z),
    }

    print("\nPoint-Geometry Validation")
    print(json.dumps(validation, indent=2))
    print(
        "\nNote: estimated_minus_actual_base_height is meaningful only on flat ground with wheels in contact. "
        "In suspended tests, use length/theta ranges and left-right consistency instead."
    )
    return validation


def _print_result(best: dict, all_results: list[dict]) -> None:
    print("\nBest virtual-leg calibration")
    print(json.dumps(best, indent=2))
    print("\nTop sign candidates")
    for item in all_results[:5]:
        print(
            "  order={} signs={} rmse={:.6f} m offset={:.6f} l1={:.6f} l2={:.6f}".format(
                item["joint_order_names"],
                item["signs"],
                item["rmse_total_m"],
                item["offset_m"],
                item["l1_m"],
                item["l2_m"],
            )
        )

    lf, lr, rf, rr = best["signs"]
    print("\nSuggested config values")
    print(f"VIRTUAL_LEG_OFFSET = {best['offset_m']:.8f}")
    print(f"VIRTUAL_LEG_L1 = {best['l1_m']:.8f}")
    print(f"VIRTUAL_LEG_L2 = {best['l2_m']:.8f}")
    print(f"joint order [LF, LR, RF, RR] = {best['joint_order_names']}")
    print(
        "signs: "
        f"left_front_sign={lf:.0f}, left_rear_sign={lr:.0f}, "
        f"right_front_sign={rf:.0f}, right_rear_sign={rr:.0f}"
    )
    print(f"q0 offsets rad [LF, LR, RF, RR] = {best['q0_rad']}")
    print("If q0 offsets are not close to zero, add encoder zero-offset correction before using the geometry online.")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if not args_cli.keep_safety_terminations:
        _disable_safety_terminations(env_cfg)

    env_cfg.actions.leg_joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINTS,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    try:
        q, target, base_z = _collect_samples(env)
        _print_dataset_stats(q, target)
        point_validation = _print_point_geometry_validation(target, base_z)
        if args_cli.preview_only:
            print(f"\nPreview-only collection finished with {q.shape[0]} accepted samples. Skipping fit.")
            if args_cli.linger:
                print("GUI will stay open until you close Isaac Sim.")
                while simulation_app.is_running():
                    env.sim.render()
                    time.sleep(1.0 / 60.0)
            return
        if args_cli.analysis_mode == "point":
            if args_cli.output is not None:
                payload = {
                    "task": args_cli.task,
                    "samples": int(q.shape[0]),
                    "joint_names": LEG_JOINTS,
                    "body_names": LEG_KINEMATICS_BODIES,
                    "left_hip_offset_b": LEFT_HIP_OFFSET_B,
                    "right_hip_offset_b": RIGHT_HIP_OFFSET_B,
                    "wheel_radius_for_height_reward": WHEEL_RADIUS_FOR_HEIGHT_REWARD,
                    "point_validation": point_validation,
                }
                args_cli.output.parent.mkdir(parents=True, exist_ok=True)
                args_cli.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"\nWrote {args_cli.output}")
            return
        print(f"\nFitting {q.shape[0]} samples...")
        results = _fit_geometry(q, target)
        best = results[0]
        _print_result(best, results)
        if args_cli.output is not None:
            payload = {
                "task": args_cli.task,
                "samples": int(q.shape[0]),
                "joint_names": LEG_JOINTS,
                "body_names": LEG_KINEMATICS_BODIES,
                "left_hip_offset_b": LEFT_HIP_OFFSET_B,
                "right_hip_offset_b": RIGHT_HIP_OFFSET_B,
                "best": best,
                "top_results": results[:5],
            }
            args_cli.output.parent.mkdir(parents=True, exist_ok=True)
            args_cli.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nWrote {args_cli.output}")
        if args_cli.linger:
            print("\nCalibration finished. GUI will stay open until you close Isaac Sim.")
            while simulation_app.is_running():
                env.sim.render()
                time.sleep(1.0 / 60.0)
    finally:
        env.close()


def _disable_safety_terminations(env_cfg) -> None:
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


if __name__ == "__main__":
    main()
    simulation_app.close()
