"""Debug WheelLeg rollouts without PPO updates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a short WheelLeg rollout and print diagnostics.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--steps", type=int, default=500, help="Number of rollout steps.")
parser.add_argument("--print_interval", type=int, default=25, help="Print diagnostics every N steps.")
parser.add_argument("--mode", choices=("zero", "random"), default="random", help="Action source.")
parser.add_argument("--random_action_std", type=float, default=0.3, help="Random action standard deviation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401
from envs.wheel_leg.wheel_leg_env_cfg import (
    CONTROLLED_JOINTS,
)


def _stats(name: str, value: torch.Tensor) -> str:
    value = value.detach()
    finite = torch.isfinite(value)
    if not torch.any(finite):
        return f"{name}: all_non_finite shape={tuple(value.shape)}"
    finite_value = value[finite]
    return (
        f"{name}: min={finite_value.min().item(): .4e} "
        f"mean={finite_value.mean().item(): .4e} "
        f"max={finite_value.max().item(): .4e} "
        f"finite={finite.sum().item()}/{value.numel()}"
    )


def _obs_tensor(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        if "policy" in obs:
            return obs["policy"]
        first_key = next(iter(obs))
        return obs[first_key]
    return obs


def _print_reward_terms(env, top_k: int = 12) -> None:
    reward_manager = env.reward_manager
    step_reward = reward_manager._step_reward.detach()
    names = reward_manager.active_terms
    summaries = []
    for index, name in enumerate(names):
        value = step_reward[:, index]
        finite = torch.isfinite(value)
        if torch.any(finite):
            finite_value = value[finite]
            abs_max = torch.max(torch.abs(finite_value)).item()
            mean = finite_value.mean().item()
            min_value = finite_value.min().item()
            max_value = finite_value.max().item()
        else:
            abs_max = float("inf")
            mean = float("nan")
            min_value = float("nan")
            max_value = float("nan")
        summaries.append((abs_max, name, min_value, mean, max_value, finite.sum().item(), value.numel()))

    summaries.sort(reverse=True, key=lambda item: item[0])
    print("reward_terms:")
    for _, name, min_value, mean, max_value, finite_count, total_count in summaries[:top_k]:
        print(
            f"  {name:24s} min={min_value: .4e} "
            f"mean={mean: .4e} max={max_value: .4e} finite={finite_count}/{total_count}"
        )


def _print_command_terms(env) -> None:
    print("command_terms:")
    for name in env.command_manager.active_terms:
        command = env.command_manager.get_command(name)
        print(f"  {name:24s} {_stats('', command).lstrip(': ')}")
        if name == "base_height" and command.shape[1] >= 2:
            print(f"  {'base_height/height':24s} {_stats('', command[:, 0]).lstrip(': ')}")
            print(f"  {'base_height/rate':24s} {_stats('', command[:, 1]).lstrip(': ')}")
            command_term = env.command_manager.get_term(name)
            if hasattr(command_term, "elapsed_time"):
                print(f"  {'base_height/elapsed':24s} {_stats('', command_term.elapsed_time).lstrip(': ')}")
            if hasattr(command_term, "height_direction"):
                print(f"  {'base_height/direction':24s} {_stats('', command_term.height_direction[:, 0]).lstrip(': ')}")


def _print_termination_terms(env) -> None:
    termination_manager = env.termination_manager
    print("termination_terms:")
    for name in termination_manager.active_terms:
        value = termination_manager.get_term(name)
        print(f"  {name:24s} count={value.sum().item()}/{value.numel()}")


def _print_observation_terms(env, obs_policy: torch.Tensor, group_name: str = "policy") -> None:
    observation_manager = env.observation_manager
    term_names = observation_manager.active_terms[group_name]
    term_dims = observation_manager.group_obs_term_dim[group_name]
    concat_dim = observation_manager._group_obs_concatenate_dim[group_name]
    if concat_dim > 0:
        concat_dim -= 1

    print("observation_terms:")
    index = 0
    for term_name, term_dim in zip(term_names, term_dims, strict=True):
        width = term_dim[concat_dim]
        term_value = obs_policy.narrow(dim=concat_dim, start=index, length=width)
        print(f"  {term_name:24s} {_stats('', term_value).lstrip(': ')}")
        index += width


def _print_non_finite_observation_details(env, obs_policy: torch.Tensor, group_name: str = "policy") -> None:
    observation_manager = env.observation_manager
    term_names = observation_manager.active_terms[group_name]
    term_dims = observation_manager.group_obs_term_dim[group_name]
    concat_dim = observation_manager._group_obs_concatenate_dim[group_name]
    if concat_dim > 0:
        concat_dim -= 1

    print("non_finite_observation_details:")
    index = 0
    for term_name, term_dim in zip(term_names, term_dims, strict=True):
        width = term_dim[concat_dim]
        term_value = obs_policy.narrow(dim=concat_dim, start=index, length=width)
        finite = torch.isfinite(term_value)
        if not torch.all(finite):
            bad_indices = torch.nonzero(~finite, as_tuple=False)
            bad_env_ids = torch.unique(bad_indices[:, 0]).detach().cpu().tolist()
            bad_cols = torch.unique(bad_indices[:, 1]).detach().cpu().tolist()
            print(
                f"  {term_name:24s} start={index} width={width} "
                f"bad_env_ids={bad_env_ids[:16]} bad_cols={bad_cols[:16]} "
                f"bad_count={bad_indices.shape[0]}/{term_value.numel()}"
            )
        index += width


def _print_bad_env_state(env, actions: torch.Tensor, obs_policy: torch.Tensor) -> None:
    robot = env.scene["robot"]
    bad_env_mask = torch.any(~torch.isfinite(obs_policy), dim=1)
    bad_env_ids = torch.nonzero(bad_env_mask, as_tuple=False).flatten()
    if bad_env_ids.numel() == 0:
        return

    print("bad_env_state:")
    print(f"  bad_env_ids={bad_env_ids[:16].detach().cpu().tolist()} count={bad_env_ids.numel()}")
    ids = bad_env_ids[:8]
    print(_stats("  bad_root_pos_w", robot.data.root_pos_w[ids]))
    print(_stats("  bad_root_quat_w", robot.data.root_quat_w[ids]))
    print(_stats("  bad_root_lin_vel_b", robot.data.root_lin_vel_b[ids]))
    print(_stats("  bad_root_ang_vel_b", robot.data.root_ang_vel_b[ids]))
    print(_stats("  bad_projected_gravity_b", robot.data.projected_gravity_b[ids]))
    print(_stats("  bad_joint_pos", robot.data.joint_pos[ids]))
    print(_stats("  bad_joint_vel", robot.data.joint_vel[ids]))
    print(_stats("  bad_action", actions[ids]))


def _print_leg_and_action_terms(env, actions: torch.Tensor) -> None:
    robot = env.scene["robot"]
    joint_ids = robot.find_joints(CONTROLLED_JOINTS, preserve_order=True)[0]
    torques = robot.data.applied_torque[:, joint_ids]

    print("actions_by_joint:")
    for index, joint_name in enumerate(CONTROLLED_JOINTS):
        print(f"  {joint_name:24s} {_stats('', actions[:, index]).lstrip(': ')}")
    print("torques_by_joint:")
    for index, joint_name in enumerate(CONTROLLED_JOINTS):
        print(f"  {joint_name:24s} {_stats('', torques[:, index]).lstrip(': ')}")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.scene["robot"]
    actions = torch.zeros_like(env.action_manager.action)
    generator = torch.Generator(device=actions.device)
    generator.manual_seed(args_cli.seed)

    print(f"\nDebug rollout: mode={args_cli.mode}, num_envs={args_cli.num_envs}, steps={args_cli.steps}")
    print(f"action_shape={tuple(actions.shape)} step_dt={env.step_dt:.6f}")

    for step in range(1, args_cli.steps + 1):
        if args_cli.mode == "random":
            actions = torch.randn(
                actions.shape,
                generator=generator,
                device=actions.device,
                dtype=actions.dtype,
            )
            actions = actions * args_cli.random_action_std
        else:
            actions = torch.zeros_like(actions)

        obs, reward, terminated, truncated, _ = env.step(actions)
        done = terminated | truncated
        obs_policy = _obs_tensor(obs)
        has_non_finite = not torch.all(torch.isfinite(obs_policy)) or not torch.all(torch.isfinite(reward))

        if step % args_cli.print_interval == 0 or step == 1 or step == args_cli.steps or has_non_finite:
            root_height = robot.data.root_pos_w[:, 2]
            root_lin_speed = torch.linalg.norm(robot.data.root_lin_vel_b, dim=1)
            root_ang_speed = torch.linalg.norm(robot.data.root_ang_vel_b, dim=1)
            joint_speed = torch.max(torch.abs(robot.data.joint_vel), dim=1)[0]
            episode_time = env.episode_length_buf.to(dtype=torch.float32) * env.step_dt

            print(f"\n[step {step}] done={done.sum().item()}/{done.numel()}")
            print(_stats("obs_policy", obs_policy))
            print(_stats("reward", reward))
            print(_stats("action", actions))
            print(_stats("episode_time", episode_time))
            print(_stats("root_height", root_height))
            print(_stats("root_lin_speed", root_lin_speed))
            print(_stats("root_ang_speed", root_ang_speed))
            print(_stats("max_joint_speed", joint_speed))
            _print_command_terms(env)
            _print_leg_and_action_terms(env, actions)
            _print_termination_terms(env)
            _print_observation_terms(env, obs_policy)
            _print_reward_terms(env)

            if has_non_finite:
                _print_non_finite_observation_details(env, obs_policy)
                _print_bad_env_state(env, actions, obs_policy)
                print("\nNon-finite rollout value detected. Stopping early.")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
