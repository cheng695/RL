"""Debug a trained WheelLeg policy rollout and locate non-finite observations."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Run a trained WheelLeg policy and print NaN diagnostics.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--steps", type=int, default=2000, help="Maximum rollout steps.")
parser.add_argument("--print_interval", type=int, default=100, help="Print summary every N steps.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import envs.wheel_leg  # noqa: F401


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


def _resolve_checkpoint(log_root_path: str, agent_cfg) -> str:
    checkpoint = Path(agent_cfg.load_checkpoint) if agent_cfg.load_checkpoint else None
    if checkpoint is not None and checkpoint.is_absolute():
        return str(checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _obs_tensor(obs) -> torch.Tensor:
    if hasattr(obs, "get"):
        try:
            policy_obs = obs.get("policy", None)
            if policy_obs is not None:
                return policy_obs
        except TypeError:
            pass
    if isinstance(obs, dict):
        return obs["policy"] if "policy" in obs else obs[next(iter(obs))]
    return obs


def _print_observation_terms(env, obs_policy: torch.Tensor) -> None:
    observation_manager = env.observation_manager
    term_names = observation_manager.active_terms["policy"]
    term_dims = observation_manager.group_obs_term_dim["policy"]
    concat_dim = observation_manager._group_obs_concatenate_dim["policy"]
    if concat_dim > 0:
        concat_dim -= 1

    print("observation_terms:")
    index = 0
    for term_name, term_dim in zip(term_names, term_dims, strict=True):
        width = term_dim[concat_dim]
        term_value = obs_policy.narrow(dim=concat_dim, start=index, length=width)
        print(f"  {term_name:24s} {_stats('', term_value).lstrip(': ')}")
        if not torch.all(torch.isfinite(term_value)):
            bad_indices = torch.nonzero(~torch.isfinite(term_value), as_tuple=False)
            bad_env_ids = torch.unique(bad_indices[:, 0]).detach().cpu().tolist()
            bad_cols = torch.unique(bad_indices[:, 1]).detach().cpu().tolist()
            print(
                f"    non_finite start={index} width={width} "
                f"bad_env_ids={bad_env_ids[:16]} bad_cols={bad_cols[:16]} "
                f"bad_count={bad_indices.shape[0]}/{term_value.numel()}"
            )
        index += width


def _print_bad_env_state(env, actions: torch.Tensor, obs_policy: torch.Tensor) -> None:
    robot = env.scene["robot"]
    bad_env_ids = torch.nonzero(torch.any(~torch.isfinite(obs_policy), dim=1), as_tuple=False).flatten()
    if bad_env_ids.numel() == 0:
        return
    ids = bad_env_ids[:8]
    print("bad_env_state:")
    print(f"  bad_env_ids={bad_env_ids[:16].detach().cpu().tolist()} count={bad_env_ids.numel()}")
    print(_stats("  root_pos_w", robot.data.root_pos_w[ids]))
    print(_stats("  root_quat_w", robot.data.root_quat_w[ids]))
    print(_stats("  root_lin_vel_b", robot.data.root_lin_vel_b[ids]))
    print(_stats("  root_ang_vel_b", robot.data.root_ang_vel_b[ids]))
    print(_stats("  projected_gravity_b", robot.data.projected_gravity_b[ids]))
    print(_stats("  joint_pos", robot.data.joint_pos[ids]))
    print(_stats("  joint_vel", robot.data.joint_vel[ids]))
    print(_stats("  action", actions[ids]))


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
    resume_path = handle_deprecated_rsl_rl_checkpoint(_resolve_checkpoint(log_root_path, agent_cfg), installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env_cfg.log_dir = os.path.dirname(resume_path)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    for step in range(1, args_cli.steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, reward, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        obs_policy = _obs_tensor(obs)
        has_bad_obs = not torch.all(torch.isfinite(obs_policy))
        has_bad_reward = not torch.all(torch.isfinite(reward))
        if step == 1 or step % args_cli.print_interval == 0 or has_bad_obs or has_bad_reward:
            print(f"\n[step {step}] done={dones.sum().item()}/{dones.numel()}")
            print(_stats("obs_policy", obs_policy))
            print(_stats("reward", reward))
            print(_stats("action", actions))
            _print_observation_terms(env.unwrapped, obs_policy)
        if has_bad_obs or has_bad_reward:
            _print_bad_env_state(env.unwrapped, actions, obs_policy)
            print("\nNon-finite policy rollout value detected. Stopping early.")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
