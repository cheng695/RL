from __future__ import annotations

import argparse
import random


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add common RSL-RL command line arguments."""
    group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL.")
    group.add_argument("--experiment_name", type=str, default=None, help="Override experiment folder name.")
    group.add_argument("--run_name", type=str, default=None, help="Run name suffix.")
    group.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
    group.add_argument("--load_run", type=str, default=None, help="Run folder to load from, such as 2026-08-14_12-00-00.")
    group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file name or path.")
    group.add_argument("--logger", type=str, default=None, choices={"tensorboard", "wandb", "neptune"}, help="Logger type.")
    group.add_argument("--log_project_name", type=str, default=None, help="Project name for wandb/neptune.")


def update_rsl_rl_cfg(agent_cfg, args_cli: argparse.Namespace):
    """Apply CLI overrides to an RSL-RL runner cfg."""
    if args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    if getattr(agent_cfg, "logger", None) in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    return agent_cfg
