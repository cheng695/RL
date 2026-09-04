"""验证 base height 命令跟踪：加载训练 checkpoint 跑一段 rollout，记录并画图。

与 train.py 一样从 CWD 相对解析 logs 路径，请在项目根目录的上级 RL 目录下运行：

    cd /home/whc/桌面/RL
    python3 Wheel-Leg/scripts/rsl_rl/verify_base_height.py \
        --load_run 2026-08-19_12-33-16 --checkpoint model_1403.pt --steps 2400

checkpoint 也支持绝对路径（--checkpoint /完整/路径/model_xxx.pt，此时 --load_run 可省略）。

产物：
    base_height_trace.csv —— 每步一行：t, cmd_z, z, z_err, cmd_vz, vz, desired_vz, vz_err, phase
    base_height_trace.png —— 上：命令 vs 实际高度；中：高度误差；下：vz vs desired_vz
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata as metadata
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Verify base-height command tracking with a trained checkpoint.")
parser.add_argument("--task", type=str, default="WheelLeg-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--steps", type=int, default=2400, help="Rollout policy steps (20s episode = 1200 steps).")
parser.add_argument("--output", type=str, default="base_height_trace", help="Output file prefix for csv/png.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
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

    # 与 base_height_velocity_error_l2 相同的反馈参数（优先读当前配置接线）
    kp = 0.5
    max_desired_vz = 0.05
    vel_term = getattr(env_cfg.rewards, "base_height_vel", None)
    if vel_term is not None:
        kp = vel_term.params.get("kp", kp)
        max_desired_vz = vel_term.params.get("max_desired_velocity", max_desired_vz)
    print(f"[INFO] desired_vz 反馈参数: kp={kp}, max_desired_vz={max_desired_vz}")

    sim = env.unwrapped.scene["robot"]
    command_manager = env.unwrapped.command_manager
    dt = env.unwrapped.step_dt

    rows = []
    obs = env.get_observations()

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        # step 之后读取：state 与 command 都对应 t = (step+1)*dt 时刻
        cmd = command_manager.get_command("base_height")[0]  # [z, rate]
        cmd_z = float(cmd[0])
        cmd_vz = float(cmd[1])
        z = float(sim.data.root_pos_w[0, 2])
        vz = float(sim.data.root_lin_vel_w[0, 2])
        desired_vz = float(
            torch.clamp(cmd[1] + kp * (cmd[0] - z), min=-max_desired_vz, max=max_desired_vz)
        )
        phase = "+" if cmd_vz > 0 else ("-" if cmd_vz < 0 else "0")

        rows.append(
            {
                "step": step,
                "t": round(step * dt, 4),
                "cmd_z": round(cmd_z, 5),
                "z": round(z, 5),
                "z_err": round(z - cmd_z, 5),
                "cmd_vz": round(cmd_vz, 5),
                "vz": round(vz, 5),
                "desired_vz": round(desired_vz, 5),
                "vz_err": round(vz - desired_vz, 5),
                "phase": phase,
                "done": bool(dones[0]),
            }
        )

    env.close()

    # ---- 统计 ----
    z_err = torch.tensor([r["z_err"] for r in rows])
    vz_err = torch.tensor([r["vz_err"] for r in rows])
    phase = torch.tensor([1 if r["phase"] == "+" else (-1 if r["phase"] == "-" else 0) for r in rows])

    def rms(mask: torch.Tensor, x: torch.Tensor) -> float:
        if not mask.any():
            return float("nan")
        return float(torch.sqrt(torch.mean(torch.square(x[mask]))))

    all_mask = torch.ones_like(z_err, dtype=torch.bool)
    up_mask = phase > 0
    down_mask = phase < 0
    print("\n===== 高度跟踪验证 =====")
    print(f"steps: {len(rows)}, dt: {dt:.4f}s, 总时长: {len(rows) * dt:.1f}s")
    print(f"rms 高度误差       : {rms(all_mask, z_err) * 1000:.1f} mm")
    print(f"  - 上升段         : {rms(up_mask, z_err) * 1000:.1f} mm")
    print(f"  - 下降段         : {rms(down_mask, z_err) * 1000:.1f} mm")
    print(f"rms |vz - desired| : {rms(all_mask, vz_err) * 100:.1f} cm/s")
    print(f"max |z_err|        : {float(z_err.abs().max()) * 1000:.1f} mm")
    done_steps = sum(1 for r in rows if r["done"])
    print(f"episode 重置次数   : {done_steps}")

    # ---- 存 CSV ----
    csv_path = f"{args_cli.output}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV 已保存: {csv_path}")

    # ---- 画图 ----
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib 不可用，跳过画图；可用 CSV 自行分析。")
        return

    t = [r["t"] for r in rows]
    cmd_z_list = [r["cmd_z"] for r in rows]
    z_list = [r["z"] for r in rows]
    z_err_list = [r["z_err"] for r in rows]
    vz_list = [r["vz"] for r in rows]
    desired_vz_list = [r["desired_vz"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(t, cmd_z_list, label="cmd_z", color="tab:orange")
    axes[0].plot(t, z_list, label="base z", color="tab:blue")
    axes[0].set_ylabel("height [m]")
    axes[0].set_title("Base height: command vs actual")
    axes[0].legend()
    axes[1].plot(t, [e * 1000 for e in z_err_list], color="tab:red")
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_ylabel("z error [mm]")
    axes[1].set_title("Height tracking error (z - cmd_z)")
    axes[2].plot(t, vz_list, label="vz_w", color="tab:blue")
    axes[2].plot(t, desired_vz_list, label="desired_vz", color="tab:green", ls="--")
    axes[2].set_ylabel("vz [m/s]")
    axes[2].set_xlabel("t [s]")
    axes[2].set_title("Vertical velocity vs desired (kp feedback)")
    axes[2].legend()
    fig.tight_layout()

    png_path = f"{args_cli.output}.png"
    fig.savefig(png_path, dpi=150)
    print(f"[INFO] 图已保存: {png_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
