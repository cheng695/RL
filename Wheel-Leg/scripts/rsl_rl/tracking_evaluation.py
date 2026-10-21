"""Finite fixed-command evaluation, independent of training reward scales."""

import csv
import json
import math
from pathlib import Path
from statistics import mean


def summarize(samples):
    """Summarize measured, non-reset states; empty trials remain unavailable."""
    if not samples:
        return {"samples": 0}
    return {
        "samples": len(samples),
        "vx_actual_mps": mean(s["vx_actual_mps"] for s in samples),
        "vx_mae_mps": mean(abs(s["vx_actual_mps"] - s["vx_target_mps"]) for s in samples),
        "speed_xy_mps": mean(math.hypot(s["vx_actual_mps"], s["vy_actual_mps"]) for s in samples),
        "height_actual_m": mean(s["height_actual_m"] for s in samples),
        "height_bias_cm": 100 * mean(s["height_actual_m"] - s["height_target_m"] for s in samples),
        "height_mae_cm": 100 * mean(abs(s["height_actual_m"] - s["height_target_m"]) for s in samples),
    }


def run_tracking_evaluation(env, policy, app, force_velocity, args, checkpoint, output):
    """One environment; five cases with independent resets and repeated trials.

    Measurements exclude settling and terminal steps because env.step() already
    resets terminated environments. Failures are retained in the trial report.
    """
    import torch

    raw = env.unwrapped
    robot = raw.scene["robot"]
    height_term = raw.command_manager.get_term("base_height")
    velocity_term = raw.command_manager.get_term("base_velocity")
    body_ids, _ = robot.find_bodies(height_term.cfg.body_name)
    if len(body_ids) != 1:
        raise ValueError("Evaluation requires one height-tracking body")
    dt = raw.step_dt
    settle_steps = math.ceil(args.eval_settle_s / dt)
    measure_steps = math.ceil(args.eval_measure_s / dt)
    cases = [("height_low", 0.0, .25), ("height_mid", 0.0, .30),
             ("height_high", 0.0, .35), ("reverse", -.2, .30), ("forward", .2, .30)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    trials, groups = [], []
    interrupted = False
    fields = ["case", "trial", "time_s", "vx_target_mps", "vx_actual_mps", "vy_actual_mps",
              "height_target_m", "height_actual_m"]
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, vx, height in cases:
            group_samples, group_trials = [], []
            for trial in range(args.eval_trials):
                if not app.is_running():
                    interrupted = True
                    break
                env.reset()
                if hasattr(policy, "reset"):
                    policy.reset(torch.ones(raw.num_envs, dtype=torch.bool, device=raw.device))
                samples = []
                status = "complete"
                elapsed = 0.0
                for step in range(settle_steps + measure_steps):
                    if not app.is_running():
                        status = "interrupted"
                        interrupted = True
                        break
                    # Keep commands fixed through env.step(), including parent
                    # command updates and the initially sampled standing flag.
                    velocity_term.is_standing_env[:] = False
                    velocity_term.time_left[:] = float("inf")
                    height_term.time_left[:] = float("inf")
                    effective = force_velocity(env, vx, 0.0)
                    height_term.command[:, 0] = height
                    obs = env.get_observations()
                    _, _, dones, extras = env.step(policy(obs))
                    elapsed = (step + 1) * dt
                    if bool(dones[0]):
                        timeouts = extras.get("time_outs")
                        status = "timeout" if timeouts is not None and bool(timeouts[0]) else "failed"
                        break
                    if step >= settle_steps:
                        from my_robot_lab.tasks.manager_based.locomotion.velocity.mdp.observations import local_ground_height
                        ground = float(local_ground_height(raw)[0])
                        row = dict(case=name, trial=trial, time_s=elapsed,
                                   vx_target_mps=float(effective[0]),
                                   vx_actual_mps=float(robot.data.root_lin_vel_b[0, 0]),
                                   vy_actual_mps=float(robot.data.root_lin_vel_b[0, 1]),
                                   height_target_m=height,
                                   height_actual_m=float(robot.data.body_pos_w[0, body_ids[0], 2]) - ground)
                        if not all(math.isfinite(row[key]) for key in fields[2:]):
                            status = "failed"
                            break
                        samples.append(row)
                        writer.writerow(row)
                result = dict(case=name, trial=trial, status=status, elapsed_s=elapsed,
                              vx_target_mps=vx, height_target_m=height, **summarize(samples))
                trials.append(result)
                group_trials.append(result)
                group_samples.extend(samples)
                stream.flush()
                print(f"[EVAL] {name} trial {trial + 1}: {status}, {summarize(samples)}", flush=True)
                if interrupted:
                    break
            group = dict(case=name, vx_target_mps=vx, height_target_m=height,
                         trials=len(group_trials),
                         failed_trials=sum(t["status"] == "failed" for t in group_trials),
                         complete_trials=sum(t["status"] == "complete" for t in group_trials),
                         **summarize(group_samples))
            groups.append(group)
            if interrupted:
                break
    report = {
        "checkpoint": str(Path(checkpoint).resolve()), "seed": raw.cfg.seed,
        "step_dt": dt, "settle_s": settle_steps * dt, "measure_s": measure_steps * dt,
        "interrupted": interrupted,
        "note": "Metrics pool nonterminal post-settling samples, including partial failed trials. "
                "Always interpret errors together with complete/failed trial counts. "
                "Uses current task configuration; see env.yaml and agent.yaml.",
        "cases": groups, "trials": trials,
    }
    # A constant-height policy has near-zero response gain. Perfect response is 1.
    by_name = {g["case"]: g for g in groups}
    low, high = by_name.get("height_low", {}), by_name.get("height_high", {})
    if (low.get("complete_trials") == args.eval_trials and high.get("complete_trials") == args.eval_trials
            and low.get("samples", 0) and high.get("samples", 0)):
        report["height_response_gain"] = (high["height_actual_m"] - low["height_actual_m"]) / .10
    reverse, forward = by_name.get("reverse", {}), by_name.get("forward", {})
    if (reverse.get("complete_trials") == args.eval_trials and forward.get("complete_trials") == args.eval_trials
            and reverse.get("samples", 0) and forward.get("samples", 0)):
        report["velocity_response_gain"] = (forward["vx_actual_mps"] - reverse["vx_actual_mps"]) / .4
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[EVAL] Report: {output / 'report.json'}", flush=True)
    print(f"[EVAL] Height response gain: {report.get('height_response_gain', 'unavailable')}; "
          f"velocity response gain: {report.get('velocity_response_gain', 'unavailable')}")
