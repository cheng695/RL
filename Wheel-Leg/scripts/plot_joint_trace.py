"""Plot target, measured state, and estimated PD torque from a Play CSV trace."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV created by play.py --record_joint.")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: next to the CSV).")
    args = parser.parse_args()

    csv_path = args.csv_path.expanduser().resolve()
    with csv_path.open("r", newline="", encoding="utf-8") as trace_file:
        rows = list(csv.DictReader(trace_file))
    if not rows:
        raise ValueError(f"Trace contains no samples: {csv_path}")

    time_s = [float(row["time_s"]) for row in rows]
    q_target = [float(row["q_target_rad"]) for row in rows]
    q_actual = [float(row["q_actual_rad"]) for row in rows]
    qdot_actual = [float(row["qdot_actual_rad_s"]) for row in rows]
    tau_estimate = [float(row["tau_pd_estimate_nm"]) for row in rows]
    joint_name = rows[0]["joint_name"]

    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(time_s, q_target, label="Target angle", linewidth=1.2)
    axes[0].plot(time_s, q_actual, label="Actual angle", linewidth=1.0)
    axes[0].set_ylabel("Angle (rad)")
    axes[0].legend(loc="best")

    axes[1].plot(time_s, qdot_actual, color="tab:orange", label="Actual angular velocity", linewidth=1.0)
    axes[1].set_ylabel("Velocity (rad/s)")
    axes[1].legend(loc="best")

    axes[2].plot(time_s, tau_estimate, color="tab:green", label="Clipped PD torque estimate", linewidth=1.0)
    axes[2].set_ylabel("Torque estimate (N·m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="best")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.suptitle(f"Joint trace: {joint_name}")

    output_path = args.output.expanduser().resolve() if args.output else csv_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    print(f"[INFO] Plot saved: {output_path}")


if __name__ == "__main__":
    main()
