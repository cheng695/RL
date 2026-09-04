"""Plot a low-dimensional 3D training trajectory from TensorBoard scalars.

This is not the true PPO parameter-space loss landscape.  It is a diagnostic
projection of training metrics, useful for checking whether learning keeps
moving toward a better region or circulates around a likely local optimum.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tensorboard.backend.event_processing import event_accumulator


DEFAULT_RUN_ROOT = Path("logs/rsl_rl/wheel_leg")


parser = argparse.ArgumentParser(description="Plot 3D metric trajectory from an RSL-RL TensorBoard run.")
parser.add_argument("--run", type=str, default=None, help="Run name or path. Defaults to the latest run directory.")
parser.add_argument("--log_root", type=str, default=str(DEFAULT_RUN_ROOT), help="Root directory containing run folders.")
parser.add_argument("--x_tag", type=str, default="Metrics/base_height/error_height", help="TensorBoard scalar tag for x.")
parser.add_argument("--y_tag", type=str, default="Metrics/base_velocity/error_vel_xy", help="TensorBoard scalar tag for y.")
parser.add_argument("--z_tag", type=str, default="Train/mean_reward", help="TensorBoard scalar tag for z.")
parser.add_argument("--color_tag", type=str, default=None, help="Optional scalar tag for point color. Defaults to step.")
parser.add_argument("--smooth", type=int, default=5, help="Moving-average window. Use 1 to disable smoothing.")
parser.add_argument("--arrow_stride", type=int, default=50, help="Draw one trajectory arrow every N points. Use 0 to disable.")
parser.add_argument("--surface", action="store_true", default=True, help="Also plot a triangulated objective surface.")
parser.add_argument("--no_surface", action="store_false", dest="surface", help="Disable the triangulated surface plot.")
parser.add_argument(
    "--surface_mode",
    choices=("cost", "reward"),
    default="cost",
    help="Use -z_tag as a minimization cost surface, or z_tag as a reward surface.",
)
parser.add_argument("--output", type=str, default=None, help="Output prefix. Defaults to <run>/training_landscape.")
parser.add_argument("--list_tags", action="store_true", help="List scalar tags and exit.")
args = parser.parse_args()


def _resolve_run_path(run: str | None, log_root: str) -> Path:
    if run is not None:
        path = Path(run)
        if path.exists():
            return path
        candidate = Path(log_root) / run
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Run path not found: {run}")

    root = Path(log_root)
    runs = sorted(path for path in root.iterdir() if path.is_dir() and any(path.glob("events.out.tfevents.*")))
    if not runs:
        raise FileNotFoundError(f"No TensorBoard runs found under: {root}")
    return runs[-1]


def _load_scalars(run_path: Path):
    accumulator = event_accumulator.EventAccumulator(
        str(run_path),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    accumulator.Reload()
    return accumulator


def _series(accumulator, tag: str) -> dict[int, float]:
    if tag not in accumulator.Tags().get("scalars", []):
        raise KeyError(f"Scalar tag not found: {tag}")
    return {event.step: event.value for event in accumulator.Scalars(tag)}


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result = []
    total = 0.0
    queue = []
    for value in values:
        queue.append(value)
        total += value
        if len(queue) > window:
            total -= queue.pop(0)
        result.append(total / len(queue))
    return result


def _aligned_series(accumulator, tags: list[str]) -> tuple[list[int], list[list[float]]]:
    series_by_tag = [_series(accumulator, tag) for tag in tags]
    common_steps = sorted(set.intersection(*(set(series) for series in series_by_tag)))
    if not common_steps:
        raise RuntimeError(f"No common steps for tags: {tags}")
    values = [[series[step] for step in common_steps] for series in series_by_tag]
    return common_steps, values


def _unique_xy_surface_points(x: list[float], y: list[float], z: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average z values for nearly identical x/y bins to keep triangulation stable."""
    bins: dict[tuple[float, float], list[float]] = {}
    for x_value, y_value, z_value in zip(x, y, z, strict=True):
        if not (np.isfinite(x_value) and np.isfinite(y_value) and np.isfinite(z_value)):
            continue
        key = (round(float(x_value), 5), round(float(y_value), 5))
        bins.setdefault(key, []).append(float(z_value))

    xs, ys, zs = [], [], []
    for (x_value, y_value), z_values in bins.items():
        xs.append(x_value)
        ys.append(y_value)
        zs.append(sum(z_values) / len(z_values))
    return np.asarray(xs), np.asarray(ys), np.asarray(zs)


def _plot_surface(
    output_prefix: Path,
    run_name: str,
    steps: list[int],
    x: list[float],
    y: list[float],
    z: list[float],
    x_label: str,
    y_label: str,
    z_label: str,
    smooth: int,
    arrow_stride: int,
    surface_mode: str,
) -> Path:
    surface_z = [-value for value in z] if surface_mode == "cost" else z
    surface_label = f"-{z_label} (cost)" if surface_mode == "cost" else z_label
    xs, ys, zs = _unique_xy_surface_points(x, y, surface_z)
    if len(xs) < 4:
        raise RuntimeError("Need at least four unique x/y points to draw a surface.")

    triangulation = mtri.Triangulation(xs, ys)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surface = ax.plot_trisurf(
        triangulation,
        zs,
        cmap="turbo",
        linewidth=0.15,
        edgecolor="0.35",
        alpha=0.82,
        antialiased=True,
    )

    z_floor = float(np.nanmin(zs))
    z_range = float(np.nanmax(zs) - z_floor)
    contour_offset = z_floor - 0.08 * z_range if z_range > 0.0 else z_floor - 1.0
    ax.tricontour(
        triangulation,
        zs,
        levels=18,
        zdir="z",
        offset=contour_offset,
        cmap="turbo",
        linewidths=0.8,
        alpha=0.8,
    )

    ax.plot(x, y, surface_z, color="red", linewidth=1.8, marker="o", markersize=2.8, label="training path")
    if arrow_stride > 0 and len(steps) > arrow_stride:
        arrow_indices = range(0, len(steps) - arrow_stride, arrow_stride)
        ax.quiver(
            [x[index] for index in arrow_indices],
            [y[index] for index in arrow_indices],
            [surface_z[index] for index in arrow_indices],
            [x[index + arrow_stride] - x[index] for index in arrow_indices],
            [y[index + arrow_stride] - y[index] for index in arrow_indices],
            [surface_z[index + arrow_stride] - surface_z[index] for index in arrow_indices],
            length=1.0,
            normalize=False,
            color="black",
            linewidth=0.9,
            alpha=0.8,
        )
    ax.scatter([x[0]], [y[0]], [surface_z[0]], c="white", edgecolor="red", s=70, label="start")
    ax.scatter([x[-1]], [y[-1]], [surface_z[-1]], c="lime", edgecolor="black", s=70, label="end")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(surface_label)
    ax.set_zlim(contour_offset, float(np.nanmax(zs)))
    ax.set_title(f"Projected training surface: {run_name} | smoothing={smooth}")
    ax.legend(loc="best")
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.08, label=surface_label)
    fig.tight_layout()

    suffix = "_surface_cost.png" if surface_mode == "cost" else "_surface_reward.png"
    surface_path = output_prefix.with_name(output_prefix.name + suffix)
    fig.savefig(surface_path, dpi=180)
    plt.close(fig)
    return surface_path


def main() -> None:
    run_path = _resolve_run_path(args.run, args.log_root)
    accumulator = _load_scalars(run_path)

    if args.list_tags:
        print(f"[INFO] Run: {run_path}")
        for tag in accumulator.Tags().get("scalars", []):
            print(tag)
        return

    tags = [args.x_tag, args.y_tag, args.z_tag]
    color_tag = args.color_tag
    if color_tag is not None and color_tag not in tags:
        tags.append(color_tag)

    steps, values = _aligned_series(accumulator, tags)
    x, y, z = [_moving_average(values[index], args.smooth) for index in range(3)]
    color = steps if color_tag is None else _moving_average(values[tags.index(color_tag)], args.smooth)

    output_prefix = Path(args.output) if args.output else run_path / "training_landscape"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["step", args.x_tag, args.y_tag, args.z_tag, color_tag or "step"])
        for row in zip(steps, x, y, z, color, strict=True):
            writer.writerow(row)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(x, y, z, color="0.35", linewidth=1.0, alpha=0.7)
    points = ax.scatter(x, y, z, c=color, cmap="viridis", s=18)
    if args.arrow_stride > 0 and len(steps) > args.arrow_stride:
        arrow_indices = range(0, len(steps) - args.arrow_stride, args.arrow_stride)
        ax.quiver(
            [x[index] for index in arrow_indices],
            [y[index] for index in arrow_indices],
            [z[index] for index in arrow_indices],
            [x[index + args.arrow_stride] - x[index] for index in arrow_indices],
            [y[index + args.arrow_stride] - y[index] for index in arrow_indices],
            [z[index + args.arrow_stride] - z[index] for index in arrow_indices],
            length=1.0,
            normalize=False,
            color="tab:orange",
            linewidth=0.8,
            alpha=0.7,
        )
    ax.scatter([x[0]], [y[0]], [z[0]], c="tab:red", s=55, label="start")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], c="tab:green", s=55, label="end")
    ax.set_xlabel(args.x_tag)
    ax.set_ylabel(args.y_tag)
    ax.set_zlabel(args.z_tag)
    ax.set_title("3D metric trajectory")
    ax.legend(loc="best")
    fig.colorbar(points, ax=ax, shrink=0.65, label=color_tag or "step")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(steps, x, label=args.x_tag)
    ax2.plot(steps, y, label=args.y_tag)
    ax2.set_xlabel("step")
    ax2.set_title("Errors over training")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(steps, z, label=args.z_tag, color="tab:green")
    ax3.set_xlabel("step")
    ax3.set_title("Objective proxy over training")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    fig.suptitle(f"Run: {run_path.name} | smoothing={args.smooth}")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    surface_path = None
    if args.surface:
        surface_path = _plot_surface(
            output_prefix=output_prefix,
            run_name=run_path.name,
            steps=steps,
            x=x,
            y=y,
            z=z,
            x_label=args.x_tag,
            y_label=args.y_tag,
            z_label=args.z_tag,
            smooth=args.smooth,
            arrow_stride=args.arrow_stride,
            surface_mode=args.surface_mode,
        )

    print(f"[INFO] Run: {run_path}")
    print(f"[INFO] Points: {len(steps)}")
    print(f"[INFO] Wrote: {csv_path}")
    print(f"[INFO] Wrote: {png_path}")
    if surface_path is not None:
        print(f"[INFO] Wrote: {surface_path}")
    print("[INFO] Interpretation: a shrinking path toward lower errors and higher reward is healthy;")
    print("       a tight loop or flat cluster with unchanged errors suggests a local optimum or reward loophole.")


if __name__ == "__main__":
    main()
