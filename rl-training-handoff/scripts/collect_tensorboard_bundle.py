#!/usr/bin/env python3
"""Export TensorBoard scalar series and run metadata into one JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REQUIRED_METADATA_KEYS = (
    "run",
    "checkpoint",
    "event_files",
    "env",
    "obs",
    "action",
    "reward",
    "training",
    "preview",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional JSON object with task/config/checkpoint metadata.",
    )
    return parser.parse_args()


def collect(logdir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    accumulator = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, dict[str, Any]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        series = [
            {
                "step": item.step,
                "wall_time": item.wall_time,
                "value": item.value,
            }
            for item in accumulator.Scalars(tag)
        ]
        scalars[tag] = series
        values = [item["value"] for item in series]
        if values:
            summary[tag] = {
                "count": len(values),
                "first": values[0],
                "last": values[-1],
                "minimum": min(values),
                "maximum": max(values),
            }
    return {
        "schema_version": 1,
        "metadata": metadata,
        "tensorboard": {
            "scalar_tags": sorted(scalars),
            "summary": summary,
            "scalars": scalars,
        },
    }


def main() -> None:
    args = parse_args()
    if not args.logdir.is_dir():
        raise SystemExit(f"logdir does not exist: {args.logdir}")
    metadata: dict[str, Any] = {}
    if args.metadata:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise SystemExit("--metadata must contain a JSON object")
        missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
        if missing:
            raise SystemExit(
                "--metadata is missing required keys: " + ", ".join(missing)
            )
    bundle = collect(args.logdir, metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
