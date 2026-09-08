#!/usr/bin/env python3
"""Small local dashboard for RSL-RL TensorBoard runs."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"
STATIC_ROOT = Path(__file__).parent


def run_dirs(log_root: Path) -> list[Path]:
    return sorted(
        (path for path in log_root.glob("*/*") if path.is_dir() and list(path.glob("events.out.tfevents.*"))),
        key=lambda path: path.name,
        reverse=True,
    )


def newest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = list(run_dir.glob("model_*.pt"))
    return max(checkpoints, key=lambda path: path.stat().st_mtime) if checkpoints else None


def load_run(run_dir: Path) -> dict:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for tag in accumulator.Tags().get("scalars", []):
        series = [
            {"step": item.step, "wall_time": item.wall_time, "value": item.value}
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

    checkpoint = newest_checkpoint(run_dir)
    return {
        "id": str(run_dir.relative_to(DEFAULT_LOG_ROOT)),
        "name": run_dir.name,
        "experiment": run_dir.parent.name,
        "checkpoint": checkpoint.name if checkpoint else None,
        "checkpoint_size": checkpoint.stat().st_size if checkpoint else None,
        "event_files": [path.name for path in run_dir.glob("events.out.tfevents.*")],
        "params": {
            name: (run_dir / "params" / name).read_text(encoding="utf-8", errors="replace")
            for name in ("env.yaml", "agent.yaml")
            if (run_dir / "params" / name).exists()
        },
        "scalar_tags": sorted(scalars),
        "summary": summary,
        "scalars": scalars,
    }


class Handler(BaseHTTPRequestHandler):
    log_root = DEFAULT_LOG_ROOT

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/runs":
            self.send_json(self.list_runs())
            return
        if parsed.path == "/api/run":
            run_id = parse_qs(parsed.query).get("id", [""])[0]
            self.send_run(run_id)
            return
        self.send_static(parsed.path)

    def list_runs(self) -> dict:
        runs = []
        for path in run_dirs(self.log_root):
            checkpoint = newest_checkpoint(path)
            event_files = list(path.glob("events.out.tfevents.*"))
            runs.append(
                {
                    "id": str(path.relative_to(self.log_root)),
                    "name": path.name,
                    "experiment": path.parent.name,
                    "checkpoint": checkpoint.name if checkpoint else None,
                    "checkpoint_size": checkpoint.stat().st_size if checkpoint else None,
                    "event_files": len(event_files),
                    "mtime": max(file.stat().st_mtime for file in event_files),
                }
            )
        return {"runs": runs}

    def send_run(self, run_id: str) -> None:
        if not run_id or re.search(r"(^|/|\\)\.\.($|/|\\)", run_id):
            self.send_error(400, "invalid run id")
            return
        run_dir = self.log_root / run_id
        if not run_dir.is_dir() or not run_dir.resolve().is_relative_to(self.log_root.resolve()):
            self.send_error(404, "run not found")
            return
        self.send_json(load_run(run_dir))

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in ("", "/") else request_path.lstrip("/")
        path = (STATIC_ROOT / relative).resolve()
        if not path.is_file() or not path.is_relative_to(STATIC_ROOT.resolve()):
            self.send_error(404, "not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.log_root = args.log_root.resolve()
    if not Handler.log_root.is_dir():
        raise SystemExit(f"log root does not exist: {Handler.log_root}")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"RL dashboard: http://{args.host}:{args.port}")
    print(f"Reading runs from: {Handler.log_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
