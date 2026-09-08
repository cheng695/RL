"""Play a trained UZ-05 RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import io
import importlib.metadata as metadata
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_SOURCE = PROJECT_ROOT / "source" / "my_robot_lab"
if str(EXTENSION_SOURCE) not in sys.path:
    sys.path.insert(0, str(EXTENSION_SOURCE))

from isaaclab.app import AppLauncher

import cli_args

parser = argparse.ArgumentParser(description="Play a trained UZ-05 RSL-RL checkpoint.")
parser.add_argument("--task", type=str, default="MyRobot-Velocity-Flat-Play-v0", help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real_time", action="store_true", default=False, help="Try to run at real-time speed.")
parser.add_argument("--command_vx", type=float, default=None, help="Override forward velocity command in m/s.")
parser.add_argument("--command_yaw", type=float, default=None, help="Override yaw-rate command in rad/s.")
parser.add_argument("--print_state", action="store_true", default=False, help="Print env0 state while playing.")
parser.add_argument("--print_interval_s", type=float, default=0.5, help="Seconds between state prints.")
parser.add_argument("--web_port", type=int, default=8766, help="Local web preview port.")
parser.add_argument("--no_web", action="store_true", default=False, help="Disable the browser preview bridge.")
parser.add_argument("--web_frame_fps", type=float, default=15.0, help="Viewport capture FPS for the browser preview.")
parser.add_argument("--web_frame_quality", type=int, default=92, help="JPEG quality for browser viewport frames.")
parser.add_argument("--web_frame_width", type=int, default=1920, help="Browser viewport capture width.")
parser.add_argument("--web_frame_height", type=int, default=1080, help="Browser viewport capture height.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaacsim.core.utils.extensions import enable_extension
from PIL import Image
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import my_robot_lab  # noqa: F401

enable_extension("isaacsim.asset.importer.mjcf")


class LivePreviewState:
    """Thread-safe command/state bridge between Isaac Lab and the browser."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.command_vx = 0.0
        self.command_yaw = 0.0
        self.command_height = 0.30
        self.paused = False
        self.reset_requested = False
        self.state = {"running": False, "step": 0}
        self.frame = None
        self.frame_content_type = "image/jpeg"
        self.frame_index = 0
        self.frame_size = None
        self.camera = {
            "follow": True,
            "yaw": -2.35,
            "pitch": 0.35,
            "distance": 3.0,
            "target_height": 0.35,
        }

    def update_command(self, payload: dict) -> None:
        with self.lock:
            for key, limit in (("command_vx", 1.0), ("command_yaw", 2.0), ("command_height", 0.4)):
                if key in payload:
                    value = float(payload[key])
                    if key == "command_height":
                        value = max(0.25, min(limit, value))
                    else:
                        value = max(-limit, min(limit, value))
                    setattr(self, key, value)

    def update_camera(self, payload: dict) -> None:
        with self.lock:
            camera = dict(self.camera)
            if "follow" in payload:
                camera["follow"] = bool(payload["follow"])
            if "yaw_delta" in payload:
                camera["yaw"] = float(camera["yaw"]) + float(payload["yaw_delta"])
            if "pitch_delta" in payload:
                camera["pitch"] = float(camera["pitch"]) + float(payload["pitch_delta"])
            if "distance_delta" in payload:
                camera["distance"] = float(camera["distance"]) + float(payload["distance_delta"])
            for key in ("yaw", "pitch", "distance", "target_height"):
                if key in payload:
                    camera[key] = float(payload[key])
            camera["pitch"] = max(-0.1, min(1.25, float(camera["pitch"])))
            camera["distance"] = max(0.8, min(12.0, float(camera["distance"])))
            camera["target_height"] = max(0.0, min(2.0, float(camera["target_height"])))
            self.camera = camera

    def command(self) -> tuple[float, float, float]:
        with self.lock:
            return self.command_vx, self.command_yaw, self.command_height

    def camera_snapshot(self) -> dict:
        with self.lock:
            return dict(self.camera)

    def toggle_pause(self) -> None:
        with self.lock:
            self.paused = not self.paused

    def request_reset(self) -> None:
        with self.lock:
            self.reset_requested = True

    def consume_reset(self) -> bool:
        with self.lock:
            requested = self.reset_requested
            self.reset_requested = False
            return requested

    def update_state(self, state: dict) -> None:
        with self.lock:
            self.state = state

    def update_frame(self, frame: bytes, content_type: str, width: int, height: int) -> None:
        with self.lock:
            self.frame = frame
            self.frame_content_type = content_type
            self.frame_index += 1
            self.frame_size = [width, height]

    def snapshot_frame(self) -> tuple[bytes | None, str, int, list[int] | None]:
        with self.lock:
            return self.frame, self.frame_content_type, self.frame_index, self.frame_size

    def snapshot(self) -> dict:
        with self.lock:
            return {
                **self.state,
                "paused": self.paused,
                "command": {"vx": self.command_vx, "yaw": self.command_yaw, "height": self.command_height},
                "camera": dict(self.camera),
                "frame_index": self.frame_index,
                "frame_size": self.frame_size,
            }


class ViewportFrameStreamer:
    """Capture Isaac Sim viewport frames for the browser preview."""

    def __init__(
        self,
        preview_state: LivePreviewState,
        fps: float,
        jpeg_quality: int,
        width: int,
        height: int,
    ) -> None:
        self.preview_state = preview_state
        self.frame_interval = 1.0 / max(fps, 1.0e-6)
        self.jpeg_quality = max(1, min(95, int(jpeg_quality)))
        self.width = max(320, int(width))
        self.height = max(240, int(height))
        self.next_capture_time = 0.0
        self.capture_pending = False
        self.viewport = None
        self.viewport_window = None
        self.enabled = fps > 0.0
        self.warning_printed = False
        self.resolution_warning_printed = False

    def maybe_capture(self) -> None:
        if not self.enabled or self.capture_pending or time.time() < self.next_capture_time:
            return
        try:
            if self.viewport is None:
                from omni.kit.viewport.utility import create_viewport_window, get_active_viewport

                self.viewport_window = create_viewport_window(
                    "Browser Preview Capture",
                    width=self.width,
                    height=self.height,
                    position_x=40,
                    position_y=40,
                    resolution=(self.width, self.height),
                )
                self.viewport = (
                    self.viewport_window.viewport_api if self.viewport_window is not None else get_active_viewport()
                )
            if self.viewport is None:
                self._warn_once("[WARN] No active Isaac Sim viewport available for browser frame capture.")
                return
            self._sync_resolution()

            from omni.kit.viewport.utility import capture_viewport_to_buffer

            self.capture_pending = True
            self.next_capture_time = time.time() + self.frame_interval
            capture_viewport_to_buffer(self.viewport, self._on_capture)
        except Exception as exc:
            self.capture_pending = False
            self.next_capture_time = time.time() + 1.0
            self._warn_once(f"[WARN] Viewport frame capture disabled: {exc}")

    def _on_capture(self, buffer, buffer_size, width, height, byte_format) -> None:
        self.capture_pending = False
        try:
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            pointer = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
            rgba = ctypes.string_at(pointer, buffer_size)
            image = Image.frombytes("RGBA", (width, height), rgba)
            image = image.convert("RGB")
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=self.jpeg_quality, optimize=True)
            self.preview_state.update_frame(output.getvalue(), "image/jpeg", width, height)
        except Exception as exc:
            self._warn_once(f"[WARN] Failed to encode viewport frame: {exc}")

    def _warn_once(self, message: str) -> None:
        if not self.warning_printed:
            print(message)
            self.warning_printed = True

    def _sync_resolution(self) -> None:
        try:
            self.viewport.resolution = (self.width, self.height)
            if hasattr(self.viewport, "resolution_scale"):
                self.viewport.resolution_scale = 1.0
        except Exception as exc:
            if not self.resolution_warning_printed:
                print(f"[WARN] Could not set browser capture resolution: {exc}")
                self.resolution_warning_printed = True


class LiveCameraController:
    """Move the Isaac Sim viewport camera from browser orbit controls."""

    def __init__(self, env, preview_state: LivePreviewState) -> None:
        self.env = env
        self.preview_state = preview_state
        self.warning_printed = False

    def update(self) -> None:
        camera = self.preview_state.camera_snapshot()
        if not camera.get("follow", True):
            return
        try:
            robot = self.env.unwrapped.scene["robot"]
            target = robot.data.root_pos_w[0].detach().cpu()
            yaw = float(camera["yaw"])
            pitch = float(camera["pitch"])
            distance = float(camera["distance"])
            target_z = float(target[2].item()) + float(camera.get("target_height", 0.35))
            horizontal = distance * math.cos(pitch)
            eye = [
                float(target[0].item()) + horizontal * math.cos(yaw),
                float(target[1].item()) + horizontal * math.sin(yaw),
                target_z + distance * math.sin(pitch),
            ]
            look_at = [float(target[0].item()), float(target[1].item()), target_z]
            self.env.unwrapped.sim.set_camera_view(eye=eye, target=look_at)
        except Exception as exc:
            if not self.warning_printed:
                print(f"[WARN] Live camera follow disabled: {exc}")
                self.warning_printed = True


def start_preview_server(preview_state: LivePreviewState, port: int) -> ThreadingHTTPServer:
    page = PROJECT_ROOT.parent / "rl-training-handoff" / "dashboard" / "live.html"

    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/live-state":
                self.send_json(preview_state.snapshot())
                return
            if path == "/api/frame":
                self.send_frame()
                return
            if path in ("/", "/live.html"):
                data = page.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

        def send_frame(self) -> None:
            frame, content_type, frame_index, frame_size = preview_state.snapshot_frame()
            if frame is None:
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Index", str(frame_index))
            if frame_size is not None:
                self.send_header("X-Frame-Size", f"{frame_size[0]}x{frame_size[1]}")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path not in ("/api/live-command", "/api/live-control", "/api/live-camera"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length))
                if path == "/api/live-camera":
                    preview_state.update_camera(payload)
                else:
                    preview_state.update_command(payload)
            except (ValueError, TypeError, json.JSONDecodeError):
                self.send_error(400, "invalid command")
                return
            if path == "/api/live-control":
                if payload.get("action") == "pause":
                    preview_state.toggle_pause()
                elif payload.get("action") == "reset":
                    preview_state.request_reset()
            self.send_json(preview_state.snapshot())

        def send_json(self, payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", port), PreviewHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[INFO] Live browser preview: http://127.0.0.1:{port}")
    return server


def _force_velocity_command(env, vx: float | None, yaw: float | None) -> None:
    command = env.unwrapped.command_manager.get_command("base_velocity")
    if vx is not None:
        command[:, 0] = vx
    command[:, 1] = 0.0
    if yaw is not None:
        command[:, 2] = yaw


def _print_state(env, step_count: int) -> None:
    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("base_velocity")[0]
    root_pos_w = robot.data.root_pos_w[0]
    root_lin_vel_b = robot.data.root_lin_vel_b[0]
    root_ang_vel_b = robot.data.root_ang_vel_b[0]
    projected_gravity_b = robot.data.projected_gravity_b[0]
    tilt = torch.acos(torch.clamp(-projected_gravity_b[2], -1.0, 1.0))
    print(
        f"[step {step_count:06d}] "
        f"cmd=({command[0].item():+.2f}, {command[1].item():+.2f}, {command[2].item():+.2f}) "
        f"pos_z={root_pos_w[2].item():.3f} "
        f"vel_b=({root_lin_vel_b[0].item():+.2f}, {root_lin_vel_b[1].item():+.2f}, {root_lin_vel_b[2].item():+.2f}) "
        f"yaw_rate={root_ang_vel_b[2].item():+.2f} "
        f"tilt={tilt.item():.3f}"
    )


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
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        resume_path = args_cli.checkpoint
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    preview_state = LivePreviewState()
    preview_server = None if args_cli.no_web else start_preview_server(preview_state, args_cli.web_port)
    frame_streamer = None if args_cli.no_web else ViewportFrameStreamer(
        preview_state,
        args_cli.web_frame_fps,
        args_cli.web_frame_quality,
        args_cli.web_frame_width,
        args_cli.web_frame_height,
    )
    camera_controller = None if args_cli.no_web else LiveCameraController(env, preview_state)

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    step_count = 0
    next_print_time = 0.0

    while simulation_app.is_running():
        start_time = time.time()
        if preview_state.consume_reset():
            reset_result = env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            step_count = 0
        if preview_state.snapshot().get("paused", False):
            time.sleep(0.03)
            continue
        with torch.inference_mode():
            if args_cli.command_vx is not None:
                preview_state.command_vx = args_cli.command_vx
            if args_cli.command_yaw is not None:
                preview_state.command_yaw = args_cli.command_yaw
            command_vx, command_yaw, command_height = preview_state.command()
            _force_velocity_command(env, command_vx, command_yaw)
            env.unwrapped.command_manager.get_command("base_height")[:, 0] = command_height
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy, "reset"):
                policy.reset(dones)

        robot = env.unwrapped.scene["robot"]
        if camera_controller is not None:
            camera_controller.update()
        if frame_streamer is not None:
            frame_streamer.maybe_capture()
        projected_gravity_b = robot.data.projected_gravity_b[0]
        tilt = torch.acos(torch.clamp(-projected_gravity_b[2], -1.0, 1.0))
        preview_state.update_state(
            {
                "running": True,
                "step": step_count,
                "position": robot.data.root_pos_w[0].detach().cpu().tolist(),
                "linear_velocity": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
                "angular_velocity": robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
                "projected_gravity": projected_gravity_b.detach().cpu().tolist(),
                "height": float(robot.data.root_pos_w[0, 2].item()),
                "tilt": float(tilt.item()),
                "joint_positions": robot.data.joint_pos[0].detach().cpu().tolist(),
                "joint_velocities": robot.data.joint_vel[0].detach().cpu().tolist(),
            }
        )

        if args_cli.print_state and time.time() >= next_print_time:
            _print_state(env, step_count)
            next_print_time = time.time() + args_cli.print_interval_s

        step_count += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0.0:
            time.sleep(sleep_time)

    if preview_server is not None:
        preview_server.shutdown()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
