"""Restore mass-experiment settings for training/play without importing Isaac Sim."""

import os
from pathlib import Path

import yaml


def restore_checkpoint_mass_settings(env_cfg, checkpoint, environ=None):
    """Restore known UZ05 asset/COM settings unless explicitly overridden.

    Other PLAY settings, including command ranges and disabled pushes, are owned
    by the selected task. BaseLoader reads saved Python-tagged YAML as data only.
    """
    environ = os.environ if environ is None else environ
    snapshot = Path(checkpoint).expanduser().resolve().parent / "params/env.yaml"
    if not snapshot.is_file():
        return [f"[INFO] No checkpoint env.yaml at {snapshot}; using current mass/COM settings."]
    saved = yaml.load(snapshot.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    messages = []
    spawn = env_cfg.scene.robot.spawn
    saved_spawn = saved.get("scene", {}).get("robot", {}).get("spawn", {})
    source = saved_spawn.get("asset_path", "")
    known = {"uz05_isaac.xml": "uz05", "uz05_isaac_equal_leg_masses.xml": "uz05_equal_leg_masses"}
    filename = Path(source).name
    if filename in known and "UZ05_MASS_VARIANT" not in environ:
        path = Path(source).expanduser()
        if not path.is_file():
            path = Path(spawn.asset_path).parent / filename
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint mass model is missing: {source}")
        spawn.asset_path = str(path.resolve())
        spawn.usd_dir = str(Path(spawn.usd_dir).parent / known[filename])
        messages.append(f"[INFO] Checkpoint mass model restored from checkpoint: {spawn.asset_path}")
    else:
        messages.append(f"[INFO] Checkpoint mass model from current/explicit settings: {spawn.asset_path}")

    events = saved.get("events", {})
    if "UZ05_DISABLE_BASE_COM" in environ:
        messages.append("[INFO] Checkpoint base COM randomization uses explicit UZ05_DISABLE_BASE_COM override.")
    elif "base_com" in events:
        event = events["base_com"]
        if event in (None, "null", "None", "~"):
            env_cfg.events.base_com = None
            messages.append("[INFO] Checkpoint base COM randomization disabled, matching checkpoint training.")
        elif isinstance(event, dict):
            if env_cfg.events.base_com is None:
                raise ValueError("Checkpoint used base COM randomization but PLAY has no event; "
                                 "set UZ05_DISABLE_BASE_COM=1 to explicitly test without it.")
            ranges = event.get("params", {}).get("com_range")
            if ranges:
                env_cfg.events.base_com.params["com_range"] = {
                    key: tuple(float(x) for x in values) for key, values in ranges.items()
                }
            messages.append("[INFO] Checkpoint base COM randomization enabled, matching checkpoint training.")
    return messages
