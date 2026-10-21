"""Failure checks for bounded, manually promoted terrain stages."""
import torch


def invalid_ground_scan(env):
    hits = env.scene["height_scanner"].data.ray_hits_w
    return ~torch.isfinite(hits).all(dim=-1).all(dim=-1)


def terrain_tile_exit(env, margin: float = 0.5):
    # Keep episodes inside their assigned terrain; leaving is not success.
    half = torch.tensor(env.cfg.scene.terrain.terrain_generator.size, device=env.device) / 2 - margin
    return ((env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs() > half).any(-1)
