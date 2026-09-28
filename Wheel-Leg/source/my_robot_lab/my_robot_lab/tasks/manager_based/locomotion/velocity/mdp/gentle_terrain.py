"""Difficulty-aware wrapper around Isaac Lab's random height-field generator."""


def progressive_random_rough(difficulty, cfg):
    """Interpolate sampled height amplitude without mutating the shared terrain config.

    Upstream random_uniform_terrain ignores difficulty. Its cubic interpolation can
    overshoot the sampled noise range slightly; this range is not a hard mesh bound.
    """
    from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain

    low, high = cfg.amplitude_range
    amplitude = low + max(0., min(1., float(difficulty))) * (high - low)
    local_cfg = cfg.copy()
    local_cfg.noise_range = (-amplitude, amplitude)
    return random_uniform_terrain(difficulty, local_cfg)
