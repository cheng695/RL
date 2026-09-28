"""Two square-ring steps, retaining the terrain generator's center spawn convention."""
import numpy as np
import trimesh


def _two_steps(difficulty, cfg, ascending):
    sx, sy = cfg.size
    inner = cfg.platform_width / 2
    outer = inner + cfg.step_width
    if inner <= 0 or cfg.step_width <= 0 or outer >= min(sx, sy) / 2:
        raise ValueError("Two-step terrain requires a central platform and room for both treads.")
    # Fixed risers, independent of the generator row/difficulty.
    first, second = cfg.first_step_height, cfg.second_step_height
    if first <= 0 or second <= 0:
        raise ValueError("Both step risers must be positive.")
    total = first + second
    meshes = []

    def box(x0, x1, y0, y1, z):
        mesh = trimesh.creation.box(extents=(x1-x0, y1-y0, z+.1))
        mesh.apply_translation(((x0+x1)/2, (y0+y1)/2, (z-.1)/2))
        meshes.append(mesh)

    # Non-overlapping rectangular regions: center, intermediate ring, outer landing.
    box(-inner, inner, -inner, inner, 0. if ascending else total)
    for a, bx, by, z in ((inner, outer, outer, first if ascending else second),
                         (outer, sx/2, sy/2, total if ascending else 0.)):
        box(-bx, -a, -by, by, z)
        box(a, bx, -by, by, z)
        box(-a, a, -by, -a, z)
        box(-a, a, a, by, z)
    for mesh in meshes:
        mesh.apply_translation((sx/2, sy/2, 0.))
    return meshes, np.array([sx/2, sy/2, 0. if ascending else total])


def two_step_up(difficulty, cfg):
    return _two_steps(difficulty, cfg, True)


def two_step_down(difficulty, cfg):
    return _two_steps(difficulty, cfg, False)
