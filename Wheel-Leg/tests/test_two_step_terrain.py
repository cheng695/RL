import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

path = Path(__file__).parents[1] / 'source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/two_step_terrain.py'
spec = importlib.util.spec_from_file_location('two_step_terrain', path)
terrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(terrain)


@pytest.mark.parametrize('difficulty', [0., .5, 1.])
@pytest.mark.parametrize('ascending', [True, False])
def test_exactly_two_steps(difficulty, ascending):
    cfg = SimpleNamespace(size=(8., 8.), platform_width=3., step_width=.8,
                          first_step_height=.15, second_step_height=.20)
    meshes, origin = terrain._two_steps(difficulty, cfg, ascending)
    assert np.allclose(origin, [4., 4., 0. if ascending else .35])
    heights = []
    for x in [4., 5.7, 6.7, 7.9]:
        tops = [m.bounds[1, 2] for m in meshes
                if m.bounds[0, 0] < x < m.bounds[1, 0]
                and m.bounds[0, 1] < 4. < m.bounds[1, 1]]
        assert len(tops) == 1
        heights.append(tops[0])
    assert np.allclose(heights, [0., .15, .35, .35] if ascending else [.35, .20, 0., 0.])
    assert all(m.is_watertight for m in meshes)
