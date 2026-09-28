"""CPU regressions for progressive roughness and terrain-aware balance rewards."""
import ast
import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

MDP = Path(__file__).parents[1] / 'source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp'


def test_rough_amplitude_increases_without_mutating_config(monkeypatch):
    spec = importlib.util.spec_from_file_location('gentle_terrain', MDP / 'gentle_terrain.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    upstream = ModuleType('isaaclab.terrains.height_field.hf_terrains')
    upstream.random_uniform_terrain = lambda difficulty, cfg: cfg.noise_range
    monkeypatch.setitem(sys.modules, upstream.__name__, upstream)
    cfg = NS(amplitude_range=(.003, .020), noise_range=(-.02, .02))
    cfg.copy = lambda: copy.copy(cfg)
    for difficulty, amplitude in [(0., .003), (.5, .0115), (1., .020)]:
        assert module.progressive_random_rough(difficulty, cfg) == pytest.approx((-amplitude, amplitude))
        assert cfg.noise_range == (-.02, .02)


def test_gentle_orientation_allows_pitch_but_still_penalizes_roll():
    tree = ast.parse((MDP / 'rough_posture.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('pitch_deadband', 'gentle_orientation')]
    pitch = next(n for n in tree.body if n.name == 'pitch_deadband')
    pitch.body = [n for n in pitch.body if not isinstance(n, ast.ImportFrom)]
    scan = torch.zeros(1, 45)
    scope = {'torch': torch, 'terrain_scan': lambda env: scan}
    exec(compile(tree, 'rough_posture', 'exec'), scope)
    angle = torch.tensor(10. * torch.pi / 180.)
    data = NS(projected_gravity_b=torch.tensor([[angle.sin(), 0., -angle.cos()]]))
    env = NS(scene={'robot': NS(data=data)})
    reward = scope['gentle_orientation']
    flat_pitch_reward = reward(env).item()
    scan[0, 0] = .1
    assert reward(env).item() == pytest.approx(1.)
    assert flat_pitch_reward < reward(env).item()
    data.projected_gravity_b[:] = torch.tensor([[0., angle.sin(), -angle.cos()]])
    assert reward(env).item() < 1.
    scan[:] = 0.
    data.projected_gravity_b[:] = torch.tensor([[0., 0., -1.]])
    assert reward(env).item() == pytest.approx(1.)


def test_failure_is_single_event_and_does_not_penalize_timeout_only():
    tree = ast.parse((MDP / 'rough_posture.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'failure_event']
    scope = {}
    exec(compile(tree, 'rough_posture', 'exec'), scope)
    for dt in (.01, .02, .04):
        env = NS(step_dt=dt, termination_manager=NS(
            terminated=torch.tensor([False, True, False, True]),
            time_outs=torch.tensor([False, False, True, True])))
        torch.testing.assert_close(scope['failure_event'](env) * dt, torch.tensor([0., 1., 0., 1.]))
