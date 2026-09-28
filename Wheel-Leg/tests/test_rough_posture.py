"""Geometry regression: stair height and heading must not look like leg splitting."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import torch

path = Path(__file__).parents[1] / 'source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/rough_posture.py'
tree = ast.parse(path.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'wheel_fore_aft_split']
scope = {'torch': torch, 'SceneEntityCfg': object}
exec(compile(tree, str(path), 'exec'), scope)


def test_split_ignores_vertical_step_and_rotates_with_heading():
    cfg = NS(name='robot', body_ids=[0, 1])
    data = NS(body_pos_w=torch.tensor([[[0., .2, .2], [0., -.2, 0.]]]),
              root_quat_w=torch.tensor([[1., 0., 0., 0.]]))
    env = NS(scene={'robot': NS(data=data)})
    reward = scope['wheel_fore_aft_split']
    assert reward(env, cfg).item() == 0
    data.body_pos_w[0, 0, 0] = .3
    original = reward(env, cfg)
    assert original.item() > 0
    points = data.body_pos_w.clone()
    data.body_pos_w[..., 0] = -points[..., 1]
    data.body_pos_w[..., 1] = points[..., 0]
    data.root_quat_w[:] = torch.tensor([[math.sqrt(.5), 0, 0, math.sqrt(.5)]])
    torch.testing.assert_close(reward(env, cfg), original)


def test_gentle_split_has_no_large_split_plateau():
    cfg = NS(name='robot', body_ids=[0, 1])
    # Includes reversed leg ordering; only fore/aft distance should matter.
    separations = torch.tensor([0., .03, .04, .10, .30, .60, -.30])
    wheels = torch.zeros(len(separations), 2, 3)
    wheels[:, 0, 0] = separations
    data = NS(body_pos_w=wheels, root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(len(separations), 1))
    cost = scope['wheel_fore_aft_split'](
        NS(scene={'robot': NS(data=data)}), cfg,
        deadband=.04, scale=.10, linear_tail=True, max_value=None)
    torch.testing.assert_close(cost[:3], torch.zeros(3))
    assert cost[3] > 0
    assert cost[5] > cost[4] > cost[3]
    torch.testing.assert_close(cost[4], cost[6])
