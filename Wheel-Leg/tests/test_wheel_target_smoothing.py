"""CPU checks of the real velocity limiter and reward, without starting Isaac Sim."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_reward_numerics import rewards


class JointVelocityStub:
    def __init__(self, cfg, env):
        self._offset = torch.tensor([[0.25, -0.25], [0.25, -0.25]])
        self._processed_actions = torch.zeros(2, 2)

    @property
    def processed_actions(self):
        return self._processed_actions

    def process_actions(self, actions):
        self._processed_actions.copy_(actions * 50.0 + self._offset)

    def reset(self, env_ids=None):
        self._processed_actions[slice(None) if env_ids is None else env_ids] = 0.0


path = (Path(__file__).resolve().parents[1]
        / "source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/actions.py")
tree = ast.parse(path.read_text())
tree.body = [n for n in tree.body if
             (isinstance(n, ast.ImportFrom) and n.module == "__future__")
             or (isinstance(n, ast.ClassDef) and n.name == "RateLimitedJointVelocityAction")]
namespace = {"torch": torch, "JointVelocityAction": JointVelocityStub}
exec(compile(tree, str(path), "exec"), namespace)


class WheelTargetSmoothingTest(unittest.TestCase):
    def setUp(self):
        self.term = namespace["RateLimitedJointVelocityAction"](
            SimpleNamespace(max_velocity_rate=100.0), SimpleNamespace(step_dt=0.02))
        self.env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: self.term))

    def reward(self):
        return rewards.wheel_target_delta_l2(self.env, "wheel_joint_vel", 2.0, 4.0)

    def test_final_target_change_and_repeat_reads(self):
        self.term.process_actions(torch.tensor([[1.0, -1.0], [0.02, 0.0]]))
        torch.testing.assert_close(self.term.target_delta, torch.tensor([[2., -2.], [1., 0.]]))
        torch.testing.assert_close(self.reward(), torch.tensor([1., 0.125]))
        torch.testing.assert_close(self.reward(), torch.tensor([1., 0.125]))

    def test_constant_nonzero_target_has_no_cost(self):
        action = torch.full((2, 2), 0.02)
        self.term.process_actions(action)
        self.term.process_actions(action)
        torch.testing.assert_close(self.reward(), torch.zeros(2))

    def test_partial_and_full_reset(self):
        self.term.process_actions(torch.ones(2, 2))
        self.term.reset([0])
        torch.testing.assert_close(self.term.target_delta, torch.tensor([[0., 0.], [2., 2.]]))
        self.term.process_actions(torch.full((2, 2), -1.0))
        torch.testing.assert_close(self.term.processed_actions,
                                   torch.tensor([[-1.75, -2.25], [0.25, -0.25]]))
        self.term.reset()
        torch.testing.assert_close(self.term.target_delta, torch.zeros(2, 2))
        self.term.process_actions(torch.zeros(2, 2))
        torch.testing.assert_close(self.reward(), torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
