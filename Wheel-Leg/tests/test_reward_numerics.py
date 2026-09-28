"""CPU tensor regression checks, without launching Isaac Sim.

Run: python -m unittest discover -s Wheel-Leg/tests -p 'test_reward_numerics.py'
Only Isaac Lab imports are replaced; reward function bodies execute unchanged.
These checks do not validate physics, asset import, or training convergence.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
)
tree = ast.parse(SOURCE.read_text())
tree.body = [
    node for node in tree.body
    if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab"))
]
namespace = {"SceneEntityCfg": lambda name: SimpleNamespace(name=name)}
exec(compile(tree, str(SOURCE), "exec"), namespace)
rewards = SimpleNamespace(**{name: value for name, value in namespace.items() if callable(value)})


def make_env(count=4):
    data = SimpleNamespace(
        root_lin_vel_b=torch.zeros(count, 3),
        joint_vel=torch.zeros(count, 2),
        applied_torque=torch.zeros(count, 2),
    )
    command = torch.zeros(count, 3)
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: command),
        action_manager=SimpleNamespace(action=torch.zeros(count, 6), prev_action=torch.zeros(count, 6)),
        episode_length_buf=torch.full((count,), 5),
        common_step_counter=5,
    )


class RewardNumericsTest(unittest.TestCase):
    def test_height_reward_uses_local_ground(self):
        env = make_env(count=2)
        env.scene["robot"].data.body_pos_w = torch.tensor([[[0., 0., .3]], [[0., 0., .5]]])
        env.command_manager.get_command("base_height")[:, 0] = .3
        env.command_manager.get_term = lambda name: SimpleNamespace(
            reference_ground_height=lambda: torch.tensor([0., .2]))
        cfg = SimpleNamespace(name="robot", body_ids=[0])
        actual = rewards.base_height_command_exp_kernel(env, "base_height", 200., cfg, True)
        torch.testing.assert_close(actual, torch.ones(2))
        actual = rewards.base_height_command_l2(env, "base_height", .05, 9., cfg, True)
        torch.testing.assert_close(actual, torch.zeros(2))

    def test_straight_pitch_transition_gate_and_symmetry(self):
        env = make_env(count=6)
        pitch = torch.tensor([0.15, -0.15, 0.02, 0.15, 0.15, 0.15])
        env.scene["robot"].data.projected_gravity_b = torch.stack(
            (pitch.sin(), torch.zeros_like(pitch), -pitch.cos()), -1)
        command = torch.zeros(6, 3)
        command[:, 0] = torch.tensor([0.3, -0.3, 0.3, 0.3, 0.3, 0.3])
        command[5, 2] = 1.
        velocity = SimpleNamespace(command=command, command_age=torch.tensor([2., 2., 2., 0., 2., 2.]))
        height = SimpleNamespace(command_age=torch.tensor([2., 2., 2., 2., 0., 2.]))
        env.command_manager.get_term = lambda name: velocity if name == "base_velocity" else height
        result = rewards.straight_pitch_deadband_l2(env, "base_velocity", "base_height", 1., 0.05, 0.1, 4.)
        torch.testing.assert_close(result, torch.tensor([1., 1., 0., 0., 0., 0.]))

    def test_huber_invalid_states_get_maximum_penalty(self):
        env = make_env()
        env.scene["robot"].data.root_lin_vel_b[:, 0] = torch.tensor([0.0, float("nan"), float("inf"), -float("inf")])
        actual = rewards.vx_tracking_huber(env, "base_velocity", beta=0.3, max_value=2.0)
        torch.testing.assert_close(actual, torch.tensor([0.0, 2.0, 2.0, 2.0]))

    def test_huber_small_large_and_capped_errors(self):
        env = make_env()
        env.scene["robot"].data.root_lin_vel_b[:, 0] = torch.tensor([0.0, 0.15, -1.0, 10.0])
        actual = rewards.vx_tracking_huber(env, "base_velocity", beta=0.3, max_value=2.0)
        torch.testing.assert_close(actual, torch.tensor([0.0, 0.0375, 0.85, 2.0]))

    def test_partial_reset_and_shared_second_order_cache(self):
        env = make_env(count=2)
        env.action_manager.prev_action.fill_(2.0)
        env.action_manager.action.fill_(3.0)
        rewards.action_second_order_l2(env, 1000.0, (0, 4))
        env.common_step_counter += 1
        env.episode_length_buf[:] = torch.tensor([1, 6])
        env.action_manager.prev_action[0] = 0.0
        env.action_manager.prev_action[1] = 3.0
        env.action_manager.action[0] = 1.0
        env.action_manager.action[1] = 4.0
        # Reset env: 1 - 0 + 0; continuing env: 4 - 2*3 + 2 = 0.
        legs = rewards.action_second_order_l2(env, 1000.0, (0, 4))
        wheels = rewards.action_second_order_l2(env, 1000.0, (4, 6))
        torch.testing.assert_close(legs, torch.tensor([4.0, 0.0]))
        torch.testing.assert_close(wheels, torch.tensor([2.0, 0.0]))
        torch.testing.assert_close(rewards.action_second_order_l2(env, 1000.0, (0, 4)), legs)

    def test_invalid_actions_are_penalized(self):
        env = make_env()
        env.action_manager.action[:, 4] = torch.tensor([0.0, float("nan"), float("inf"), -float("inf")])
        expected = torch.tensor([0.0, 1000.0, 1000.0, 1000.0])
        for fn in (rewards.action_rate_l2, rewards.action_second_order_l2):
            torch.testing.assert_close(fn(env, 1000.0, (4, 6)), expected)

    def test_positive_power_and_invalid_power(self):
        env = make_env(count=5)
        data = env.scene["robot"].data
        data.joint_vel.fill_(1.0)
        data.applied_torque[:, 0] = torch.tensor([2.0, -2.0, float("nan"), float("inf"), -float("inf")])
        cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
        actual = rewards.wheel_power_l1_positive(env, max_value=100.0, asset_cfg=cfg)
        torch.testing.assert_close(actual, torch.tensor([2.0, 0.0, 100.0, 100.0, 100.0]))

    def test_standstill_gate_still_allows_translation(self):
        env = make_env(count=2)
        env.scene["robot"].data.root_lin_vel_b[:] = torch.tensor([[0.3, 0.4, 0.0], [0.3, 0.4, 0.0]])
        env.command_manager.get_command("base_velocity")[1, 0] = 0.5
        actual = rewards.stand_still_lin_vel_l2(env, "base_velocity", 0.08, 4.0)
        torch.testing.assert_close(actual, torch.tensor([0.25, 0.0]))

    def test_scaled_standing_drift_gate_and_invalid_values(self):
        env = make_env(count=6)
        velocity = env.scene["robot"].data.root_lin_vel_b
        velocity[:, 0] = torch.tensor([0.03, 0.03, 0.03, float("nan"), 100., 0.])
        velocity[5, 2] = 1.0  # Vertical motion must not incur this cost.
        command = env.command_manager.get_command("base_velocity")
        command[1, 2] = 0.5  # Rotation is not standing.
        command[2, 0] = -0.3  # Reverse motion is not standing.
        actual = rewards.stand_still_drift_huber(env, "base_velocity", 1e-6, 0.05, 10.)
        torch.testing.assert_close(actual, torch.tensor([0.18, 0., 0., 10., 10., 0.]))
        with self.assertRaises(ValueError):
            rewards.stand_still_drift_huber(env, "base_velocity", 1e-6, 0., 10.)


if __name__ == "__main__":
    unittest.main()
