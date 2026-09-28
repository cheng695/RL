"""CPU tests of custom metric accumulation with an isolated command base stub."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class CommandBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = 4, "cpu"
        self.metrics = {}
        self.command = torch.tensor([[0.5, 0., 0.], [-0.8, 0., 0.], [0., 0., 1.], [0.8, 0., 1.]])
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_b=torch.zeros(4, 3), root_ang_vel_b=torch.zeros(4, 3)))

    def _update_metrics(self):
        pass

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in self.metrics.values():
            value[ids] = 0
        return {}


source = (Path(__file__).resolve().parents[1]
          / "source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py")
tree = ast.parse(source.read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name == "PositiveBiasedVelocityCommand"]
namespace = {"torch": torch, "UniformVelocityCommand": CommandBase, "Sequence": list}
exec(compile(tree, str(source), "exec"), namespace)


class VelocityMetricsTest(unittest.TestCase):
    def test_standing_metrics_and_partial_reset(self):
        env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(4, 6)), step_dt=0.02)
        cmd = namespace["PositiveBiasedVelocityCommand"](SimpleNamespace(action_limit=1.), env)
        cmd.command[:2] = 0
        cmd.robot.data.root_lin_vel_b[:, 0] = torch.tensor([0.03, -0.01, 1., 1.])
        cmd._update_metrics()
        cmd._update_metrics()
        result = cmd.reset([0, 2])
        self.assertEqual(result["stand_sample_count"], 2.)
        self.assertAlmostEqual(result["stand_vx_bias_mps"], 0.03)
        self.assertAlmostEqual(result["stand_path_length_m"], 0.0012)
        result = cmd.reset([1, 3])
        self.assertAlmostEqual(result["stand_vx_bias_mps"], -0.01)
        self.assertAlmostEqual(result["stand_speed_xy_mps"], 0.01)
        self.assertEqual(cmd.reset()["stand_sample_count"], 0.)

    def test_balanced_forward_reverse_sampling(self):
        env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(4, 6)), step_dt=0.02)
        cfg = SimpleNamespace(action_limit=1., ranges=SimpleNamespace(lin_vel_x=(-1., 1.)),
                              min_abs_lin_vel_x=0.08, positive_lin_vel_x_prob=0.5)
        cmd = namespace["PositiveBiasedVelocityCommand"](cfg, env)
        with torch.random.fork_rng():
            torch.manual_seed(0)
            values = cmd._sample_lin_vel_x(10000)
        self.assertLess(abs((values > 0).float().mean().item() - 0.5), 0.02)
        self.assertTrue(((values.abs() >= 0.08) & (values.abs() <= 1.)).all())

    def test_twist_projection_preserves_direction(self):
        env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(4, 6)), step_dt=0.02)
        cfg = SimpleNamespace(action_limit=1., wheel_radius=.055, wheel_half_track=.210335,
                              max_wheel_speed=20.)
        cmd = namespace["PositiveBiasedVelocityCommand"](cfg, env)
        cmd.vel_command_b = torch.zeros(4, 3)
        cmd.vel_command_b[0, :3] = torch.tensor([1., 0., 5.])
        cmd._project_to_wheel_speed_limit(torch.tensor([0]))
        projected = cmd.vel_command_b[0]
        self.assertLess(projected[0].item(), 1.)
        self.assertLess(projected[2].item(), 5.)
        self.assertAlmostEqual(projected[0].item() / projected[2].item(), 1. / 5., places=5)
        left = (projected[0] - .210335 * projected[2]) / .055
        right = (projected[0] + .210335 * projected[2]) / .055
        self.assertLessEqual(torch.maximum(left.abs(), right.abs()).item(), 20.00001)

    def test_direction_speed_boundary_and_partial_reset(self):
        env = SimpleNamespace(action_manager=SimpleNamespace(action=torch.zeros(4, 6)), step_dt=0.02)
        cmd = namespace["PositiveBiasedVelocityCommand"](SimpleNamespace(action_limit=1.), env)
        cmd._update_metrics()
        result = cmd.reset([0, 2])
        self.assertAlmostEqual(result["vx_mae_straight_forward_mps"], 0.5)
        self.assertEqual(result["straight_low_speed_sample_count"], 1.)
        self.assertEqual(result["straight_reverse_sample_count"], 0.)
        result = cmd.reset([1, 3])
        self.assertAlmostEqual(result["vx_mae_straight_reverse_mps"], 0.8)
        self.assertEqual(result["straight_high_speed_sample_count"], 1.)
        self.assertEqual(result["mixed_sample_count"], 1.)
        result = cmd.reset()
        self.assertEqual(result["straight_sample_count"], 0.)


class HeightBase:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device, self.metrics = 4, "cpu", {}

    def reset(self, env_ids=None):
        self._resample_command(range(4) if env_ids is None else env_ids)
        return {}


height_tree = ast.parse(source.read_text())
height_tree.body = [node for node in height_tree.body if isinstance(node, ast.ClassDef)
                    and node.name == "UniformBaseHeightCommand"]
height_namespace = {"torch": torch, "CommandTerm": HeightBase, "Sequence": list}
exec(compile(height_tree, str(source), "exec"), height_namespace)


class HeightCommandTest(unittest.TestCase):
    def test_local_ground_height_and_missing_ray_fallback(self):
        cmd = self.make_command()
        class Scene(dict):
            env_origins = torch.tensor([[0., 0., 0.], [0., 0., 0.15],
                                        [0., 0., 0.2], [0., 0., 0.4]])
        scene = Scene(cmd._env.scene)
        hits = torch.zeros(4, 3, 3)
        hits[:, :, 2] = torch.tensor([[0., 0., 0.], [.15, .15, float("inf")],
                                      [.2, .2, .2], [float("nan"), float("inf"), float("nan")]])
        scene["ground"] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
        scene["robot"].data.body_pos_w[:, 0, 2] = torch.tensor([.31, .46, .51, .71])
        cmd._env.scene = scene
        cmd.cfg.ground_sensor_name = "ground"
        cmd.height_command.fill_(.31)
        torch.testing.assert_close(cmd.reference_ground_height(), torch.tensor([0., .15, .2, .4]))
        cmd._update_metrics()
        torch.testing.assert_close(cmd.metrics["height_actual_cm"], torch.full((4,), 31.))
        torch.testing.assert_close(cmd.metrics["height_mae_cm"], torch.zeros(4), atol=1e-5, rtol=0)

    def make_command(self):
        robot = SimpleNamespace(find_bodies=lambda name: ([0], [name]),
                                data=SimpleNamespace(body_pos_w=torch.zeros(4, 1, 3)))
        robot.data.body_pos_w[:, 0, 2] = 0.31
        env = SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.ones(4))
        cfg = SimpleNamespace(asset_name="robot", body_name="chassis",
                              height_range=(0.28, 0.34), endpoint_switch_prob=1.)
        return height_namespace["UniformBaseHeightCommand"](cfg, env)

    def test_alternating_endpoints_and_partial_resample(self):
        cmd = self.make_command()
        cmd.height_command[:, 0] = torch.tensor([0.28, 0.34, 0.30, 0.33])
        cmd._resample_command([0, 1])
        torch.testing.assert_close(cmd.command[:, 0], torch.tensor([0.34, 0.28, 0.30, 0.33]))
        cmd._resample_command([0, 1])
        torch.testing.assert_close(cmd.command[:, 0], torch.tensor([0.28, 0.34, 0.30, 0.33]))
        cmd.cfg.endpoint_switch_prob = 0.
        cmd._resample_command(range(4))
        self.assertTrue(((cmd.command >= 0.28) & (cmd.command <= 0.34)).all())

    def test_constant_height_is_visible_in_both_target_bins(self):
        cmd = self.make_command()
        cmd.height_command[:, 0] = torch.tensor([0.28, 0.34, 0.31, 0.28])
        cmd._update_metrics()
        result = cmd.reset([0, 1])
        self.assertAlmostEqual(result["height_low_actual_cm"], 31., places=4)
        self.assertAlmostEqual(result["height_high_actual_cm"], 31., places=4)
        self.assertAlmostEqual(result["height_high_mae_cm"], 3., places=4)
        self.assertEqual(result["height_mid_sample_count"], 0.)
        self.assertEqual(cmd._height_bins["low"][3, 0].item(), 1.)


if __name__ == "__main__":
    unittest.main()
