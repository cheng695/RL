"""Check that manual play commands use the training projection before observation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from test_velocity_metrics import namespace


source = Path(__file__).resolve().parents[1] / "scripts/rsl_rl/play.py"
tree = ast.parse(source.read_text())
tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "_force_velocity_command"]
scope = {}
exec(compile(tree, str(source), "exec"), scope)


class PlayWorkspaceTest(unittest.TestCase):
    def test_manual_command_projects_all_envs_and_preserves_snapshot(self):
        cfg = SimpleNamespace(wheel_radius=.055, wheel_half_track=.210335, max_wheel_speed=20.)
        term = namespace["PositiveBiasedVelocityCommand"](cfg, SimpleNamespace())
        term.vel_command_b = term.command
        env = SimpleNamespace(unwrapped=SimpleNamespace(command_manager=SimpleNamespace(
            get_term=lambda name: term)))
        for vx, yaw in ((1., 5.), (-1., -5.), (1., -5.), (-1., 5.), (0., 5.), (1., 0.), (0., 0.)):
            result = scope["_force_velocity_command"](env, vx, yaw)
            scale = min(1., 1.1 / max(abs(vx) + .210335 * abs(yaw), 1e-6))
            expected = torch.tensor([vx * scale, 0., yaw * scale])
            torch.testing.assert_close(term.command, expected.repeat(4, 1))
            term.command.zero_()  # env.step can resample after action selection.
            torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
