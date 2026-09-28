"""CPU regressions for transfer identity and obstacle success/curriculum decisions."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("transfer", ROOT / "scripts/rsl_rl/transfer_policy.py")
transfer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer)
path = ROOT / "source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/mdp/step_course.py"
tree = ast.parse(path.read_text())
tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))
             and not (isinstance(n, ast.ClassDef) and n.name == "StepCourseCommandCfg")]
scope = {"torch": torch, "PositiveBiasedVelocityCommand": object}
exec(compile(tree, str(path), "exec"), scope)


class CourseTest(unittest.TestCase):
    def test_official_curriculum_skips_initial_reset(self):
        cfg_path = ROOT / "source/my_robot_lab/my_robot_lab/tasks/manager_based/locomotion/velocity/config/my_robot/rough_scan_env_cfg.py"
        tree = ast.parse(cfg_path.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "initialized_terrain_levels"]
        calls = []
        def official(env, ids):
            calls.append(ids.tolist())
            return torch.tensor(1.)
        ns = {"terrain_levels_vel": official}
        exec(compile(tree, str(cfg_path), "exec"), ns)
        command = torch.tensor([[.5, 0., 0.], [-.5, 0., 0.]])
        env = NS(episode_length_buf=torch.zeros(2), scene=NS(terrain=NS(terrain_levels=torch.zeros(2))),
                 command_manager=NS(get_command=lambda name: command))
        ns["initialized_terrain_levels"](env, torch.tensor([0, 1]))
        self.assertEqual(calls, [])
        env.episode_length_buf[1] = 10
        ns["initialized_terrain_levels"](env, torch.tensor([0, 1]))
        self.assertEqual(calls, [[1]])
        env.episode_length_buf[0] = 10
        ns["initialized_terrain_levels"](env, torch.tensor([0, 1]))
        self.assertEqual(calls, [[1], [0, 1]])
        command[0, 2] = 2.  # A circle must not be treated as insufficient translation.
        command[1] = 0.  # Standing also leaves the level unchanged.
        ns["initialized_terrain_levels"](env, torch.tensor([0, 1]))
        self.assertEqual(calls, [[1], [0, 1]])
        command[1, 2] = 2.  # Pure turn.
        ns["initialized_terrain_levels"](env, torch.tensor([0, 1]))
        self.assertEqual(calls, [[1], [0, 1]])

    def test_transfer_preserves_full_network_output_with_arbitrary_scan(self):
        old = torch.nn.Sequential(torch.nn.Linear(49, 512), torch.nn.ELU(),
                                  torch.nn.Linear(512, 256), torch.nn.ELU(),
                                  torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 6))
        new = torch.nn.Sequential(torch.nn.Linear(94, 512), torch.nn.ELU(),
                                  torch.nn.Linear(512, 256), torch.nn.ELU(),
                                  torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 6))
        src = {"mlp." + k: v for k, v in old.state_dict().items()}
        dst = {"mlp." + k: v for k, v in new.state_dict().items()}
        expanded = transfer.expand_state(src, dst)
        new.load_state_dict({k[4:]: v for k, v in expanded.items()})
        x = torch.randn(32, 49)
        torch.testing.assert_close(old(x), new(torch.cat((x, torch.randn(32, 45)), -1)))
        self.assertEqual(torch.count_nonzero(new[0].weight[:, 49:]).item(), 0)

    def test_success_requires_crossing_both_wheels_and_settled_support(self):
        term = scope["StepCourseCommand"].__new__(scope["StepCourseCommand"])
        term.device = "cpu"
        term.wheel_ids = term.contact_ids = [0, 1]
        term.hold = torch.zeros(3)
        term.success = torch.zeros(3, dtype=torch.bool)
        term.travel_sign = torch.tensor([1., -1., 1.])
        term._last_check = -1
        wheels = torch.tensor([[[1.2, .2, .08], [1.2, -.2, .08]],
                               [[-1.2, .2, .03], [-1.2, -.2, .03]],
                               [[1.2, .2, .08], [.9, -.2, .08]]])
        term.robot = NS(data=NS(body_pos_w=wheels,
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(3, 1), root_lin_vel_b=torch.zeros(3, 3)))
        class Scene(dict):
            env_origins = torch.zeros(3, 3)
            terrain = NS(terrain_types=torch.tensor([6, 13, 6]), terrain_levels=torch.zeros(3, dtype=torch.long))
        scene = Scene(contact_forces=NS(data=NS(net_forces_w=torch.ones(3, 2, 3)*3)))
        term._env = NS(scene=scene, common_step_counter=0, step_dt=.1)
        for step in range(4):
            term._env.common_step_counter = step
            self.assertFalse(term.check_success().any())
            term.check_success()  # Multiple reward/termination reads must not double-count time.
        term._env.common_step_counter = 4
        self.assertEqual(term.check_success().tolist(), [True, True, False])

    def test_curriculum_only_promotes_after_three_completed_successes(self):
        terrain = NS(terrain_types=torch.tensor([0, 6, 13]), terrain_levels=torch.tensor([0, 0, 7]),
                     terrain_origins=torch.zeros(8, 20, 3), env_origins=torch.zeros(3, 3))
        term = NS(success=torch.tensor([True, True, False]), success_streak=torch.zeros(3), failure_streak=torch.zeros(3))
        env = NS(device="cpu", scene=NS(terrain=terrain), episode_length_buf=torch.ones(3),
                 command_manager=NS(get_term=lambda name: term),
                 termination_manager=NS(get_term=lambda name: torch.zeros(3, dtype=torch.bool)))
        for _ in range(2):
            scope["step_levels"](env, [0, 1, 2])
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 0, 6])
        scope["step_levels"](env, [0, 1, 2])
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 1, 6])


if __name__ == "__main__":
    unittest.main()
