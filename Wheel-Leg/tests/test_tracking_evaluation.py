"""Regression tests for pooled metrics and fixed-command evaluation summaries."""

import sys
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from weighted_metrics import install_weighted_metrics, pool_episode_metrics
from tracking_evaluation import run_tracking_evaluation, summarize


class WeightedMetricsTest(unittest.TestCase):
    def record(self, count, total):
        prefix = "Metrics/base_height/"
        return {prefix + "height_low_target_cm": total / max(count, 1),
                prefix + "__sum__height_low_target_cm": total,
                prefix + "__count__height_low_target_cm": count,
                prefix + "height_low_sample_count": count,
                "Episode_Reward/test": 2.0}

    def test_empty_batches_do_not_dilute_height_and_unequal_counts_are_weighted(self):
        records = [self.record(0, 0), self.record(1, 25), self.record(9, 243)]
        result = pool_episode_metrics(records)
        key = "Metrics/base_height/height_low_target_cm"
        self.assertAlmostEqual(result[0][key], 26.8)
        self.assertEqual(result[0]["Metrics/base_height/height_low_sample_count"], 10)
        self.assertNotIn(key, result[1])
        self.assertNotIn(key, result[2])
        self.assertEqual(records[0][key], 0)  # Do not mutate manager extras.
        self.assertEqual(result[1]["Episode_Reward/test"], 2)
        self.assertFalse(any("__sum__" in k or "__count__" in k for r in result for k in r))

    def test_absent_mode_is_not_reported_as_zero_error(self):
        result = pool_episode_metrics([self.record(0, 0)])
        self.assertNotIn("Metrics/base_height/height_low_target_cm", result[0])
        self.assertEqual(result[0]["Metrics/base_height/height_low_sample_count"], 0)

    def test_logger_hook_keeps_original_logging_and_clear(self):
        logger = SimpleNamespace(ep_extras=[self.record(1, 25), self.record(3, 81)])
        seen = []
        def log(iteration):
            seen.append((iteration, logger.ep_extras.copy()))
            logger.ep_extras.clear()
        logger.log = log
        install_weighted_metrics(logger)
        logger.log(42)
        self.assertEqual(seen[0][0], 42)
        self.assertEqual(seen[0][1][0]["Metrics/base_height/height_low_target_cm"], 26.5)
        self.assertEqual(logger.ep_extras, [])

    def test_empty_evaluation_is_unavailable_and_signed_error_cannot_cancel_mae(self):
        self.assertEqual(summarize([]), {"samples": 0})
        rows = [dict(vx_target_mps=0., vx_actual_mps=v, vy_actual_mps=0.,
                     height_target_m=.30, height_actual_m=h)
                for v, h in [(-.1, .28), (.1, .32)]]
        result = summarize(rows)
        self.assertAlmostEqual(result["vx_actual_mps"], 0)
        self.assertAlmostEqual(result["vx_mae_mps"], .1)
        self.assertAlmostEqual(result["height_bias_cm"], 0)
        self.assertAlmostEqual(result["height_mae_cm"], 2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch is required for the mock environment")
class EvaluationLoopTest(unittest.TestCase):
    def run_mock(self, fail):
        import torch
        velocity = SimpleNamespace(command=torch.zeros(1, 3), is_standing_env=torch.ones(1, dtype=torch.bool),
                                   time_left=torch.zeros(1))
        height = SimpleNamespace(command=torch.zeros(1, 1), time_left=torch.zeros(1),
                                 cfg=SimpleNamespace(body_name="chassis"))
        data = SimpleNamespace(root_lin_vel_b=torch.zeros(1, 3), body_pos_w=torch.zeros(1, 1, 3))
        robot = SimpleNamespace(data=data, find_bodies=lambda name: ([0], [name]))
        raw = SimpleNamespace(scene={"robot": robot}, step_dt=.1, num_envs=1, device="cpu",
                              cfg=SimpleNamespace(seed=42),
                              command_manager=SimpleNamespace(get_term=lambda name: height if name == "base_height" else velocity))
        resets = []
        def step(action):
            self.assertFalse(bool(velocity.is_standing_env[0]))
            data.root_lin_vel_b.copy_(velocity.command)
            data.body_pos_w[0, 0, 2] = 999 if fail else height.command[0, 0]
            return None, None, torch.tensor([fail]), {"time_outs": torch.tensor([False])}
        env = SimpleNamespace(unwrapped=raw, reset=lambda: resets.append(1),
                              get_observations=lambda: None, step=step)
        def force(env, vx, yaw):
            velocity.command[0] = torch.tensor([vx, 0., yaw])
            return velocity.command[0].clone()
        args = SimpleNamespace(eval_settle_s=.1, eval_measure_s=.2, eval_trials=2)
        with tempfile.TemporaryDirectory() as tmp:
            run_tracking_evaluation(env, lambda obs: None, SimpleNamespace(is_running=lambda: True),
                                    force, args, "fake.pt", tmp)
            report = json.loads((Path(tmp) / "report.json").read_text())
        self.assertEqual(len(resets), 10)
        return report

    def test_perfect_response_and_independent_trials(self):
        report = self.run_mock(False)
        self.assertAlmostEqual(report["height_response_gain"], 1, places=5)
        self.assertAlmostEqual(report["velocity_response_gain"], 1, places=5)
        self.assertTrue(all(case["complete_trials"] == 2 for case in report["cases"]))
        self.assertTrue(all(case["samples"] == 4 for case in report["cases"]))

    def test_failure_does_not_score_auto_reset_state(self):
        report = self.run_mock(True)
        self.assertNotIn("height_response_gain", report)
        self.assertNotIn("velocity_response_gain", report)
        self.assertTrue(all(case["failed_trials"] == 2 for case in report["cases"]))
        self.assertTrue(all(case["samples"] == 0 for case in report["cases"]))


if __name__ == "__main__":
    unittest.main()
