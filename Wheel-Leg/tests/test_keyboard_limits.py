"""Keyboard settings must respect the selected task's training speed range."""
import importlib.util
from pathlib import Path
import sys

path = Path(__file__).parents[1] / 'scripts/rsl_rl/command_manager.py'
spec = importlib.util.spec_from_file_location('play_command_manager_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_task_limits_hold_after_repeated_arrow_inputs():
    manager = module.CommandManager(velocity_max=.5, yaw_max=1., smooth=False)
    manager.update_settings(v_delta=10., w_delta=10.)
    state = manager.advance(.02, {'W', 'A'})
    assert state.vx_cmd == .5
    assert state.yaw_rate_cmd == 1.
    state = manager.advance(.02, {'S', 'D'})
    assert state.vx_cmd == -.5
    assert state.yaw_rate_cmd == -1.
    state = manager.advance(.02, {'SPACE', 'W', 'A'})
    assert state.vx_cmd == state.yaw_rate_cmd == 0.
