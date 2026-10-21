"""Exercise the real keyboard callback without launching Isaac Sim."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest


@pytest.fixture
def controller():
    path = Path(__file__).parents[1] / "scripts/rsl_rl/keyboard_controller.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scope = {"CommandManager": object,
             "carb": NS(input=NS(KeyboardEventType=NS(KEY_PRESS=1, KEY_RELEASE=2)))}
    exec(compile(tree, str(path), "exec"), scope)
    instance = scope["KeyboardController"].__new__(scope["KeyboardController"])
    instance.pressed = set()
    instance.commands = Mock(velocity_step=.1, yaw_step=.2, height_step=.01)
    instance.commands.state = NS(reset_requested=False)
    return instance


@pytest.mark.parametrize("as_enum", [False, True])
def test_press_release_reset_and_setting_edges(controller, as_enum):
    def send(key, event_type=1):
        value = NS(name=key) if as_enum else key.lower()
        assert controller._on_keyboard_event(NS(input=value, type=event_type)) is True

    for key in ("W", "A", "S", "D", "SPACE"):
        send(key)
        assert key in controller.pressed
        send(key, 2)
        assert key not in controller.pressed
    send("R")
    assert controller.commands.state.reset_requested
    send("UP")
    send("UP")
    controller.commands.update_settings.assert_called_once_with(v_delta=.1)
    send("UP", 2)
    send("UP")
    assert controller.commands.update_settings.call_count == 2
    send("I")
    controller.commands.update_height.assert_called_once_with(.01)
    send("H")
    controller.commands.reset_height.assert_called_once_with()


def test_unrecognized_input_is_ignored(controller):
    assert controller._on_keyboard_event(NS(input=None, type=1)) is True
    assert controller.pressed == set()
    assert not controller.commands.state.reset_requested
