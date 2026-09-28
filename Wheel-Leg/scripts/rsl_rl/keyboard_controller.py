"""Isaac Sim keyboard input for interactive policy playback."""

from __future__ import annotations

import weakref

import carb
import omni

from command_manager import CommandManager


class KeyboardController:
    """WASD hold commands, setting-key edge commands, and reset/stop handling."""

    def __init__(self, commands: CommandManager) -> None:
        self.commands = commands
        self.pressed: set[str] = set()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def close(self) -> None:
        if self._subscription is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None

    def _on_keyboard_event(self, event, *args):
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            first_press = key not in self.pressed
            self.pressed.add(key)
            if first_press:
                if key == "UP":
                    self.commands.update_settings(v_delta=self.commands.velocity_step)
                elif key == "DOWN":
                    self.commands.update_settings(v_delta=-self.commands.velocity_step)
                elif key == "RIGHT":
                    self.commands.update_settings(w_delta=self.commands.yaw_step)
                elif key == "LEFT":
                    self.commands.update_settings(w_delta=-self.commands.yaw_step)
                elif key == "I":
                    self.commands.update_height(self.commands.height_step)
                elif key == "K":
                    self.commands.update_height(-self.commands.height_step)
                elif key == "H":
                    self.commands.reset_height()
                elif key == "R":
                    self.commands.state.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key)
        return True
