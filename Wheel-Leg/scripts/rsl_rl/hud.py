"""Small Isaac Sim UI overlay for interactive playback."""

from __future__ import annotations


class RobotHUD:
    def __init__(self) -> None:
        self.window = None
        self.label = None
        try:
            import omni.ui as ui

            self.window = ui.Window("UZ-05 Play", width=360, height=330)
            with self.window.frame:
                self.label = ui.Label("Starting play...", word_wrap=True)
        except Exception as exc:
            print(f"[WARN] HUD unavailable: {exc}")

    def update(self, text: str) -> None:
        if self.label is not None:
            self.label.text = text

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
