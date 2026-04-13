"""Interactive UI panels for the Taichi GGUI viewer.

Provides parameter sliders, playback controls, and real-time status display
using Taichi's native imgui integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class SliderConfig:
    """Configuration for a single slider widget."""

    label: str
    min_val: float
    max_val: float
    default: float
    step: float = 0.0  # 0 = auto
    fmt: str = "%.3f"


@dataclass
class PlaybackState:
    """Simulation playback controls state."""

    playing: bool = True
    speed: float = 1.0
    single_step_requested: bool = False
    reset_requested: bool = False
    sim_time: float = 0.0
    frame_count: int = 0
    real_fps: float = 0.0

    # Frame timing for FPS calculation
    _last_time: float = 0.0
    _frame_times: list[float] = field(default_factory=list)

    def update_fps(self, current_time: float):
        """Update rolling FPS estimate."""
        if self._last_time > 0:
            dt = current_time - self._last_time
            self._frame_times.append(dt)
            if len(self._frame_times) > 30:
                self._frame_times.pop(0)
            avg_dt = sum(self._frame_times) / len(self._frame_times)
            self.real_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        self._last_time = current_time


class UIPanel:
    """Interactive UI panel manager for SimViewer.

    Usage::

        panel = UIPanel()
        panel.add_slider("stiffness", 1e2, 1e6, 1e5)
        panel.add_slider("friction", 0.0, 1.5, 0.5)

        # In render loop:
        panel.render(window)
        k = panel.get("stiffness")

    Parameters are stored as floats and can be read at any time.
    """

    def __init__(self, panel_name: str = "Controls"):
        self.panel_name = panel_name
        self._sliders: list[SliderConfig] = []
        self._values: dict[str, float] = {}
        self._checkboxes: dict[str, bool] = {}
        self._buttons: dict[str, bool] = {}  # True for one frame when clicked
        self.playback = PlaybackState()

        # Layout
        self._panel_x: float = 0.01
        self._panel_y: float = 0.01
        self._panel_width: float = 0.28
        self._panel_height: float = 0.5

        # Status info (set externally)
        self._info_lines: list[str] = []

        # Joint control
        self._joint_names: list[str] = []
        self._joint_values: np.ndarray | None = None
        self._joint_limits: list[tuple[float, float]] = []
        self._joint_control_enabled: bool = False
        self._joint_target: np.ndarray | None = None

    # ── Configuration ──

    def add_slider(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float = 0.0,
        fmt: str = "%.3f",
    ) -> None:
        """Register a slider parameter."""
        self._sliders.append(SliderConfig(
            label=label, min_val=min_val, max_val=max_val,
            default=default, step=step, fmt=fmt,
        ))
        self._values[label] = default

    def add_checkbox(self, label: str, default: bool = False) -> None:
        """Register a checkbox."""
        self._checkboxes[label] = default

    def add_button(self, label: str) -> None:
        """Register a button (check with was_clicked)."""
        self._buttons[label] = False

    def setup_joint_control(
        self,
        joint_names: list[str],
        initial_values: np.ndarray,
        limits: list[tuple[float, float]],
    ) -> None:
        """Enable joint angle control sliders."""
        self._joint_names = joint_names
        self._joint_values = initial_values.copy()
        self._joint_limits = limits
        self._joint_target = initial_values.copy()
        self._joint_control_enabled = True

    # ── Getters ──

    def get(self, label: str) -> float:
        """Get current slider value."""
        return self._values.get(label, 0.0)

    def get_bool(self, label: str) -> bool:
        """Get current checkbox state."""
        return self._checkboxes.get(label, False)

    def was_clicked(self, label: str) -> bool:
        """Check if a button was clicked this frame (auto-resets)."""
        val = self._buttons.get(label, False)
        self._buttons[label] = False
        return val

    def get_joint_targets(self) -> np.ndarray | None:
        """Get current joint target values (if joint control is active)."""
        return self._joint_target.copy() if self._joint_target is not None else None

    # ── Status ──

    def set_info(self, lines: list[str] | str):
        """Set status info lines."""
        if isinstance(lines, str):
            self._info_lines = lines.split("\n")
        else:
            self._info_lines = list(lines)

    def update_joint_values(self, values: np.ndarray):
        """Update displayed joint values (from simulation)."""
        if self._joint_values is not None:
            self._joint_values[:] = values[:len(self._joint_values)]

    # ── Rendering ──

    def render(self, window) -> None:
        """Render all UI panels.

        Parameters
        ----------
        window : ti.ui.Window
        """
        gui = window.get_gui()

        # Calculate dynamic height
        n_items = (
            3                              # playback row: play/pause, step, reset
            + 2                            # speed slider, fps
            + len(self._sliders)
            + len(self._checkboxes)
            + len(self._buttons)
            + len(self._info_lines)
            + 2                            # section headers
        )
        if self._joint_control_enabled:
            n_items += len(self._joint_names) + 1  # header + sliders
        height = min(0.95, 0.04 + 0.032 * n_items)

        with gui.sub_window(
            self.panel_name,
            x=self._panel_x, y=self._panel_y,
            width=self._panel_width, height=height,
        ):
            # ── Playback controls ──
            gui.text("[ Playback ]")

            # Play / Pause toggle
            btn_label = "Pause" if self.playback.playing else "Play"
            if gui.button(btn_label):
                self.playback.playing = not self.playback.playing

            # Single step (only when paused)
            if not self.playback.playing:
                if gui.button("Step"):
                    self.playback.single_step_requested = True

            # Reset
            if gui.button("Reset"):
                self.playback.reset_requested = True

            # Speed slider
            self.playback.speed = gui.slider_float(
                "Speed", self.playback.speed, 0.1, 5.0,
            )

            # FPS / sim time
            gui.text(f"FPS: {self.playback.real_fps:.0f}  "
                     f"t={self.playback.sim_time:.2f}s")

            # ── Parameter sliders ──
            if self._sliders:
                gui.text("[ Parameters ]")
                for slider in self._sliders:
                    self._values[slider.label] = gui.slider_float(
                        slider.label,
                        self._values[slider.label],
                        slider.min_val,
                        slider.max_val,
                    )

            # ── Checkboxes ──
            for label in list(self._checkboxes.keys()):
                self._checkboxes[label] = gui.checkbox(
                    label, self._checkboxes[label],
                )

            # ── Buttons ──
            for label in list(self._buttons.keys()):
                if gui.button(label):
                    self._buttons[label] = True

            # ── Joint control ──
            if self._joint_control_enabled and self._joint_target is not None:
                gui.text("[ Joints ]")
                for i, name in enumerate(self._joint_names):
                    lo, hi = self._joint_limits[i]
                    self._joint_target[i] = gui.slider_float(
                        name, self._joint_target[i], lo, hi,
                    )

            # ── Status info ──
            if self._info_lines:
                gui.text("[ Status ]")
                for line in self._info_lines:
                    gui.text(line)

    def should_step(self) -> bool:
        """Check if simulation should advance this frame.

        Handles play/pause and single-step logic.
        """
        if self.playback.playing:
            return True
        if self.playback.single_step_requested:
            self.playback.single_step_requested = False
            return True
        return False


class InfoPanel:
    """Lightweight read-only info panel (no sliders, just text).

    Useful for overlaying simulation status without interactive controls.
    """

    def __init__(
        self,
        panel_name: str = "Info",
        x: float = 0.01, y: float = 0.01,
        width: float = 0.35,
    ):
        self.panel_name = panel_name
        self._x = x
        self._y = y
        self._width = width
        self._lines: list[str] = []

    def set_text(self, text: str):
        """Set panel content."""
        self._lines = text.split("\n")

    def render(self, window) -> None:
        """Render the info panel."""
        if not self._lines:
            return
        gui = window.get_gui()
        height = 0.03 + 0.03 * len(self._lines)
        with gui.sub_window(
            self.panel_name,
            x=self._x, y=self._y,
            width=self._width, height=min(0.95, height),
        ):
            for line in self._lines:
                gui.text(line)
