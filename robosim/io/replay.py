"""Replay player: plays back logged simulation frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from robosim.io.logger import SimLogger


class ReplayPlayer:
    """Plays back a recorded simulation.

    Usage
    -----
    player = ReplayPlayer.from_file("run.npz")
    viewer = SimViewer(...)
    viewer.add_callback(player.make_callback(robot, fem_bodies, viewer))
    viewer.show()
    """

    def __init__(self, logger: SimLogger):
        self.logger = logger
        self._frame_idx = 0
        self._paused = False
        self._speed = 1.0

    @classmethod
    def from_file(cls, path: str | Path) -> "ReplayPlayer":
        return cls(SimLogger.load(path))

    @property
    def n_frames(self) -> int:
        return self.logger.n_frames

    @property
    def current_frame(self):
        idx = min(self._frame_idx, self.n_frames - 1)
        return self.logger.frames[idx]

    @property
    def progress(self) -> float:
        if self.n_frames <= 1:
            return 1.0
        return self._frame_idx / (self.n_frames - 1)

    def toggle_pause(self):
        self._paused = not self._paused

    def set_speed(self, speed: float):
        self._speed = max(0.1, min(speed, 10.0))

    def seek(self, fraction: float):
        """Jump to a position (0.0 = start, 1.0 = end)."""
        self._frame_idx = int(np.clip(fraction, 0, 1) * (self.n_frames - 1))

    def advance(self) -> bool:
        """Advance by one frame. Returns False if at end."""
        if self._paused or self._frame_idx >= self.n_frames - 1:
            return False
        self._frame_idx += 1
        return True

    def make_callback(self, robot=None, fem_bodies=None, viewer=None):
        """Create a step callback for SimViewer that plays back frames.

        Parameters
        ----------
        robot : Robot instance to update q/qd
        fem_bodies : list of DeformableBody to update positions
        viewer : SimViewer for mesh updates and HUD
        """

        def callback(step):
            if not self.advance():
                return

            frame = self.current_frame

            # Update RBD state
            if robot is not None and frame.rbd_q is not None:
                robot.q = frame.rbd_q
                robot.qd = frame.rbd_qd

            # Update FEM positions
            if fem_bodies is not None:
                for i, body in enumerate(fem_bodies):
                    if i < len(frame.fem_positions):
                        body.x = frame.fem_positions[i]

            # HUD
            if viewer is not None:
                t = frame.time
                total = self.logger.frames[-1].time if self.n_frames > 0 else 0
                info = (
                    f"REPLAY  t={t:.3f}s / {total:.3f}s\n"
                    f"Frame {self._frame_idx}/{self.n_frames - 1}\n"
                    f"{'PAUSED' if self._paused else 'PLAYING'}"
                )
                viewer.add_text(info)

        return callback
