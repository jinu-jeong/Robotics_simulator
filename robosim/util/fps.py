"""FPS counter — instantaneous (smoothed) + cumulative running average.

Usage::

    fps = FPSCounter(smoothing=10)
    while running:
        ...do one frame of work...
        fps.tick()
        viewer.add_text(fps.format())          # "FPS: 60.2 (avg 58.4)"
        # or in headless: print(fps.format())

The first ``tick()`` only marks t0; FPS becomes meaningful from the
second call onwards.
"""

from __future__ import annotations

import time
from collections import deque


class FPSCounter:
    """Tracks per-frame timing.

    ``current`` — FPS over the most recent ``smoothing`` frame intervals
    (1 = literal last frame, larger = visually less jittery).

    ``average`` — cumulative FPS over the entire session since the first
    ``tick()``. Equals ``n_frames / (t_now − t_first)``.
    """

    def __init__(self, smoothing: int = 10):
        self.smoothing = max(1, int(smoothing))
        self._times: deque[float] = deque(maxlen=self.smoothing + 1)
        self._t0: float | None = None
        self._n: int = 0

    def tick(self) -> None:
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._n += 1
        self._times.append(now)

    @property
    def current(self) -> float:
        """Smoothed FPS over the trailing window."""
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0:
            return 0.0
        return (len(self._times) - 1) / dt

    @property
    def average(self) -> float:
        """Cumulative FPS since the first tick()."""
        if self._t0 is None or self._n < 2 or not self._times:
            return 0.0
        dt = self._times[-1] - self._t0
        if dt <= 0:
            return 0.0
        return (self._n - 1) / dt

    @property
    def n_frames(self) -> int:
        return self._n

    def format(self, decimals: int = 1) -> str:
        return f"FPS: {self.current:.{decimals}f} (avg {self.average:.{decimals}f})"

    def reset(self) -> None:
        self._times.clear()
        self._t0 = None
        self._n = 0
