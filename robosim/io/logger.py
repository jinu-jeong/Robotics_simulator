"""Frame logger: records simulation state for replay and analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    """A single snapshot of simulation state."""
    time: float
    rbd_q: np.ndarray | None = None
    rbd_qd: np.ndarray | None = None
    fem_positions: list[np.ndarray] = field(default_factory=list)
    fem_velocities: list[np.ndarray] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class SimLogger:
    """Records simulation frames at a configurable interval.

    Usage
    -----
    logger = SimLogger(interval=10)  # log every 10 steps
    for step in range(1000):
        ...
        logger.log(step, time=t, rbd_q=robot.q, fem_positions=[body.x])
    logger.save("run.npz")
    """

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.frames: list[Frame] = []

    def log(
        self,
        step: int,
        time: float,
        rbd_q: np.ndarray | None = None,
        rbd_qd: np.ndarray | None = None,
        fem_positions: list[np.ndarray] | None = None,
        fem_velocities: list[np.ndarray] | None = None,
        **extra,
    ) -> bool:
        """Record a frame if step matches interval. Returns True if logged."""
        if step % self.interval != 0:
            return False

        frame = Frame(
            time=time,
            rbd_q=rbd_q.copy() if rbd_q is not None else None,
            rbd_qd=rbd_qd.copy() if rbd_qd is not None else None,
            fem_positions=[x.copy() for x in fem_positions] if fem_positions else [],
            fem_velocities=[v.copy() for v in fem_velocities] if fem_velocities else [],
            extra={k: np.array(v) if isinstance(v, (list, np.ndarray)) else v for k, v in extra.items()},
        )
        self.frames.append(frame)
        return True

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def times(self) -> np.ndarray:
        return np.array([f.time for f in self.frames])

    def save(self, path: str | Path) -> None:
        """Save recorded frames to a compressed .npz file."""
        path = Path(path)
        data = {
            "times": self.times,
            "n_frames": self.n_frames,
        }

        # RBD state
        if self.frames and self.frames[0].rbd_q is not None:
            data["rbd_q"] = np.array([f.rbd_q for f in self.frames])
            data["rbd_qd"] = np.array([f.rbd_qd for f in self.frames])

        # FEM state (per body)
        if self.frames and self.frames[0].fem_positions:
            n_bodies = len(self.frames[0].fem_positions)
            for b in range(n_bodies):
                data[f"fem_x_{b}"] = np.array([f.fem_positions[b] for f in self.frames])
                if self.frames[0].fem_velocities:
                    data[f"fem_v_{b}"] = np.array([f.fem_velocities[b] for f in self.frames])

        # Extra scalar/array data
        extra_keys = set()
        for f in self.frames:
            extra_keys.update(f.extra.keys())
        for key in extra_keys:
            vals = [f.extra.get(key, None) for f in self.frames]
            if all(v is not None for v in vals):
                data[f"extra_{key}"] = np.array(vals)

        np.savez_compressed(str(path), **data)

    @classmethod
    def load(cls, path: str | Path) -> "SimLogger":
        """Load frames from an .npz file."""
        path = Path(path)
        npz = np.load(str(path), allow_pickle=True)

        logger = cls()
        n_frames = int(npz["n_frames"])
        times = npz["times"]

        has_rbd = "rbd_q" in npz
        rbd_q_all = npz["rbd_q"] if has_rbd else None
        rbd_qd_all = npz["rbd_qd"] if has_rbd else None

        # Count FEM bodies
        fem_body_count = 0
        while f"fem_x_{fem_body_count}" in npz:
            fem_body_count += 1

        # Extra keys
        extra_keys = [k[6:] for k in npz.files if k.startswith("extra_")]

        for i in range(n_frames):
            fem_pos = [npz[f"fem_x_{b}"][i] for b in range(fem_body_count)]
            fem_vel = []
            for b in range(fem_body_count):
                key = f"fem_v_{b}"
                if key in npz:
                    fem_vel.append(npz[key][i])

            extra = {}
            for key in extra_keys:
                extra[key] = npz[f"extra_{key}"][i]

            frame = Frame(
                time=float(times[i]),
                rbd_q=rbd_q_all[i] if has_rbd else None,
                rbd_qd=rbd_qd_all[i] if has_rbd else None,
                fem_positions=fem_pos,
                fem_velocities=fem_vel,
                extra=extra,
            )
            logger.frames.append(frame)

        return logger
