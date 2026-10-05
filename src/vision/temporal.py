"""Temporal filters on reduced coordinates ``q``.

A single still frame has no past. A video does: independent marker noise on a
slowly changing ``q`` can be averaged. ``KalmanQ`` is a random-walk Kalman
filter; with tiny process noise it is a recursive mean. ``ema_q`` is the
one-pole filter already used on grasp force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanQ:
    """Random-walk Kalman filter on ``q ∈ R^r``.

    Predict:  q ← q,   P ← P + Q
    Update:   z = q̂_frame,  K = P (P+R)⁻¹,  q ← q + K (z − q)
    """

    r: int
    Q: np.ndarray
    R: np.ndarray
    q: np.ndarray | None = None
    P: np.ndarray | None = None
    n_updates: int = 0

    @classmethod
    def create(
        cls,
        r: int,
        *,
        process: float = 1e-8,
        measure: float = 1e-4,
    ) -> "KalmanQ":
        eye = np.eye(int(r))
        return cls(r=int(r), Q=float(process) * eye, R=float(measure) * eye)

    def reset(self) -> None:
        self.q = None
        self.P = None
        self.n_updates = 0

    def update(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, float).reshape(-1)
        if z.size != self.r:
            raise ValueError(f"measurement length {z.size} != {self.r}")
        if self.q is None:
            self.q = z.copy()
            self.P = self.R.copy()
            self.n_updates = 1
            return self.q.copy()
        P = self.P + self.Q
        S = P + self.R
        K = np.linalg.solve(S, P.T).T
        self.q = self.q + K @ (z - self.q)
        self.P = (np.eye(self.r) - K) @ P
        self.n_updates += 1
        return self.q.copy()


def filter_sequence(zs: np.ndarray, kind: str, *, r: int | None = None,
                    ema_alpha: float = 0.25, process: float = 1e-8,
                    measure: float = 1e-4) -> np.ndarray:
    """Filter a ``(T, r)`` stream. ``kind`` is ``raw`` | ``batch`` | ``ema`` | ``kalman``."""
    zs = np.asarray(zs, float)
    if zs.ndim != 2:
        raise ValueError("zs must be (T, r)")
    T, rr = zs.shape
    r = int(r or rr)
    kind = kind.lower()
    if kind == "raw":
        return zs.copy()
    if kind == "batch":
        out = np.zeros_like(zs)
        c = np.zeros(r)
        for t in range(T):
            c = c + zs[t]
            out[t] = c / (t + 1)
        return out
    if kind == "ema":
        a = float(ema_alpha)
        out = np.zeros_like(zs)
        x = zs[0].copy()
        out[0] = x
        for t in range(1, T):
            x = (1.0 - a) * x + a * zs[t]
            out[t] = x
        return out
    if kind == "kalman":
        kf = KalmanQ.create(r, process=process, measure=measure)
        out = np.zeros_like(zs)
        for t in range(T):
            out[t] = kf.update(zs[t])
        return out
    raise ValueError(f"unknown filter {kind!r}")
