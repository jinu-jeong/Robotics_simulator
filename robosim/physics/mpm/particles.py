"""Particle state container for MPM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Particles:
    """Lagrangian carriers of mass, momentum, and deformation history.

    All arrays are float64 with leading dimension ``P`` (particle count).
    ``F`` starts at identity and ``C`` (APIC affine velocity) at zero; both
    evolve each step.
    """

    x: np.ndarray        # (P, 3) positions
    v: np.ndarray        # (P, 3) velocities
    m: np.ndarray        # (P,)   masses
    V0: np.ndarray       # (P,)   reference volumes
    F: np.ndarray = field(default=None)   # (P, 3, 3) deformation gradient
    C: np.ndarray = field(default=None)   # (P, 3, 3) APIC affine matrix

    def __post_init__(self) -> None:
        self.x = np.ascontiguousarray(self.x, dtype=np.float64)
        self.v = np.ascontiguousarray(self.v, dtype=np.float64)
        self.m = np.ascontiguousarray(self.m, dtype=np.float64)
        self.V0 = np.ascontiguousarray(self.V0, dtype=np.float64)
        P = self.x.shape[0]
        if self.x.shape != (P, 3):
            raise ValueError(f"x must be (P, 3), got {self.x.shape}")
        if self.v.shape != (P, 3):
            raise ValueError(f"v must be (P, 3), got {self.v.shape}")
        if self.m.shape != (P,):
            raise ValueError(f"m must be (P,), got {self.m.shape}")
        if self.V0.shape != (P,):
            raise ValueError(f"V0 must be (P,), got {self.V0.shape}")
        if self.F is None:
            self.F = np.broadcast_to(np.eye(3), (P, 3, 3)).copy()
        if self.C is None:
            self.C = np.zeros((P, 3, 3), dtype=np.float64)

    @property
    def n(self) -> int:
        return self.x.shape[0]
