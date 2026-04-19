"""Background Eulerian grid for MPM.

Uniform Cartesian node grid. Node (i, j, k) sits at ``origin + (i, j, k) * dx``.
Mass and velocity (or momentum during assembly) live at nodes and are
re-zeroed every substep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    dx: float
    origin: np.ndarray           # (3,) world coord of node (0, 0, 0)
    shape: tuple[int, int, int]  # node counts along each axis

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        nx, ny, nz = self.shape
        self.m = np.zeros((nx, ny, nz), dtype=np.float64)
        self.v = np.zeros((nx, ny, nz, 3), dtype=np.float64)

    def reset(self) -> None:
        self.m.fill(0.0)
        self.v.fill(0.0)

    @classmethod
    def from_bounds(
        cls,
        lower: np.ndarray,
        upper: np.ndarray,
        dx: float,
        pad: int = 2,
    ) -> "Grid":
        """Build a grid covering ``[lower, upper]`` with ``pad`` extra nodes
        on each side (needed so the 3-node stencil never walks off the grid)."""
        lower = np.asarray(lower, dtype=np.float64).reshape(3)
        upper = np.asarray(upper, dtype=np.float64).reshape(3)
        origin = lower - pad * dx
        extent = (upper - origin) + pad * dx
        shape = tuple(int(np.ceil(e / dx)) + 1 for e in extent)
        return cls(dx=float(dx), origin=origin, shape=shape)  # type: ignore[arg-type]
