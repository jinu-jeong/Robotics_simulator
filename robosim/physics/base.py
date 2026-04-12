"""Base protocols for physics solvers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PhysicsSolver(Protocol):
    """Protocol for any physics solver (RBD, FEM, etc.)."""

    def initialize(self, dt: float) -> None: ...
    def step(self, dt: float, external_forces: np.ndarray) -> None: ...
    def get_positions(self) -> np.ndarray: ...
    def get_velocities(self) -> np.ndarray: ...


@runtime_checkable
class CouplingStrategy(Protocol):
    """Protocol for RBD-FEM boundary coupling."""

    def setup(self, rbd_solver: object, fem_solver: object, boundary_map: object) -> None: ...
    def enforce_constraints(self, dt: float) -> None: ...
    def get_coupling_forces(self) -> tuple[np.ndarray, np.ndarray]: ...
