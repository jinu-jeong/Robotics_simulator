"""PhysicsWorld: unified simulation loop for RBD + FEM + Contact + Coupling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.physics.rbd.solver import RBDSolver
from robosim.physics.fem.solver import FEMSolver
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.coupling.penalty import PenaltyCoupling
from robosim.physics.coupling.interface import BoundaryMap


@dataclass
class PhysicsWorld:
    """Orchestrates RBD, FEM, contact, and coupling each time step."""

    rbd_solver: RBDSolver | None = None
    fem_solver: FEMSolver | None = None
    contact: ContactSolver | None = None
    coupling: PenaltyCoupling | None = None

    dt: float = 0.001
    _time: float = field(default=0.0, repr=False)

    def initialize(self, dt: float) -> None:
        self.dt = dt
        if self.rbd_solver is not None:
            self.rbd_solver.initialize(dt=dt)
        if self.fem_solver is not None:
            self.fem_solver.initialize(dt=dt)
        self._time = 0.0

    def setup_coupling(
        self,
        boundary_maps: list[BoundaryMap],
        stiffness: float = 1e5,
        damping: float = 1e3,
    ) -> None:
        """Set up penalty coupling between RBD and FEM."""
        self.coupling = PenaltyCoupling(stiffness=stiffness, damping=damping)
        self.coupling.setup(self.rbd_solver, self.fem_solver, boundary_maps)

    def step(self) -> None:
        """Advance the world by one time step."""
        dt = self.dt

        # --- 1. Coupling forces ---
        rbd_wrenches: dict[int, np.ndarray] = {}
        fem_extra: dict[int, np.ndarray] = {}

        if self.coupling is not None:
            rbd_wrenches, fem_extra = self.coupling.compute_forces()

        # --- 2. RBD contact forces ---
        if self.rbd_solver is not None and self.contact is not None:
            self.contact.register_rbd(self.rbd_solver) if not self.contact._body_map else None
            contact_wrenches = self.contact.compute_rbd_contact_forces(self.rbd_solver)
            for link_idx, w in contact_wrenches.items():
                if link_idx in rbd_wrenches:
                    rbd_wrenches[link_idx] += w
                else:
                    rbd_wrenches[link_idx] = w

        # --- 3. Step RBD ---
        if self.rbd_solver is not None:
            self.rbd_solver.clear_external_forces()
            for link_idx, wrench in rbd_wrenches.items():
                self.rbd_solver.set_external_force(link_idx, wrench)
            self.rbd_solver.step()

        # --- 4. Step FEM ---
        if self.fem_solver is not None:
            self.fem_solver.step(extra_forces=fem_extra if fem_extra else None)

        # --- 5. FEM ground contact (impulse-based, post-step) ---
        if self.fem_solver is not None and self.contact is not None:
            for body in self.fem_solver.bodies:
                self.contact.resolve_fem_contact(body, restitution=0.3, friction_mu=0.5)

        # --- 6. FEM-FEM contact (impulse-based, post-step) ---
        if self.fem_solver is not None and self.contact is not None:
            if len(self.fem_solver.bodies) > 1:
                if not self.contact._fem_colliders:
                    self.contact.register_fem(self.fem_solver)
                self.contact.resolve_fem_fem_all(
                    self.fem_solver, restitution=0.1, friction_mu=0.5,
                )

        # --- 7. RBD-FEM contact (impulse-based, post-step) ---
        if (self.rbd_solver is not None and self.fem_solver is not None
                and self.contact is not None):
            if not self.contact._fem_colliders:
                self.contact.register_fem(self.fem_solver)
            self.contact.resolve_rbd_fem_all(
                self.rbd_solver, self.fem_solver,
                restitution=0.1, friction_mu=0.5,
            )

        self._time += dt

    @property
    def time(self) -> float:
        return self._time
