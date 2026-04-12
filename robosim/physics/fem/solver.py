"""FEM Solver: manages deformable body simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean
from robosim.physics.fem.assembly import (
    assemble_forces,
    assemble_mass_matrix,
    precompute_element_data,
)
from robosim.physics.fem.integrator import implicit_euler_step


@dataclass
class DeformableBody:
    """A single FEM deformable body.

    Attributes
    ----------
    name : identifier
    mesh : reference tet mesh
    material : constitutive model
    density : material density (kg/m^3)
    fixed_nodes : indices of nodes with Dirichlet BCs (clamped)
    """

    name: str
    mesh: TetMesh
    material: CorotationalElastic | NeoHookean = field(
        default_factory=lambda: CorotationalElastic()
    )
    density: float = 1000.0
    fixed_nodes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    # State (set on initialize)
    x: np.ndarray | None = field(default=None, repr=False)
    v: np.ndarray | None = field(default=None, repr=False)

    # Precomputed (set on initialize)
    _dN_list: list[np.ndarray] | None = field(default=None, repr=False)
    _volumes: np.ndarray | None = field(default=None, repr=False)
    _M: sp.csr_matrix | None = field(default=None, repr=False)
    _fixed_dofs: np.ndarray | None = field(default=None, repr=False)

    @property
    def fixed_dofs(self) -> np.ndarray:
        """Convert fixed node indices to DOF indices (3 DOFs per node)."""
        if self._fixed_dofs is not None:
            return self._fixed_dofs
        dofs = []
        for n in self.fixed_nodes:
            dofs.extend([n * 3, n * 3 + 1, n * 3 + 2])
        self._fixed_dofs = np.array(dofs, dtype=np.int64)
        return self._fixed_dofs


@dataclass
class FEMSolver:
    """Finite Element Method solver for deformable bodies.

    Manages one or more DeformableBody instances and advances them
    using implicit Euler integration.
    """

    bodies: list[DeformableBody] = field(default_factory=list)
    dt: float = 0.01
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    damping: float = 0.01

    _time: float = field(default=0.0, repr=False)

    def add_body(self, body: DeformableBody):
        self.bodies.append(body)

    def initialize(self, dt: float) -> None:
        """Initialize all bodies: precompute element data, mass matrices, state."""
        self.dt = dt

        for body in self.bodies:
            # Set initial state = reference configuration
            body.x = body.mesh.nodes.copy()
            body.v = np.zeros_like(body.mesh.nodes)

            # Precompute
            body._dN_list, body._volumes = precompute_element_data(body.mesh)
            body._M = assemble_mass_matrix(body.mesh, body.density, body._volumes)
            body._fixed_dofs = None  # reset cache

        self._time = 0.0

    def step(self, dt: float | None = None,
             extra_forces: dict[int, np.ndarray] | None = None) -> None:
        """Advance all bodies by one time step.

        Parameters
        ----------
        dt : time step override
        extra_forces : optional dict {body_index: (n_dof,) force vector}
                       e.g. contact forces to add to f_ext
        """
        if dt is None:
            dt = self.dt

        for body_idx, body in enumerate(self.bodies):
            # External forces: gravity
            f_ext = np.zeros(body.mesh.n_nodes * 3)
            M_diag = body._M.diagonal()
            for d in range(3):
                f_ext[d::3] += M_diag[d::3] * self.gravity[d]

            # Add extra forces (e.g. contact)
            if extra_forces and body_idx in extra_forces:
                f_ext += extra_forces[body_idx]

            result = implicit_euler_step(
                mesh=body.mesh,
                x=body.x,
                v=body.v,
                f_ext=f_ext,
                dt=dt,
                material=body.material,
                M=body._M,
                dN_list=body._dN_list,
                volumes=body._volumes,
                fixed_dofs=body.fixed_dofs,
                damping=self.damping,
            )

            body.x = result.x_new
            body.v = result.v_new

        self._time += dt

    @property
    def time(self) -> float:
        return self._time

    def get_positions(self) -> np.ndarray:
        """Get flattened positions of all bodies."""
        return np.concatenate([b.x.ravel() for b in self.bodies])

    def get_velocities(self) -> np.ndarray:
        return np.concatenate([b.v.ravel() for b in self.bodies])

    def elastic_energy(self, body_idx: int = 0) -> float:
        """Compute elastic strain energy of a body."""
        body = self.bodies[body_idx]
        from robosim.physics.fem.elements import compute_deformation_gradient

        energy = 0.0
        for e in range(body.mesh.n_elements):
            nodes_e = body.mesh.elements[e]
            dN = body._dN_list[e]
            vol = body._volumes[e]
            if vol < 1e-20:
                continue

            x_def = body.x[nodes_e]
            F = compute_deformation_gradient(dN, x_def)

            if isinstance(body.material, CorotationalElastic):
                _, S = __import__(
                    'robosim.physics.fem.elements', fromlist=['polar_decomposition']
                ).polar_decomposition(F)
                eps = S - np.eye(3)
                mu = body.material.mu
                lam = body.material.lam
                energy += vol * (mu * np.sum(eps**2) + 0.5 * lam * np.trace(eps)**2)
            else:
                # Neo-Hookean
                I1 = np.trace(F.T @ F)
                J = np.linalg.det(F)
                if J < 1e-10:
                    J = 1e-10
                mu = body.material.mu
                lam = body.material.lam
                energy += vol * (0.5 * mu * (I1 - 3) - mu * np.log(J) + 0.5 * lam * np.log(J)**2)

        return energy

    def kinetic_energy(self, body_idx: int = 0) -> float:
        """Compute kinetic energy of a body."""
        body = self.bodies[body_idx]
        v_flat = body.v.ravel()
        return 0.5 * float(v_flat @ body._M @ v_flat)
