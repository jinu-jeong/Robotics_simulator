"""Quasi-static 3D linear-elastic FEM solver (NumPy/SciPy).

Solves ``K u = f`` with Dirichlet constraints via DOF partition:

    K_ff u_f = f_f - K_fc u_c,           R_c = K_cf u_f + K_cc u_c - f_c

``R_c`` are the reaction forces at constrained DOFs; global equilibrium
requires ``Σ R + Σ f_applied = 0`` which is reported as a check.

:class:`LinearFEM` assembles once and keeps the sparse LU factorisation of
``K_ff`` so many load cases (dataset generation, inverse problems) reuse it.
:class:`FEMResult` bundles everything downstream code and the Taichi viewer
need, in SI units (m, N, Pa).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..geometry.finger import TetMesh
from .assembly import assemble_global_stiffness
from .boundary import DofPartition
from .material import LinearElasticMaterial
from .stress import element_strains, element_stresses, element_to_nodes, von_mises


@dataclass
class FEMResult:
    """Output of one linear FEM solve.

    ``u`` and ``f`` are stored as (N, 3) arrays; use ``.reshape(-1)`` for the
    flat DOF vector (``3*node + comp`` ordering).
    """

    mesh: TetMesh
    material: LinearElasticMaterial
    partition: DofPartition
    u: np.ndarray  # (N, 3) displacement [m]
    f: np.ndarray  # (N, 3) applied nodal forces [N]
    reactions: np.ndarray  # (N, 3) reaction forces at constrained dofs [N] (zero elsewhere)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def nodes_original(self) -> np.ndarray:
        return self.mesh.nodes

    @property
    def nodes_deformed(self) -> np.ndarray:
        return self.mesh.nodes + self.u

    @property
    def fixed_nodes(self) -> np.ndarray:
        return self.partition.fully_fixed_nodes()

    @property
    def u_flat(self) -> np.ndarray:
        return self.u.reshape(-1)

    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.u, axis=1)

    def max_displacement(self) -> float:
        return float(self.displacement_magnitude().max())

    def applied_resultant(self) -> np.ndarray:
        return self.f.sum(axis=0)

    def reaction_resultant(self) -> np.ndarray:
        return self.reactions.sum(axis=0)

    def equilibrium_residual(self) -> float:
        """|Σ f + Σ R| / |Σ f| – should be ~1e-12 for a converged linear solve."""
        fa = self.applied_resultant()
        return float(np.linalg.norm(fa + self.reaction_resultant()) / max(np.linalg.norm(fa), 1e-300))

    def to_visualization_state(
        self,
        contact_points=None,
        ground_truth_force=None,
        estimated_force=None,
        objects=None,
        observation_camera=None,
        node_scalar: np.ndarray | None = None,
        node_scalar_name: str = "scalar",
        info: dict | None = None,
    ):
        """Package the result for the Taichi viewer (import kept local so the
        FEM package stays Taichi-free)."""
        from ..visualization.state import VisualizationState

        base_info = {
            "max |u|": f"{1e3 * self.max_displacement():.4f} mm",
            "sum F applied": f"({', '.join(f'{x:+.3f}' for x in self.applied_resultant())}) N",
            "equilibrium residual": f"{self.equilibrium_residual():.2e}",
        }
        base_info.update(info or {})
        return VisualizationState(
            nodes_original=self.mesh.nodes,
            surface_faces=self.mesh.surface_faces,
            surface_edges=self.mesh.surface_edges,
            element_edges=self.mesh.element_edges,
            displacement=self.u,
            fixed_nodes=self.fixed_nodes,
            contact_points=list(contact_points or []),
            ground_truth_force=ground_truth_force,
            estimated_force=estimated_force,
            objects=list(objects or []),
            observation_camera=observation_camera,
            node_scalar=node_scalar,
            node_scalar_name=node_scalar_name,
            info=base_info,
        )


def tie_transformation(mesh: TetMesh) -> sp.csr_matrix | None:
    """Master–slave map ``T`` (3N × 3N) for hanging-node ties, or ``None`` if there are none.

    ``u = T ũ`` with ``T`` the identity on master DOFs and the tie weights on slave
    rows (slave columns are zero, so ``ũ_slave`` is a dummy that the solver pins to 0).
    """
    S = mesh.n_ties
    if S == 0:
        return None
    n = mesh.n_dofs
    eye_mask = np.ones(n, dtype=bool)
    slave_dofs = (3 * mesh.tie_slaves[:, None] + np.arange(3)[None, :]).reshape(-1)
    eye_mask[slave_dofs] = False
    keep = np.nonzero(eye_mask)[0]
    rows = [keep]
    cols = [keep]
    vals = [np.ones(len(keep))]
    for comp in range(3):
        r = np.repeat(3 * mesh.tie_slaves + comp, mesh.tie_masters.shape[1])
        c = (3 * mesh.tie_masters + comp).reshape(-1)
        rows.append(r)
        cols.append(c)
        vals.append(mesh.tie_weights.reshape(-1))
    T = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    T.sum_duplicates()
    return T


class LinearFEM:
    """Assembled linear-elastic system with reusable factorisation.

    ``K`` is the unconstrained assembled stiffness (used by the ROM). With
    hanging-node ties the solved system is ``K̃ = Tᵀ K T + P_s`` (``P_s`` pins the
    dummy slave unknowns), ``f̃ = Tᵀ f``, and the physical displacement is ``T ũ``.
    """

    def __init__(self, mesh: TetMesh, material: LinearElasticMaterial, partition: DofPartition) -> None:
        if partition.n_dof != mesh.n_dofs:
            raise ValueError("partition does not match mesh DOF count")
        self.mesh = mesh
        self.material = material
        self.partition = partition
        self.K, self.elements = assemble_global_stiffness(mesh.nodes, mesh.element_blocks, material, return_element_data=True)
        self.volumes = self.elements.volumes
        self.T = tie_transformation(mesh)
        if self.T is not None:
            slave_dofs = (3 * mesh.tie_slaves[:, None] + np.arange(3)[None, :]).reshape(-1)
            if np.intersect1d(slave_dofs, partition.fixed_dofs).size:
                raise ValueError("a hanging (tied) node cannot carry a Dirichlet condition")
            P = sp.coo_matrix((np.ones(len(slave_dofs)), (slave_dofs, slave_dofs)), shape=self.K.shape).tocsr()
            K_sys = (self.T.T @ self.K @ self.T + P).tocsr()
        else:
            K_sys = self.K.tocsr()
        f_idx, c_idx = partition.free_dofs, partition.fixed_dofs
        self.K_sys = K_sys
        self.K_ff = K_sys[f_idx][:, f_idx].tocsc()
        self.K_fc = K_sys[f_idx][:, c_idx].tocsr()
        self.K_cf = K_sys[c_idx][:, f_idx].tocsr()
        self.K_cc = K_sys[c_idx][:, c_idx].tocsr()
        self._lu = None

    # ------------------------------------------------------------ ties
    def _reduce_load(self, f: np.ndarray) -> np.ndarray:
        return f if self.T is None else np.asarray(self.T.T @ f).reshape(-1)

    def _expand(self, u_tilde: np.ndarray) -> np.ndarray:
        return u_tilde if self.T is None else np.asarray(self.T @ u_tilde).reshape(-1)

    # ------------------------------------------------------------ props
    @property
    def n_dof(self) -> int:
        return self.mesh.n_dofs

    @property
    def n_free(self) -> int:
        return self.partition.n_free

    def factorize(self) -> None:
        if self._lu is None:
            self._lu = spla.splu(self.K_ff)

    # ------------------------------------------------------------ solve
    def solve_free(self, f: np.ndarray) -> np.ndarray:
        """Return u_f for the flat load vector ``f`` (3N,) honouring prescribed u_c."""
        f = np.asarray(f, dtype=float).reshape(-1)
        if f.shape[0] != self.n_dof:
            raise ValueError(f"load vector must have length {self.n_dof}")
        self.factorize()
        ft = self._reduce_load(f)
        rhs = ft[self.partition.free_dofs] - self.K_fc @ self.partition.prescribed
        u_tilde_f = self._lu.solve(rhs)
        if self.T is None:
            return u_tilde_f
        return self._expand(self.partition.full_vector(u_tilde_f))[self.partition.free_dofs]

    def solve(self, f: np.ndarray, meta: dict | None = None) -> FEMResult:
        """Full solve returning a :class:`FEMResult`."""
        f = np.asarray(f, dtype=float).reshape(-1)
        u_f = self.solve_free(f)
        u = self.partition.full_vector(u_f)
        c = self.partition.fixed_dofs
        R = np.zeros(self.n_dof)
        if self.T is None:
            R[c] = self.K_cf @ u_f + self.K_cc @ self.partition.prescribed - f[c]
        else:  # reactions include the forces transmitted through the ties
            R[c] = np.asarray(self.T.T @ (self.K @ u - f)).reshape(-1)[c]
        return FEMResult(
            mesh=self.mesh,
            material=self.material,
            partition=self.partition,
            u=u.reshape(-1, 3),
            f=f.reshape(-1, 3),
            reactions=R.reshape(-1, 3),
            meta=dict(meta or {}),
        )

    # ------------------------------------------------------------ loads
    def body_force_vector(self, acceleration) -> np.ndarray:
        """Consistent nodal load (3N,) of a uniform body force ``ρ a`` (e.g. self-weight).

        ``f_i = ρ a ∫ N_i dV`` with the element quadrature (exact for straight-sided
        quadratic elements; ``Σ_i f_i = ρ V a``). Returns zeros when ρ = 0.
        """
        a = np.asarray(acceleration, dtype=float).reshape(3)
        f = np.zeros((self.mesh.n_nodes, 3))
        rho = float(self.material.density)
        if rho <= 0.0 or not np.any(a):
            return f.reshape(-1)
        for conn, m in zip(self.elements.conn, self.elements.mass_shares):
            np.add.at(f, conn.reshape(-1), (rho * m.reshape(-1))[:, None] * a[None, :])
        return f.reshape(-1)

    # ------------------------------------------------------------ post
    def internal_forces(self, u: np.ndarray) -> np.ndarray:
        """Generalised internal nodal forces ``Tᵀ K u`` (``K u`` without ties) for a flat or
        (N,3) displacement; zero at slave DOFs, equal to the applied load at equilibrium."""
        return self._reduce_load(self.K @ np.asarray(u, dtype=float).reshape(-1))

    def stresses(self, result: FEMResult) -> dict[str, np.ndarray]:
        """Element strains/stresses plus nodal von Mises for colouring."""
        eps = np.concatenate([element_strains(B, conn, result.u_flat) for conn, B in zip(self.elements.conn, self.elements.B)], axis=0) \
            if self.elements.conn else np.zeros((0, 6))
        sig = element_stresses(eps, self.material)
        vm = von_mises(sig)
        return {
            "strain": eps,
            "stress": sig,
            "von_mises": vm,
            "von_mises_nodal": element_to_nodes(vm, self.elements.conn, self.volumes, self.mesh.n_nodes),
        }


def solve_linear_fem(mesh: TetMesh, material: LinearElasticMaterial, partition: DofPartition, f: np.ndarray) -> FEMResult:
    """One-shot convenience wrapper around :class:`LinearFEM`."""
    return LinearFEM(mesh, material, partition).solve(f)
