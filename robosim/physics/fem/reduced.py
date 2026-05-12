"""Craig-Bampton reduced-order model with body-level corotational rotation.

Two operating modes
-------------------
**Free body** (fixed_nodes=[] — default)
    Large rigid-body rotation handled by Kabsch best-fit R each step
    (Floating Frame of Reference / stiffness warping).  Good for grasping,
    drop tests, flying objects.

**Anchored body** (fixed_nodes=<array>)
    Some nodes are clamped in place (Dirichlet BCs).  Corotational disabled
    (R=I), centroid held fixed.  Good for cantilevers, structural members.

Theory
------
Component Mode Synthesis (Craig & Bampton 1968) splits free DOFs into
boundary (b) and interior (i):

  Φ_CB = [ I      0   ]   ← free boundary DOFs
          [ Ψ_c   Φ_k ]   ← free interior DOFs

  Ψ_c  = -K_ii⁻¹ K_ib          (constraint modes)
  Φ_k  : k smallest eigenvectors of (K_ii, M_ii)   (fixed-interface modes)

Reduced constant matrices:  K_r = Φ_CB^T K_free Φ_CB
                             M_r = Φ_CB^T M_free Φ_CB

One LU factorisation of A_r = M_r/dt² + β M_r/dt + K_r gives a single
back-solve per timestep — no assembly, no Newton iterations.

References
----------
Craig R.R. & Bampton M.C.C. (1968).  AIAA Journal, 6(7), 1313-1319.
Shabana A.A. (2005). Dynamics of Multibody Systems.  Cambridge Univ. Press.
Müller M. et al. (2002). Stable real-time deformations.  ACM SCA 2002.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lu_factor, lu_solve  # kept for fallback

from robosim.physics.fem.mesh import TetMesh, FEMesh

try:
    from robosim import _cpp as _robosim_cpp
    _HAVE_CPP_CB = hasattr(_robosim_cpp, "cb")
except ImportError:
    _robosim_cpp = None
    _HAVE_CPP_CB = False


def _cpp_cb_prepare(body) -> bool:
    """Materialise C-contiguous float64 views of the precomputed CB
    arrays on first use. Returns True if everything is ready (called
    again after model rebuild → re-runs the prep)."""
    if not _HAVE_CPP_CB:
        return False
    if getattr(body, "_cpp_cb_ready", False):
        return True
    if (body._Phi_CB is None or body._M_r is None
            or body._A_r_inv is None or body._M_diag is None
            or body.x is None):
        return False
    body._cpp_Phi_CB  = np.ascontiguousarray(body._Phi_CB,  dtype=np.float64)
    body._cpp_M_r     = np.ascontiguousarray(body._M_r,     dtype=np.float64)
    body._cpp_A_r_inv = np.ascontiguousarray(body._A_r_inv, dtype=np.float64)
    body._cpp_M_diag  = np.ascontiguousarray(body._M_diag,  dtype=np.float64)
    body._cpp_nodes   = np.ascontiguousarray(body.mesh.nodes, dtype=np.float64)
    if body._anchored:
        # C_q maps free DOFs → q_r. Stored on body.
        body._cpp_C_q = np.ascontiguousarray(body._C_q, dtype=np.float64)
        body._cpp_free_dofs  = np.ascontiguousarray(body._free_dofs,  dtype=np.int32)
        body._cpp_fixed_dofs = np.ascontiguousarray(body._fixed_dofs, dtype=np.int32)
    else:
        # Free body uses C_q on the *full* (n_dof,) displacement after
        # body-frame rotation — exact same C_q as the anchored case
        # but never sliced by free_dofs (there are none).
        body._cpp_C_q = np.ascontiguousarray(body._C_q, dtype=np.float64)
        body._cpp_x_ref_body = np.ascontiguousarray(body._x_ref_body, dtype=np.float64)
    body._cpp_cb_ready = True
    return True


def _try_cpp_step_anchored(body, dt: float, extra_forces) -> bool:
    """Run the C++ anchored step. Returns True on success."""
    if not _cpp_cb_prepare(body):
        return False
    if not body._anchored:
        return False
    n_dof = body.mesh.n_nodes * 3
    if extra_forces is not None and 0 in extra_forces:
        f_ext = np.ascontiguousarray(extra_forces[0], dtype=np.float64).reshape(-1)
    else:
        f_ext = np.empty(0, dtype=np.float64)
    grav = np.ascontiguousarray(body.gravity, dtype=np.float64).reshape(3)
    x_in = np.ascontiguousarray(body.x, dtype=np.float64)
    v_in = np.ascontiguousarray(body.v, dtype=np.float64)
    x_new, v_new, q_r_new = _robosim_cpp.cb.step_anchored(
        x_in, v_in, body._cpp_nodes,
        body._cpp_Phi_CB, body._cpp_M_r, body._cpp_A_r_inv,
        body._cpp_C_q, body._cpp_M_diag,
        body._cpp_free_dofs, body._cpp_fixed_dofs,
        grav, f_ext, float(dt), float(body.damping),
    )
    body.x = np.ascontiguousarray(x_new)
    body.v = np.ascontiguousarray(v_new)
    body.q_r = q_r_new
    return True


def _try_cpp_step_free(body, dt: float, extra_forces) -> bool:
    """Run the C++ free-body step. Returns True on success."""
    if not _cpp_cb_prepare(body):
        return False
    if body._anchored:
        return False
    if extra_forces is not None and 0 in extra_forces:
        f_ext = np.ascontiguousarray(extra_forces[0], dtype=np.float64).reshape(-1)
    else:
        f_ext = np.empty(0, dtype=np.float64)
    grav = np.ascontiguousarray(body.gravity, dtype=np.float64).reshape(3)
    x_in = np.ascontiguousarray(body.x, dtype=np.float64)
    v_in = np.ascontiguousarray(body.v, dtype=np.float64)
    x_new, v_new, q_r_new = _robosim_cpp.cb.step_free(
        x_in, v_in, body._cpp_x_ref_body,
        body._cpp_Phi_CB, body._cpp_M_r, body._cpp_A_r_inv,
        body._cpp_C_q, body._cpp_M_diag, grav, f_ext,
        float(dt), float(body.damping), float(body._m_total),
    )
    body.x = np.ascontiguousarray(x_new)
    body.v = np.ascontiguousarray(v_new)
    body.q_r = q_r_new
    return True

try:
    from robosim.physics.fem.partition import nested_dissection_order
    _HAS_PYMETIS = True
except ImportError:
    _HAS_PYMETIS = False
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean
from robosim.physics.fem.assembly import (
    precompute_element_data,
    assemble_stiffness,
    assemble_mass_matrix,
    ElementIntegrationData,
    CompositeIntegrationData,
)


# ═══════════════════════════════════════════════════════════════
# Craig-Bampton Body
# ═══════════════════════════════════════════════════════════════

class CraigBamptonBody:
    """Reduced deformable body: Craig-Bampton CMS + optional body-level corotational.

    Parameters
    ----------
    mesh : TetMesh or FEMesh  reference mesh
    material : constitutive model  (default CorotationalElastic)
    density : kg/m³
    n_modes : fixed-interface normal modes to keep  (0 = pure static condensation)
    fixed_nodes : node indices with Dirichlet BCs (clamped, zero displacement).
                  If non-empty: corotational disabled (R=I), centroid fixed.
                  If empty (default): body-level Kabsch corotational enabled.
    gravity : (3,) gravity vector  [default 0,0,-9.81]
    damping : Rayleigh mass-proportional β
    name : label

    Attributes
    ----------
    x : (n_nodes, 3)  current world positions  ← directly readable/writable
    v : (n_nodes, 3)  current world velocities ← directly readable/writable
    mesh : unchanged reference mesh
    """

    def __init__(
        self,
        mesh: TetMesh | FEMesh,
        material: CorotationalElastic | NeoHookean | None = None,
        density: float = 1000.0,
        n_modes: int = 6,
        fixed_nodes: np.ndarray | None = None,
        gravity: np.ndarray | None = None,
        damping: float = 0.01,
        name: str = "cb_body",
    ) -> None:
        if material is None:
            material = CorotationalElastic()

        self.mesh = mesh
        self.material = material
        self.density = density
        self.n_modes = n_modes
        self.fixed_nodes = (np.asarray(fixed_nodes, dtype=np.int64)
                            if fixed_nodes is not None and len(fixed_nodes) > 0
                            else np.array([], dtype=np.int64))
        self.gravity = np.asarray(gravity if gravity is not None
                                  else [0.0, 0.0, -9.81], dtype=np.float64)
        self.damping = damping
        self.name = name

        # Primary state
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None

        # Precomputed (set by _build_cb_basis)
        self._Phi_CB: np.ndarray | None = None    # (n_dof, n_r)
        self._K_r: np.ndarray | None = None
        self._M_r: np.ndarray | None = None
        self._C_q: np.ndarray | None = None       # (n_r, n_free_dof) projection
        self._M_diag: np.ndarray | None = None    # (n_dof,) full mass diagonal
        self._m_total: float = 0.0
        self._free_dofs: np.ndarray | None = None  # global free DOF indices
        self._fixed_dofs: np.ndarray | None = None
        self._x_ref_body: np.ndarray | None = None
        self._t_ref: np.ndarray | None = None
        self._n_r: int = 0
        self._n_b: int = 0
        self._n_modes_actual: int = 0

        # Whether this is an anchored body (fixed nodes present)
        self._anchored: bool = len(self.fixed_nodes) > 0

        # Integration data for post-processing (von Mises etc.)
        self._dN_list = None
        self._volumes = None

        # Current reduced modal coordinates (updated each step)
        self.q_r: np.ndarray | None = None

        # LU (set by initialize)
        self._A_r_lu = None
        self._dt_cached: float = 0.0

        self._build_cb_basis()

    # ------------------------------------------------------------------
    # CB basis construction
    # ------------------------------------------------------------------

    def _build_cb_basis(self) -> None:
        mesh = self.mesh
        n_nodes = mesh.n_nodes
        n_dof = n_nodes * 3

        print(f"[C-B '{self.name}'] Assembling K₀, M₀ ({n_dof} DOF)...")
        _edata = precompute_element_data(mesh)
        if isinstance(_edata, (ElementIntegrationData, CompositeIntegrationData)):
            dN_data = _edata
            volumes = _edata
        else:
            dN_data, volumes = _edata
        K0 = assemble_stiffness(mesh, mesh.nodes, self.material, dN_data, volumes)
        M0 = assemble_mass_matrix(mesh, self.density, volumes)

        # ── Fixed / free DOF split ───────────────────────────────────
        if self._anchored:
            fixed_dofs = np.sort(np.concatenate(
                [np.arange(n * 3, n * 3 + 3) for n in self.fixed_nodes]
            ))
        else:
            fixed_dofs = np.array([], dtype=np.int64)

        free_mask = np.ones(n_dof, dtype=bool)
        free_mask[fixed_dofs] = False
        free_dofs = np.where(free_mask)[0]

        # ── Boundary / interior partition within free DOFs ───────────
        surf_faces = mesh.extract_surface()
        surf_nodes = np.unique(surf_faces.ravel())
        all_nodes = np.arange(n_nodes)
        int_nodes = np.setdiff1d(all_nodes, surf_nodes)

        # Surface / interior nodes that are FREE (not fixed)
        b_nodes = np.setdiff1d(surf_nodes, self.fixed_nodes)
        i_nodes = np.setdiff1d(int_nodes, self.fixed_nodes)

        b_dofs_global = np.sort(np.concatenate(
            [np.arange(n * 3, n * 3 + 3) for n in b_nodes]
        )) if len(b_nodes) > 0 else np.array([], dtype=np.int64)
        i_dofs_global = np.sort(np.concatenate(
            [np.arange(n * 3, n * 3 + 3) for n in i_nodes]
        )) if len(i_nodes) > 0 else np.array([], dtype=np.int64)

        n_b = len(b_dofs_global)
        n_i = len(i_dofs_global)
        print(f"[C-B '{self.name}']   fixed: {len(fixed_dofs)}  "
              f"boundary: {n_b}  interior: {n_i}")

        # ── Interior block factorisation: dense vs sparse+ND ────────
        # Both the constraint-mode solve and the fixed-interface eigsh
        # shift-invert hit K_ii⁻¹. Strategy by interior size:
        #   • n_i < ND_THRESHOLD: dense LU. BLAS-3 GEMM dominates; sparse
        #     overhead is not worth it.
        #   • n_i ≥ ND_THRESHOLD: sparse with METIS-ND permutation. 3-D
        #     fill scales O(N · log N) instead of natural's O(N^{5/3}),
        #     the difference between routine and unusable above a few
        #     thousand DOFs.
        # Crossover measured around n_i ~ 3000 on M1 (dense ↔ ND).
        ND_THRESHOLD = 3000
        K_ii_sp = K0[np.ix_(i_dofs_global, i_dofs_global)].tocsc()
        nd_perm = None
        K_ii_factor = None
        if n_i > 0:
            if _HAS_PYMETIS and n_i >= ND_THRESHOLD:
                nd_perm = nested_dissection_order(K_ii_sp)
                K_ii_perm = K_ii_sp[nd_perm, :][:, nd_perm]
                K_ii_perm = K_ii_perm + sp.eye(n_i, format="csc") * 1e-12
                K_ii_factor = spla.splu(K_ii_perm, permc_spec="NATURAL")
            else:
                # Small interior or pymetis unavailable → dense fallback.
                K_ii_dense = K_ii_sp.toarray() + 1e-12 * np.eye(n_i)
                K_ii_factor = ("dense", lu_factor(K_ii_dense))

        def _solve_K_ii(rhs: np.ndarray) -> np.ndarray:
            """K_ii⁻¹ @ rhs, applying the ND permutation transparently."""
            if isinstance(K_ii_factor, tuple) and K_ii_factor[0] == "dense":
                return lu_solve(K_ii_factor[1], rhs)
            rhs_perm = rhs[nd_perm] if rhs.ndim == 1 else rhs[nd_perm, :]
            x_perm = K_ii_factor.solve(rhs_perm)
            x = np.empty_like(x_perm)
            if x_perm.ndim == 1:
                x[nd_perm] = x_perm
            else:
                x[nd_perm, :] = x_perm
            return x

        # ── Constraint modes  Ψ_c = -K_ii⁻¹ K_ib ───────────────────
        if n_i > 0 and n_b > 0:
            K_ib = K0[np.ix_(i_dofs_global, b_dofs_global)].toarray()
            print(f"[C-B '{self.name}']   constraint modes ({n_i}×{n_b})...")
            Psi_c = _solve_K_ii(-K_ib)
        else:
            Psi_c = np.zeros((n_i, n_b))

        # ── Fixed-interface normal modes ─────────────────────────────
        n_modes_req = min(self.n_modes, max(0, n_i - 1))
        Phi_k = np.zeros((n_i, 0))

        if n_modes_req > 0 and n_i > 0:
            print(f"[C-B '{self.name}']   {n_modes_req} fixed-interface modes...")
            M_ii = M0[np.ix_(i_dofs_global, i_dofs_global)]
            # Reuse the ND-permuted factor for shift-invert: eigsh's OPinv
            # replaces its own (slower, COLAMD-based) factorisation with
            # ours, so the reorder pays off twice.
            OPinv = spla.LinearOperator(
                K_ii_sp.shape, matvec=_solve_K_ii, dtype=np.float64,
            )
            eigenvalues, evecs = spla.eigsh(
                K_ii_sp, M=M_ii, k=n_modes_req, which="LM", sigma=0.0,
                OPinv=OPinv, tol=1e-10, maxiter=5000,
            )
            Phi_k = evecs[:, np.argsort(eigenvalues)]
        else:
            n_modes_req = 0

        # ── Φ_CB in full n_dof space ─────────────────────────────────
        n_r = n_b + n_modes_req
        Phi_CB = np.zeros((n_dof, n_r))

        if n_b > 0:
            Phi_CB[b_dofs_global, np.arange(n_b)] = 1.0
        if n_i > 0:
            Phi_CB[np.ix_(i_dofs_global, np.arange(n_b))] = Psi_c
            if n_modes_req > 0:
                Phi_CB[np.ix_(i_dofs_global, np.arange(n_b, n_r))] = Phi_k

        print(f"[C-B '{self.name}']   reducing: {n_dof} → {n_r} "
              f"({n_b} boundary + {n_modes_req} modes)...")

        # ── Reduced matrices on FREE DOFs ────────────────────────────
        # Use free-DOF subblock of K, M for correctness with fixed BCs
        if self._anchored and len(free_dofs) < n_dof:
            Phi_CB_free = Phi_CB[free_dofs, :]            # (n_free, n_r)
            K_free = K0[np.ix_(free_dofs, free_dofs)].toarray()
            M_free = M0[np.ix_(free_dofs, free_dofs)].toarray()
            K_r = Phi_CB_free.T @ K_free @ Phi_CB_free
            M_r = Phi_CB_free.T @ M_free @ Phi_CB_free
            M_diag_free = M0[np.ix_(free_dofs, free_dofs)].diagonal()
            PhiT_M = Phi_CB_free.T * M_diag_free[np.newaxis, :]
        else:
            K0_dense = K0.toarray()
            M0_dense = M0.toarray()
            K_r = Phi_CB.T @ K0_dense @ Phi_CB
            M_r = Phi_CB.T @ M0_dense @ Phi_CB
            M_diag_all = M0.diagonal()
            PhiT_M = Phi_CB.T * M_diag_all[np.newaxis, :]

        M_r_inv = np.linalg.inv(M_r)
        C_q = M_r_inv @ PhiT_M          # (n_r, n_free_or_full_dof)

        M_diag = M0.diagonal()
        m_total = np.sum([M_diag[n * 3] for n in range(n_nodes)])

        # ── Store ────────────────────────────────────────────────────
        self._M = M0          # sparse mass matrix — needed by FEM-FEM contact
        self._Phi_CB = Phi_CB
        self._K_r = K_r
        self._M_r = M_r
        self._C_q = C_q
        self._M_diag = M_diag
        self._m_total = m_total
        self._free_dofs = free_dofs
        self._fixed_dofs = fixed_dofs
        self._n_r = n_r
        self._n_b = n_b
        self._n_modes_actual = n_modes_req
        self._dN_list = dN_data
        self._volumes = volumes

        # Body-frame reference (centroid-centered)
        self._t_ref = mesh.nodes.mean(axis=0).copy()
        self._x_ref_body = mesh.nodes - self._t_ref

        print(f"[C-B '{self.name}']   ready.  m_total={m_total:.4f} kg  "
              f"{'anchored' if self._anchored else 'free-floating'}")

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    def initialize(self, dt: float) -> None:
        """Set initial state and LU-factorise the constant system matrix."""
        n_nodes = self.mesh.n_nodes
        self.x = self.mesh.nodes.copy()
        self.v = np.zeros((n_nodes, 3))

        dt2_inv = 1.0 / (dt * dt)
        A_r = self._M_r * dt2_inv + self._K_r
        if self.damping > 0.0:
            A_r = A_r + self.damping * self._M_r / dt

        self._A_r_inv = np.linalg.inv(A_r)   # dense DGEMV per step, faster than lu_solve
        self._dt_cached = dt
        print(f"[C-B '{self.name}']   initialized  dt={dt}  "
              f"A_r ({self._n_r}×{self._n_r}) inverted.")

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float | None = None,
        extra_forces: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Advance one implicit-Euler step.

        Reads ``self.x``, ``self.v`` as current state (external modifications
        such as impulse-based contact or kinematic coupling are absorbed
        automatically) and writes back updated state.
        """
        if dt is None:
            dt = self._dt_cached
        elif dt != self._dt_cached:
            self.initialize(dt)

        if self._anchored:
            self._step_anchored(dt, extra_forces)
        else:
            self._step_free(dt, extra_forces)

    # ------------------------------------------------------------------
    # Free-floating step  (corotational R via Kabsch)
    # ------------------------------------------------------------------

    def _step_free(self, dt: float, extra_forces) -> None:
        # ── C++ dispatch (default on; opt-out via _use_cpp_cb=False). ─
        # The earlier port's free-body NaN drift came from missing the
        # mass-weighted projection (``self._C_q``) — using bare Phiᵀ
        # silently drops the diag(M) weighting and lets the body
        # diverge after a few gravity steps. Fixed by passing C_q
        # through; parity is now ~1e-17 over 20 steps under gravity.
        if getattr(self, "_use_cpp_cb", True) and _try_cpp_step_free(self, dt, extra_forces):
            return

        n_nodes = self.mesh.n_nodes
        n_dof = n_nodes * 3
        dt2_inv = 1.0 / (dt * dt)

        # Centroid + Kabsch rotation from current x
        t = self.x.mean(axis=0)
        x_centered = self.x - t
        R = _kabsch_rotation(x_centered, self._x_ref_body)

        # Body-frame deformation: u = x_centered @ R - x_ref_body
        u_body = x_centered @ R - self._x_ref_body    # (n_nodes, 3)
        q_r = self._C_q @ u_body.reshape(-1)

        t_dot = self.v.mean(axis=0)
        v_body = (self.v - t_dot) @ R
        q_r_dot = self._C_q @ v_body.reshape(-1)

        # External forces (world frame)
        f_ext = np.zeros(n_dof)
        for d in range(3):
            f_ext[d::3] += self._M_diag[d::3] * self.gravity[d]
        if extra_forces is not None and 0 in extra_forces:
            f_ext += extra_forces[0]

        # Body-frame force → reduced
        f_r = self._Phi_CB.T @ (f_ext.reshape(n_nodes, 3) @ R).reshape(-1)

        # Implicit Euler
        q_r_pred = q_r + dt * q_r_dot
        rhs = self._M_r @ q_r_pred * dt2_inv + f_r
        if self.damping > 0.0:
            rhs += self.damping * self._M_r @ q_r / dt
        q_r_new = self._A_r_inv @ rhs
        q_r_dot_new = (q_r_new - q_r) / dt

        # Centroid dynamics (explicit)
        f_total = f_ext.reshape(n_nodes, 3).sum(axis=0)
        t_dot_new = t_dot + dt * f_total / self._m_total
        t_new = t + dt * t_dot_new

        # Reconstruct
        u_new = (self._Phi_CB @ q_r_new).reshape(n_nodes, 3)
        self.x = t_new + (self._x_ref_body + u_new) @ R.T
        u_dot_new = (self._Phi_CB @ q_r_dot_new).reshape(n_nodes, 3)
        self.v = t_dot_new + u_dot_new @ R.T
        self.q_r = q_r_new   # store for external access (e.g. CBBodyHandle.modal_coords)

    # ------------------------------------------------------------------
    # Anchored step  (fixed nodes clamped, R = I, no centroid motion)
    # ------------------------------------------------------------------

    def _step_anchored(self, dt: float, extra_forces) -> None:
        # ── C++ dispatch (default on; opt-out via _use_cpp_cb=False). ─
        # HybridCBPlasticBody rebuilds the CB basis at runtime — the
        # bridge invalidates ``_cpp_cb_ready`` on basis change, so the
        # C++ cache regenerates correctly. Per-body opt-out is still
        # supported for parity testing.
        if getattr(self, "_use_cpp_cb", True) and _try_cpp_step_anchored(self, dt, extra_forces):
            return

        n_nodes = self.mesh.n_nodes
        n_dof = n_nodes * 3
        dt2_inv = 1.0 / (dt * dt)

        free_dofs = self._free_dofs    # global DOF indices that are free

        # Displacement of free DOFs from reference
        x_flat = self.x.reshape(-1)
        x_ref_flat = self.mesh.nodes.reshape(-1)
        u_free = x_flat[free_dofs] - x_ref_flat[free_dofs]   # (n_free,)
        q_r = self._C_q @ u_free

        v_flat = self.v.reshape(-1)
        q_r_dot = self._C_q @ v_flat[free_dofs]

        # External forces on free DOFs
        f_ext = np.zeros(n_dof)
        for d in range(3):
            f_ext[d::3] += self._M_diag[d::3] * self.gravity[d]
        if extra_forces is not None and 0 in extra_forces:
            f_ext += extra_forces[0]
        f_ext_free = f_ext[free_dofs]    # (n_free,)

        # Reduced force (no R transform since R=I)
        f_r = self._Phi_CB[free_dofs, :].T @ f_ext_free

        # Implicit Euler
        q_r_pred = q_r + dt * q_r_dot
        rhs = self._M_r @ q_r_pred * dt2_inv + f_r
        if self.damping > 0.0:
            rhs += self.damping * self._M_r @ q_r / dt
        q_r_new = self._A_r_inv @ rhs
        q_r_dot_new = (q_r_new - q_r) / dt

        # Reconstruct: only free DOFs change
        u_new = (self._Phi_CB @ q_r_new).reshape(n_nodes, 3)
        x_new = self.mesh.nodes + u_new
        # Re-clamp fixed nodes exactly at reference
        x_new.reshape(-1)[self._fixed_dofs] = x_ref_flat[self._fixed_dofs]

        u_dot_new = (self._Phi_CB @ q_r_dot_new).reshape(n_nodes, 3)
        v_new = u_dot_new
        v_new.reshape(-1)[self._fixed_dofs] = 0.0

        self.x = x_new
        self.v = v_new
        self.q_r = q_r_new   # store for external access (e.g. CBBodyHandle.modal_coords)


# ═══════════════════════════════════════════════════════════════
# Craig-Bampton Solver  (FEMSolver-compatible wrapper)
# ═══════════════════════════════════════════════════════════════

class CraigBamptonSolver:
    """Thin wrapper that gives ``CraigBamptonBody`` the same interface as
    ``FEMSolver``.

    .. code-block:: python

        body   = CraigBamptonBody(mesh, material, density=1000,
                                  fixed_nodes=fixed_nodes, n_modes=10)
        solver = CraigBamptonSolver(bodies=[body])
        solver.initialize(dt=0.005)
        solver.step()          # same API as FEMSolver
    """

    def __init__(
        self,
        bodies: list[CraigBamptonBody],
        gravity: np.ndarray | None = None,
        damping: float | None = None,
    ) -> None:
        self.bodies = bodies
        if gravity is not None:
            for b in bodies:
                b.gravity = np.asarray(gravity, dtype=np.float64)
        if damping is not None:
            for b in bodies:
                b.damping = float(damping)
        self._time: float = 0.0
        self._dt: float = 0.0

    def initialize(self, dt: float) -> None:
        for body in self.bodies:
            body.initialize(dt)
        self._dt = dt
        self._time = 0.0

    def step(
        self,
        dt: float | None = None,
        extra_forces: dict[int, np.ndarray] | None = None,
    ) -> None:
        if dt is None:
            dt = self._dt
        for idx, body in enumerate(self.bodies):
            ef = ({0: extra_forces[idx]} if extra_forces and idx in extra_forces
                  else None)
            body.step(dt=dt, extra_forces=ef)
        self._time += dt

    @property
    def time(self) -> float:
        return self._time


# ═══════════════════════════════════════════════════════════════
# Kabsch best-fit rotation
# ═══════════════════════════════════════════════════════════════

def _kabsch_rotation(x_centered: np.ndarray, x_ref_body: np.ndarray) -> np.ndarray:
    """Return R (3×3) minimising ||x_centered − x_ref_body @ R^T||²_F.

    At reference state (x_centered == x_ref_body) returns identity.
    """
    H = x_ref_body.T @ x_centered
    U, _S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R
