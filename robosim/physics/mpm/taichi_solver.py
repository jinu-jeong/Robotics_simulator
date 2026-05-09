"""Taichi-accelerated MLS-MPM solver.

Mirrors :class:`robosim.physics.mpm.solver.MPMSolver` but pushes the
hot loops (P2G, G2P, gravity, BC, F update, advection, plastic
projection) into Taichi kernels — typically a 10-50× speedup on M1
Metal vs the NumPy reference.

Materials supported (selected via ``material_kind`` at construction):
* ``"neo"``      — elastic NeoHookean (default).
* ``"vm"``       — NeoHookean elasticity + Von-Mises J2 plasticity.
* ``"dp"``       — NeoHookean elasticity + Drucker-Prager plasticity.

Out-of-scope for now (use NumPy :class:`MPMSolver` instead):
* Damage models (CD-MPM ``DamagedNeoHookean``).
* Kinematic colliders (``KinematicBoxCollider``).

Activation: caller must call ``ti.init(arch=ti.metal)`` (or another arch)
before instantiating. The class registers Taichi fields with the
currently active runtime, so creating two solvers in the same Python
process requires the same arch.
"""

# NOTE: do NOT use ``from __future__ import annotations`` here.
# Taichi 1.7's @ti.kernel argument-extractor needs the *actual* type
# object (e.g. ti.f32) at decoration time. PEP 563 lazy annotations
# turn the annotation into a string, breaking that.

from typing import Optional, Sequence

import numpy as np

try:
    import taichi as ti
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "Taichi MPM solver requires the 'taichi' package. "
        "Install with `pip install taichi`."
    ) from _exc


@ti.data_oriented
class TaichiMPMSolver:
    """Taichi MLS-MPM solver. API mirrors :class:`MPMSolver`."""

    # Material kind constants (used as ti.static keys in kernel branches).
    MAT_NEO = 0
    MAT_VM  = 1
    MAT_DP  = 2

    def __init__(
        self,
        particles_x: np.ndarray,
        particles_v: np.ndarray,
        particles_m: np.ndarray,
        particles_V0: np.ndarray,
        grid_origin: Sequence[float],
        grid_dx: float,
        grid_shape: tuple[int, int, int],
        young: float,
        poisson: float,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        bc_lower: Optional[Sequence[float]] = None,
        bc_upper: Optional[Sequence[float]] = None,
        material_kind: str = "neo",
        yield_stress: float = 1e4,
        friction_angle: float = 0.5236,   # 30 degrees
    ):
        kind_map = {"neo": self.MAT_NEO, "vm": self.MAT_VM, "dp": self.MAT_DP}
        if material_kind not in kind_map:
            raise ValueError(
                f"material_kind must be one of {list(kind_map)}, "
                f"got {material_kind!r}"
            )
        self.material_kind = material_kind
        self._mat_id = kind_map[material_kind]
        self.yield_stress = float(yield_stress)
        # Drucker-Prager slope α = √(2/3) · 2 sinφ / (3 − sinφ)
        s = float(np.sin(friction_angle))
        self.dp_alpha = float(np.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s))
        self.P = int(particles_x.shape[0])
        nx, ny, nz = grid_shape
        self.shape = (nx, ny, nz)
        self.dx  = float(grid_dx)
        self.inv_dx = 1.0 / self.dx
        self.D_inv = 4.0 / (self.dx * self.dx)

        # Lamé constants
        self.mu  = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        # Boundary box (slip). Defaults: domain extents for the grid.
        if bc_lower is None:
            bc_lower = grid_origin
        if bc_upper is None:
            bc_upper = (
                grid_origin[0] + (nx - 1) * self.dx,
                grid_origin[1] + (ny - 1) * self.dx,
                grid_origin[2] + (nz - 1) * self.dx,
            )

        # ── Taichi fields ─────────────────────────────────────────────
        self.x  = ti.Vector.field(3, dtype=ti.f32, shape=self.P)
        self.v  = ti.Vector.field(3, dtype=ti.f32, shape=self.P)
        self.F  = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.P)
        self.C_ = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.P)
        self.m_p = ti.field(dtype=ti.f32, shape=self.P)
        self.V0  = ti.field(dtype=ti.f32, shape=self.P)

        self.grid_m = ti.field(dtype=ti.f32, shape=self.shape)
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=self.shape)

        self.origin  = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.gravity = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.bc_lo   = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.bc_hi   = ti.Vector.field(3, dtype=ti.f32, shape=())

        # ── Initialise from numpy ─────────────────────────────────────
        self.x.from_numpy(particles_x.astype(np.float32))
        self.v.from_numpy(particles_v.astype(np.float32))
        self.m_p.from_numpy(particles_m.astype(np.float32))
        self.V0.from_numpy(particles_V0.astype(np.float32))
        F0 = np.broadcast_to(np.eye(3, dtype=np.float32), (self.P, 3, 3)).copy()
        self.F.from_numpy(F0)
        # C and grid fields default to 0 — Taichi zero-initialises.

        self.origin.from_numpy(np.asarray(grid_origin, dtype=np.float32))
        self.gravity.from_numpy(np.asarray(gravity, dtype=np.float32))
        self.bc_lo.from_numpy(np.asarray(bc_lower, dtype=np.float32))
        self.bc_hi.from_numpy(np.asarray(bc_upper, dtype=np.float32))

    # ── kernels ───────────────────────────────────────────────────────
    @ti.kernel
    def _reset_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0.0
            self.grid_v[I] = ti.Vector.zero(ti.f32, 3)

    @ti.kernel
    def _p2g_with_stress(self, dt: ti.f32):
        for p in range(self.P):
            xi  = (self.x[p] - self.origin[None]) * self.inv_dx
            base = ti.cast(ti.floor(xi - 0.5), ti.i32)
            u = xi - ti.cast(base, ti.f32)
            # Quadratic B-spline weights along each axis
            w0 = 0.5 * (1.5 - u) ** 2
            w1 = 0.75 - (u - 1.0) ** 2
            w2 = 0.5 * (u - 0.5) ** 2
            wx = ti.Vector([w0[0], w1[0], w2[0]])
            wy = ti.Vector([w0[1], w1[1], w2[1]])
            wz = ti.Vector([w0[2], w1[2], w2[2]])

            # Kirchhoff τ = μ(F F^T − I) + λ log(J) I  (NeoHookean)
            Fp = self.F[p]
            J  = Fp.determinant()
            tau = (self.mu * (Fp @ Fp.transpose() - ti.Matrix.identity(ti.f32, 3))
                   + self.lam * ti.log(J) * ti.Matrix.identity(ti.f32, 3))

            # APIC affine: A = m·C − dt · (4/dx²) · V₀ · τ
            affine = self.m_p[p] * self.C_[p] - dt * self.D_inv * self.V0[p] * tau
            mp_vp  = self.m_p[p] * self.v[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node   = base + offset
                W      = wx[i] * wy[j] * wz[k]
                x_node = self.origin[None] + ti.cast(node, ti.f32) * self.dx
                dpos   = x_node - self.x[p]
                self.grid_m[node] += W * self.m_p[p]
                self.grid_v[node] += W * (mp_vp + affine @ dpos)

    @ti.kernel
    def _normalise_grav_bc(self, dt: ti.f32):
        gv = self.gravity[None]
        lo = self.bc_lo[None]
        hi = self.bc_hi[None]
        for I in ti.grouped(self.grid_m):
            m = self.grid_m[I]
            if m > 0.0:
                v = self.grid_v[I] / m + dt * gv
                # Slip BC: clamp inward normal at each wall.
                pos = self.origin[None] + ti.cast(I, ti.f32) * self.dx
                if pos[0] <= lo[0] and v[0] < 0.0: v[0] = 0.0
                if pos[0] >= hi[0] and v[0] > 0.0: v[0] = 0.0
                if pos[1] <= lo[1] and v[1] < 0.0: v[1] = 0.0
                if pos[1] >= hi[1] and v[1] > 0.0: v[1] = 0.0
                if pos[2] <= lo[2] and v[2] < 0.0: v[2] = 0.0
                if pos[2] >= hi[2] and v[2] > 0.0: v[2] = 0.0
                self.grid_v[I] = v

    @ti.kernel
    def _g2p(self):
        for p in range(self.P):
            xi  = (self.x[p] - self.origin[None]) * self.inv_dx
            base = ti.cast(ti.floor(xi - 0.5), ti.i32)
            u = xi - ti.cast(base, ti.f32)
            w0 = 0.5 * (1.5 - u) ** 2
            w1 = 0.75 - (u - 1.0) ** 2
            w2 = 0.5 * (u - 0.5) ** 2
            wx = ti.Vector([w0[0], w1[0], w2[0]])
            wy = ti.Vector([w0[1], w1[1], w2[1]])
            wz = ti.Vector([w0[2], w1[2], w2[2]])

            v_new = ti.Vector.zero(ti.f32, 3)
            C_new = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node   = base + offset
                W      = wx[i] * wy[j] * wz[k]
                v_i    = self.grid_v[node]
                x_node = self.origin[None] + ti.cast(node, ti.f32) * self.dx
                dpos   = x_node - self.x[p]
                v_new += W * v_i
                C_new += W * v_i.outer_product(dpos)
            self.v[p]  = v_new
            self.C_[p] = self.D_inv * C_new

    @ti.func
    def _project_vm(self, F_trial):
        """Von-Mises radial return on Hencky principal strain. Volumetric
        part untouched (isochoric flow), deviatoric scaled to yield."""
        U, S, V = ti.svd(F_trial, ti.f32)
        eps = ti.Vector([
            ti.log(ti.max(S[0, 0], 1e-12)),
            ti.log(ti.max(S[1, 1], 1e-12)),
            ti.log(ti.max(S[2, 2], 1e-12)),
        ])
        eps_mean = (eps[0] + eps[1] + eps[2]) / 3.0
        eps_dev = eps - ti.Vector([eps_mean, eps_mean, eps_mean])
        norm_dev = eps_dev.norm()
        yield_strain = self.yield_stress / (ti.sqrt(6.0) * self.mu)
        scale = 1.0
        if norm_dev > yield_strain:
            scale = yield_strain / ti.max(norm_dev, 1e-16)
        eps_new = eps_dev * scale + ti.Vector([eps_mean, eps_mean, eps_mean])
        sigma_new = ti.Matrix([
            [ti.exp(eps_new[0]), 0.0, 0.0],
            [0.0, ti.exp(eps_new[1]), 0.0],
            [0.0, 0.0, ti.exp(eps_new[2])],
        ])
        return U @ sigma_new @ V.transpose()

    @ti.func
    def _project_dp(self, F_trial):
        """Drucker-Prager 3-case return on Hencky principal strain.
        Tension → ε=0, cone-violation → radial return, elastic → unchanged."""
        U, S, V = ti.svd(F_trial, ti.f32)
        eps = ti.Vector([
            ti.log(ti.max(S[0, 0], 1e-12)),
            ti.log(ti.max(S[1, 1], 1e-12)),
            ti.log(ti.max(S[2, 2], 1e-12)),
        ])
        trace = eps[0] + eps[1] + eps[2]
        eps_hat = eps - ti.Vector([trace, trace, trace]) / 3.0
        norm_hat = eps_hat.norm()
        coef = (3.0 * self.lam + 2.0 * self.mu) / (2.0 * self.mu)
        dgamma = norm_hat + coef * trace * self.dp_alpha
        eps_new = eps
        if trace > 0.0:
            eps_new = ti.Vector([0.0, 0.0, 0.0])
        elif dgamma > 0.0:
            safe_norm = ti.max(norm_hat, 1e-16)
            eps_new = eps - (dgamma / safe_norm) * eps_hat
        sigma_new = ti.Matrix([
            [ti.exp(eps_new[0]), 0.0, 0.0],
            [0.0, ti.exp(eps_new[1]), 0.0],
            [0.0, 0.0, ti.exp(eps_new[2])],
        ])
        return U @ sigma_new @ V.transpose()

    @ti.kernel
    def _update_F_advect(self, dt: ti.f32):
        for p in range(self.P):
            F_trial = (ti.Matrix.identity(ti.f32, 3) + dt * self.C_[p]) @ self.F[p]
            if ti.static(self._mat_id == self.MAT_NEO):
                self.F[p] = F_trial
            elif ti.static(self._mat_id == self.MAT_VM):
                self.F[p] = self._project_vm(F_trial)
            elif ti.static(self._mat_id == self.MAT_DP):
                self.F[p] = self._project_dp(F_trial)
            self.x[p] = self.x[p] + dt * self.v[p]

    # ── public step ───────────────────────────────────────────────────
    def step(self, dt: ti.f32) -> None:
        dt32 = float(dt)
        self._reset_grid()
        self._p2g_with_stress(dt32)
        self._normalise_grav_bc(dt32)
        self._g2p()
        self._update_F_advect(dt32)

    # ── numpy bridges (read-only on hot path; copy out for diagnostics) ──
    def particles_x(self) -> np.ndarray:
        return self.x.to_numpy()

    def particles_v(self) -> np.ndarray:
        return self.v.to_numpy()

    def particles_F(self) -> np.ndarray:
        return self.F.to_numpy()

    # ── MPMSolver-compatible facade (so demos can swap) ──
    @property
    def particles(self) -> "_TiParticleView":
        """Live particle view exposing ``.x / .v / .m / .V0 / .F / .C`` like
        :class:`Particles`. Each access copies from Taichi fields back to
        NumPy — fine for diagnostics (per-frame), avoid in hot loops."""
        return _TiParticleView(self)

    @property
    def material(self) -> "_TiMaterialView":
        """Stub material exposing ``.mu / .lam`` so energy diagnostics that
        read ``solver.material.mu`` work identically to the NumPy path."""
        return _TiMaterialView(self.mu, self.lam)

    @classmethod
    def from_numpy_setup(
        cls,
        *,
        particles,
        grid,
        material,
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        bcs: Optional[Sequence] = None,
    ) -> "TaichiMPMSolver":
        """High-level constructor mirroring :class:`MPMSolver`.

        Caller must have called ``ti.init(arch=...)`` before this. Only the
        first ``BoxBC`` in ``bcs`` is honoured (matches MVP scope);
        colliders are not supported yet.
        """
        if not hasattr(material, "young"):
            raise TypeError(
                f"TaichiMPMSolver requires a NeoHookean-derived material "
                f"(with .young/.poisson); got {type(material).__name__}"
            )
        cls_name = type(material).__name__
        kind = "neo"
        yield_stress = 1e4
        friction_angle = 0.5236
        if cls_name == "VonMisesPlastic":
            kind = "vm"
            yield_stress = float(getattr(material, "yield_stress", 1e4))
        elif cls_name == "DruckerPragerPlastic":
            kind = "dp"
            friction_angle = float(getattr(material, "friction_angle", 0.5236))
        elif cls_name == "DamagedNeoHookean":
            raise NotImplementedError(
                "DamagedNeoHookean (CD-MPM) not yet supported by Taichi backend"
            )
        bc_lower = None
        bc_upper = None
        if bcs:
            bc = bcs[0]
            bc_lower = bc.lower
            bc_upper = bc.upper
        return cls(
            particles_x=particles.x.copy(),
            particles_v=particles.v.copy(),
            particles_m=particles.m.copy(),
            particles_V0=particles.V0.copy(),
            grid_origin=grid.origin,
            grid_dx=grid.dx,
            grid_shape=grid.shape,
            young=float(material.young),
            poisson=float(material.poisson),
            gravity=tuple(gravity),
            bc_lower=bc_lower,
            bc_upper=bc_upper,
            material_kind=kind,
            yield_stress=yield_stress,
            friction_angle=friction_angle,
        )


class _TiParticleView:
    """``.x / .v / .m / .V0 / .F / .C / .n`` getters reading from Taichi fields."""

    __slots__ = ("_sv",)

    def __init__(self, sv: "TaichiMPMSolver") -> None:
        self._sv = sv

    @property
    def x(self) -> np.ndarray: return self._sv.x.to_numpy()
    @property
    def v(self) -> np.ndarray: return self._sv.v.to_numpy()
    @property
    def m(self) -> np.ndarray: return self._sv.m_p.to_numpy()
    @property
    def V0(self) -> np.ndarray: return self._sv.V0.to_numpy()
    @property
    def F(self) -> np.ndarray: return self._sv.F.to_numpy()
    @property
    def C(self) -> np.ndarray: return self._sv.C_.to_numpy()
    @property
    def n(self) -> int: return int(self._sv.P)


class _TiMaterialView:
    """``.mu / .lam / .young / .poisson`` for energy-diagnostic compatibility."""

    __slots__ = ("mu", "lam")

    def __init__(self, mu: float, lam: float) -> None:
        self.mu = mu
        self.lam = lam
