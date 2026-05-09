"""Hybrid CB / full-FEM body with state-machine-driven plasticity routing.

Idea (cf. Phase 2 design discussion):

Pre-partition the mesh into K METIS regions. Track each region's state:

    ELASTIC          — region's interior is reduced (CB modes); cheap path.
    PLASTIC_ACTIVE   — region exceeded yield: switch its interior to full-FEM
                        and run the J2 return mapping there. Other regions
                        keep their current state.
    REBUILD_PENDING  — region has been in PLASTIC_ACTIVE but max σ_eq has
                        sat below ``σ_Y · (1 - hysteresis)`` for several
                        steps in a row. Once *all* regions are in this
                        state, the ROM basis is rebuilt at the new
                        deformed shape (absorbing permanent set into the
                        new reference configuration) and everything goes
                        back to ELASTIC.

Phase 2B (this commit): the *skeleton* — body class with the partition,
the state field, and a delegated ``step()`` that, while every region is
ELASTIC, just calls the underlying CraigBamptonBody. PLASTIC_ACTIVE /
REBUILD branches come in subsequent commits.

The composition pattern (HybridCBPlasticBody owns a CraigBamptonBody and
will later own a DeformableBody) was chosen over inheritance so the
hybrid logic stays orthogonal to either backend's internals.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from robosim.physics.fem.assembly import (
    _batch_deformation_gradients, _batch_polar_decomposition,
)
from robosim.physics.fem.materials import (
    CorotationalElastic, CorotationalPlastic, NeoHookean,
)
from robosim.physics.fem.mesh import FEMesh, TetMesh
from robosim.physics.fem.integrator import implicit_euler_step
from robosim.physics.fem.partition import RegionPartition, build_region_partition
from robosim.physics.fem.reduced import CraigBamptonBody


class RegionState(Enum):
    ELASTIC = "elastic"
    PLASTIC_ACTIVE = "plastic_active"
    REBUILD_PENDING = "rebuild_pending"


class HybridCBPlasticBody:
    """Region-partitioned body that switches per region between CB
    reduction and full-FEM plasticity.

    The constructor accepts the same elastic-side arguments as
    :class:`CraigBamptonBody` plus plasticity parameters and the region
    count ``n_regions``. With ``n_regions=1`` (the default) the body is
    behaviourally identical to the underlying CraigBamptonBody — useful
    as a regression baseline while the multi-region path is built up.
    """

    def __init__(
        self,
        mesh: TetMesh | FEMesh,
        material: Optional[CorotationalElastic | NeoHookean] = None,
        density: float = 1000.0,
        n_modes: int = 6,
        fixed_nodes: Optional[np.ndarray] = None,
        gravity: Optional[np.ndarray] = None,
        damping: float = 0.01,
        name: str = "hybrid_body",
        # Plasticity:
        yield_stress: float = 1e4,
        hardening: float = 0.0,
        # Partition / state machine:
        n_regions: int = 1,
        hysteresis: float = 0.05,
        rebuild_after_steady_steps: int = 50,
    ) -> None:
        if material is None:
            material = CorotationalElastic()
        # Plasticity material (used only when a region transitions to
        # PLASTIC_ACTIVE; reduced path uses the elastic ``material`` arg).
        self._plastic_material = CorotationalPlastic(
            young=getattr(material, "young", 1e6),
            poisson=getattr(material, "poisson", 0.3),
            yield_stress=yield_stress,
            hardening=hardening,
        )

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

        # State-machine knobs.
        self.yield_stress = float(yield_stress)
        self.hardening = float(hardening)
        self.hysteresis = float(hysteresis)
        self.rebuild_after_steady_steps = int(rebuild_after_steady_steps)

        # Region partition: built once at construction. Re-builds happen
        # only after a coordinated REBUILD across all regions (Phase 2F).
        self.partition: RegionPartition = build_region_partition(
            mesh, n_regions=n_regions,
        )

        # Per-region state, initialised ELASTIC.
        self.region_state: list[RegionState] = [
            RegionState.ELASTIC for _ in range(self.partition.n_regions)
        ]
        # Counter used to debounce REBUILD: increments each step a region
        # sits below the yield threshold while in PLASTIC_ACTIVE; resets
        # whenever the region re-yields.
        self._rebuild_counter: list[int] = [
            0 for _ in range(self.partition.n_regions)
        ]

        # Plastic strain storage: per-element (3, 3). Allocated lazily on
        # first transition to PLASTIC_ACTIVE; for now a zero buffer so
        # diagnostics can read it unconditionally.
        self.eps_p = np.zeros((mesh.n_elements, 3, 3), dtype=np.float64)
        # Peak-nodal-speed gate on the rebuild trigger. Below this the
        # body is treated as quasi-static (no transient elastic
        # deformation to misclassify as permanent set). 1 mm/s is the
        # bench-scale "really at rest" floor for the demos here; well
        # above float noise but tight enough that a slow-settling
        # cantilever doesn't fire the rebuild while it's still
        # rebounding past its plastic-equilibrium tip position.
        self._quasistatic_v_threshold = 1e-3
        # Snapshot from the previous ``update_region_states`` call —
        # compared against ``self.eps_p`` to decide whether plastic flow
        # actually happened in the last step (state-machine trigger).
        self._eps_p_prev_update = np.zeros_like(self.eps_p)

        # Underlying CB body — handles the all-ELASTIC fast path. The
        # plastic-active full-FEM step shares this body's mesh-level
        # caches (M, dN_list, volumes) and reads/writes its x and v so
        # state stays consistent across mode switches.
        self._cb = CraigBamptonBody(
            mesh=mesh, material=material, density=density,
            n_modes=n_modes, fixed_nodes=fixed_nodes,
            gravity=gravity, damping=damping, name=name + "_cb",
        )

        # Cached fixed-DOF index array for the full-FEM path.
        if len(self.fixed_nodes) > 0:
            self._fixed_dofs = np.sort(np.concatenate([
                np.arange(int(n) * 3, int(n) * 3 + 3) for n in self.fixed_nodes
            ])).astype(np.int64)
        else:
            self._fixed_dofs = np.array([], dtype=np.int64)
        # Newton iters for the plastic implicit Euler step. Fewer than
        # the elastic baseline because the modified-Newton tangent gives
        # up quadratic convergence inside the plastic zone — we trade
        # tighter convergence for a couple of cheap extra iterations.
        self._plastic_max_newton = 8

    # ── Convenience pass-throughs ─────────────────────────────────────

    @property
    def x(self) -> np.ndarray | None:
        return self._cb.x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self._cb.x = value

    @property
    def v(self) -> np.ndarray | None:
        return self._cb.v

    @v.setter
    def v(self, value: np.ndarray) -> None:
        self._cb.v = value

    @property
    def n_regions(self) -> int:
        return self.partition.n_regions

    # ── Pass-throughs to the inner CB body for diagnostics that target
    # FEM-style deformable bodies (von Mises colouring, etc.) ──────────
    @property
    def _dN_list(self):
        return self._cb._dN_list

    @property
    def _volumes(self):
        return self._cb._volumes

    def all_elastic(self) -> bool:
        return all(s == RegionState.ELASTIC for s in self.region_state)

    def initialize(self, dt: float) -> None:
        """Build the CB basis (delegates to CraigBamptonBody)."""
        self._cb.initialize(dt=dt)
        self._cached_dt = float(dt)
        # Counter of how many ROM rebuilds have happened — useful for
        # demos / tests that want to confirm the rebuild fired.
        self._rebuild_count = 0

    # ── Stress diagnostics (drive the per-region state machine) ───────

    def per_element_sigma_eq(self) -> np.ndarray:
        """Per-element von-Mises equivalent stress at the current state.

        Built from the trial-elastic stress in the corotated frame, with
        the *current* eps_p already subtracted — i.e. matches the
        quantity the J2 yield function compares against ``σ_Y``. Length-
        ``n_elements`` float64 array. Available regardless of which
        branch (ELASTIC vs PLASTIC_ACTIVE) just ran; the caller is
        responsible for invoking it after a step rather than before.
        """
        if self._cb.x is None:
            raise RuntimeError(
                "per_element_sigma_eq called before initialize() — x is None"
            )
        if self._cb._dN_list is None or self._cb._volumes is None:
            raise RuntimeError(
                "per_element_sigma_eq called before CB basis is built"
            )

        # Tet4 path is the only one wired in Phase 1; matches the plastic
        # assembly path's element-data convention (dN_list = (ne, 4, 3)).
        if not isinstance(self._cb._dN_list, np.ndarray):
            raise NotImplementedError(
                "per_element_sigma_eq currently supports Tet4 meshes only"
            )

        F_all = _batch_deformation_gradients(
            self._cb.x, self.mesh.elements, self._cb._dN_list,
        )
        R_all, S_all = _batch_polar_decomposition(F_all)
        I3 = np.eye(3)[np.newaxis]
        # Trial elastic strain in corotated frame minus accumulated eps_p.
        eps_e = (
            0.5 * (S_all + np.transpose(S_all, (0, 2, 1)))
            - I3 - self.eps_p
        )
        tr_e = np.trace(eps_e, axis1=1, axis2=2)
        mu = self._plastic_material.mu
        lam = self._plastic_material.lam
        sigma = 2.0 * mu * eps_e + lam * tr_e[:, None, None] * I3
        # von Mises: σ_eq = sqrt(3/2) · ‖dev(σ)‖_F
        tr_s = np.trace(sigma, axis1=1, axis2=2)
        dev = sigma - (tr_s / 3.0)[:, None, None] * I3
        norm_dev = np.linalg.norm(dev, axis=(1, 2))
        return np.sqrt(1.5) * norm_dev

    def update_region_states(self) -> bool:
        """Run one tick of the per-region state machine.

        Triggers:
        * ELASTIC → PLASTIC_ACTIVE   when ``max σ_eq ≥ σ_Y`` somewhere
            in the region. (σ_eq exceeding σ_Y in trial-elastic stress
            means the elastic predictor is outside the yield surface,
            i.e. the next step will yield.)
        * PLASTIC_ACTIVE → REBUILD_PENDING when *no plastic flow*
            actually happened in the region for
            ``rebuild_after_steady_steps`` consecutive update calls.
            Plastic-flow detection: the per-element Frobenius norm of
            ``eps_p_now − eps_p_at_last_update`` falls below a small
            threshold. This is the right signal because, with hardening,
            σ_eq at rest stays *above* σ_Y (the yield surface has
            expanded) — so a σ_eq comparison would falsely keep regions
            "active" forever.
        * REBUILD_PENDING → PLASTIC_ACTIVE if any region element starts
            flowing again before the global rebuild fires.

        REBUILD_PENDING is a holding state. The actual ROM rebuild
        (Phase 2F) fires only when all regions are in this state.

        Returns ``True`` iff any region is PLASTIC_ACTIVE *after* this
        update — the caller uses this to pick dispatch on the next step.
        """
        region_max = self.per_region_max_sigma_eq()
        # Per-element ‖Δeps_p‖ over the last update window.
        delta_eps_p = np.linalg.norm(
            self.eps_p - self._eps_p_prev_update, axis=(1, 2),
        )
        # Threshold scaled to the yield strain so it tracks the model
        # naturally; below this, any change is float noise. ``5 % of
        # yield strain`` is a generous floor that ignores numerical
        # jitter without missing genuine plastic flow.
        flow_threshold = 0.05 * (self.yield_stress / max(self._plastic_material.young, 1e-12))

        any_active = False
        for r in range(self.n_regions):
            state = self.region_state[r]
            elems_r = self.partition.region_elements[r]
            sm = float(region_max[r])
            max_dflow = float(delta_eps_p[elems_r].max()) if elems_r.size > 0 else 0.0

            if state == RegionState.ELASTIC:
                if sm >= self.yield_stress:
                    self.region_state[r] = RegionState.PLASTIC_ACTIVE
                    self._rebuild_counter[r] = 0
                    any_active = True

            elif state == RegionState.PLASTIC_ACTIVE:
                if max_dflow > flow_threshold:
                    # Still flowing.
                    self._rebuild_counter[r] = 0
                    any_active = True
                else:
                    # No detectable plastic flow this step.
                    self._rebuild_counter[r] += 1
                    if self._rebuild_counter[r] >= self.rebuild_after_steady_steps:
                        self.region_state[r] = RegionState.REBUILD_PENDING
                        self._rebuild_counter[r] = 0
                    else:
                        any_active = True

            elif state == RegionState.REBUILD_PENDING:
                if max_dflow > flow_threshold:
                    self.region_state[r] = RegionState.PLASTIC_ACTIVE
                    self._rebuild_counter[r] = 0
                    any_active = True

        # Snapshot for the next call's delta computation.
        self._eps_p_prev_update = self.eps_p.copy()
        return any_active

    def per_region_max_sigma_eq(self) -> np.ndarray:
        """Maximum equivalent stress per region — the quantity the
        ELASTIC ⇄ PLASTIC_ACTIVE state machine triggers on. Length-K
        float64; regions with zero elements (impossible by construction
        but defensive) report 0."""
        sigma_eq = self.per_element_sigma_eq()
        out = np.zeros(self.n_regions, dtype=np.float64)
        for r in range(self.n_regions):
            elems_r = self.partition.region_elements[r]
            if elems_r.size > 0:
                out[r] = float(sigma_eq[elems_r].max())
        return out

    def all_rebuild_pending(self) -> bool:
        """True iff a coordinated rebuild should fire *right now*.

        Conditions:
        1. No region is currently PLASTIC_ACTIVE.
        2. At least one region is REBUILD_PENDING (so something to absorb).
        3. The body is quasi-static — peak nodal speed below
           ``_quasistatic_v_threshold``. Without this, a rebuild that
           lands mid-oscillation bakes the *current elastic* deformation
           into the new reference (plastic + transient elastic both),
           which over-counts permanent set on subsequent rebuilds.

        ELASTIC regions don't need to re-yield to unblock a rebuild;
        they just had no plastic flow to begin with.
        """
        any_pending = False
        for s in self.region_state:
            if s == RegionState.PLASTIC_ACTIVE:
                return False
            if s == RegionState.REBUILD_PENDING:
                any_pending = True
        if not any_pending:
            return False
        if self._cb.v is not None:
            v_peak = float(np.linalg.norm(self._cb.v, axis=1).max())
            if v_peak > self._quasistatic_v_threshold:
                return False
        return True

    def rebuild_rom(self) -> None:
        """Absorb the current deformed shape into the reference geometry
        and rebuild the CB basis on top of it.

        Steps
        -----
        1. ``mesh.nodes`` ← current ``x`` (so ``x − mesh.nodes`` = 0, the
           new rest configuration is the deformed one).
        2. ``eps_p`` ← 0 — the permanent set is now baked into the
           reference, not stored in plastic-strain history.
        3. Reconstruct the underlying ``CraigBamptonBody`` so dN_list /
           volumes / mass / stiffness / Φ_CB all reflect the new
           reference.
        4. Restore ``v`` so kinetic state survives the rebuild; ``q_r``
           starts at zero in the new modal basis (because ``x =
           new_ref_nodes`` ⇒ disp = 0 ⇒ q_r = 0).
        5. All regions reset to ELASTIC, all counters cleared.

        Caller normally invokes via the global trigger ``step()`` runs
        when ``all_rebuild_pending()`` is True; can also be called
        directly for testing or non-standard cadences.
        """
        if self._cb.x is None:
            raise RuntimeError("rebuild_rom called before initialize()")

        new_ref_nodes = self._cb.x.copy()
        carry_v = self._cb.v.copy() if self._cb.v is not None else None

        # ── 1. Update reference geometry on the mesh in place ──────────
        # The mesh object is owned by this body — mutating its node
        # array is the cheapest way to push the new reference through
        # every downstream precompute (shape gradients are computed at
        # construction from mesh.nodes).
        self.mesh.nodes = new_ref_nodes

        # ── 2. Drop accumulated plasticity ─────────────────────────────
        self.eps_p = np.zeros_like(self.eps_p)
        self._eps_p_prev_update = np.zeros_like(self._eps_p_prev_update)

        # ── 3. Reconstruct CB body around the new reference ────────────
        self._cb = CraigBamptonBody(
            mesh=self.mesh, material=self.material, density=self.density,
            n_modes=self.n_modes,
            fixed_nodes=(self.fixed_nodes if self.fixed_nodes.size > 0 else None),
            gravity=self.gravity, damping=self.damping,
            name=self.name + "_cb",
        )
        self._cb.initialize(dt=self._cached_dt)

        # ── 4. Restore kinetic state — q_r begins at zero ──────────────
        self._cb.x = new_ref_nodes.copy()
        if carry_v is not None:
            self._cb.v = carry_v
        if self._cb.q_r is not None:
            self._cb.q_r[:] = 0.0

        # ── 5. Region state machine reset ──────────────────────────────
        self.region_state = [RegionState.ELASTIC] * self.n_regions
        self._rebuild_counter = [0] * self.n_regions
        self._rebuild_count += 1

    def step(
        self,
        dt: float,
        extra_forces: Optional[dict] = None,
        auto_update_states: bool = True,
    ) -> None:
        """Advance one substep, then run the state-machine tick.

        Routing (Phase 2C-D, all-or-nothing):
        * Every region ELASTIC → :class:`CraigBamptonBody` reduced step.
        * Any region PLASTIC_ACTIVE → whole-body full-FEM implicit-Euler
          step driven by :class:`CorotationalPlastic`. Per-region
          full-FEM with the elastic remainder still reduced is a
          Phase 3 refinement.

        After integration, ``update_region_states`` runs by default
        (``auto_update_states=True``) so the next call sees the right
        state. Callers running their own state-machine cadence can pass
        ``False`` and call :meth:`update_region_states` themselves.
        """
        if self.all_elastic():
            self._cb.step(dt=dt, extra_forces=extra_forces)
        else:
            self._step_full_fem_plastic(dt, extra_forces)
        if auto_update_states:
            self.update_region_states()
            if self.all_rebuild_pending():
                self.rebuild_rom()

    def _step_full_fem_plastic(
        self,
        dt: float,
        extra_forces: Optional[dict],
    ) -> None:
        """One implicit-Euler step with the plastic constitutive model.

        Reads/writes the underlying CB body's x and v so that switching
        back to the all-ELASTIC path picks up the latest configuration.
        """
        body_x = self._cb.x
        body_v = self._cb.v
        if body_x is None or body_v is None:
            raise RuntimeError(
                "HybridCBPlasticBody._step_full_fem_plastic called before "
                "initialize() — x / v are not populated."
            )

        n_dof = self.mesh.n_nodes * 3

        # External forces: gravity + caller-supplied extra (e.g. contact).
        # extra_forces follows the FEMSolver convention {body_index: vec}.
        f_ext = np.zeros(n_dof)
        M_diag = self._cb._M.diagonal()
        for d in range(3):
            f_ext[d::3] += M_diag[d::3] * self.gravity[d]
        if extra_forces and 0 in extra_forces:
            f_ext += extra_forces[0]

        result = implicit_euler_step(
            mesh=self.mesh,
            x=body_x,
            v=body_v,
            f_ext=f_ext,
            dt=dt,
            material=self._plastic_material,
            M=self._cb._M,
            dN_list=self._cb._dN_list,
            volumes=self._cb._volumes,
            fixed_dofs=self._fixed_dofs if self._fixed_dofs.size > 0 else None,
            damping=self.damping,
            max_newton_iters=self._plastic_max_newton,
            eps_p=self.eps_p,
        )

        self._cb.x = result.x_new
        self._cb.v = result.v_new
        if result.eps_p_new is not None:
            self.eps_p = result.eps_p_new

        # Keep the CB body's reduced coordinate q_r in sync with the
        # plastic-step displacement so a subsequent ELASTIC step starts
        # from the right modal state. Project (x - x_ref) onto Φ_CB.
        if self._cb.q_r is not None and self._cb._Phi_CB is not None:
            disp_flat = (result.x_new - self._cb._x_ref_body).ravel()
            free_dofs = self._cb._free_dofs
            disp_free = disp_flat[free_dofs]
            self._cb.q_r = self._cb._C_q @ disp_free
