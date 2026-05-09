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

    def all_elastic(self) -> bool:
        return all(s == RegionState.ELASTIC for s in self.region_state)

    def initialize(self, dt: float) -> None:
        """Build the CB basis (delegates to CraigBamptonBody)."""
        self._cb.initialize(dt=dt)

    def step(
        self,
        dt: float,
        extra_forces: Optional[dict] = None,
    ) -> None:
        """Advance one substep.

        Routing (Phase 2C-D, all-or-nothing):
        * Every region ELASTIC → :class:`CraigBamptonBody` reduced step.
        * Any region PLASTIC_ACTIVE → fall back to a *whole-body*
          full-FEM implicit-Euler step driven by
          :class:`CorotationalPlastic`. Per-region full-FEM with the
          elastic remainder still reduced is a Phase 3 refinement; at
          this stage the simpler whole-body fallback is correct (just
          conservative on speed) and keeps the state transition trivial.
        """
        if self.all_elastic():
            return self._cb.step(dt=dt, extra_forces=extra_forces)
        return self._step_full_fem_plastic(dt, extra_forces)

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
