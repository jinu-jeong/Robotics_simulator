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
        # plastic-active full-FEM path will land in a sibling
        # DeformableBody added in Phase 2C-D.
        self._cb = CraigBamptonBody(
            mesh=mesh, material=material, density=density,
            n_modes=n_modes, fixed_nodes=fixed_nodes,
            gravity=gravity, damping=damping, name=name + "_cb",
        )

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

        Phase 2B routing: if every region is ELASTIC, this is exactly the
        underlying CraigBamptonBody.step. Subsequent commits will add the
        PLASTIC_ACTIVE branch (full-FEM in active regions, coupled at the
        inter-region boundary nodes) and the REBUILD branch (recompute
        the CB basis at the current deformed shape).
        """
        if self.all_elastic():
            return self._cb.step(dt=dt, extra_forces=extra_forces)

        # Placeholder: PLASTIC_ACTIVE / REBUILD paths come in 2C-2F.
        raise NotImplementedError(
            "HybridCBPlasticBody: only the all-ELASTIC fast path is wired in "
            "Phase 2B; PLASTIC_ACTIVE branch arrives in Phase 2C-D."
        )
