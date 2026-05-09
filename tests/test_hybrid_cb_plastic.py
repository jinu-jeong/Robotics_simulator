"""Tests for HybridCBPlasticBody.

Phase 2B: only the all-ELASTIC fast path is wired (delegates to the
underlying CraigBamptonBody). The contract checked here:

  * Construction succeeds and the region-partition / state arrays are
    correctly sized for K = 1, 4, 8.
  * With ``n_regions = 1`` and a purely elastic loading path the hybrid
    body produces *bit-identical* trajectories to a standalone
    CraigBamptonBody on the same mesh / material / forcing — i.e. wrapping
    in the hybrid container costs nothing semantically when no plasticity
    is invoked.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pymetis", reason="pymetis not installed")

from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.mesh import FEMesh, TetMesh
from robosim.physics.fem.hybrid import HybridCBPlasticBody, RegionState
from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver


def _hex_box(divisions=(4, 4, 4)) -> FEMesh:
    """Hex mesh — used for tests that don't go through the plastic path
    (constructor sizing, K=1 elastic equivalence)."""
    return FEMesh.create_hex_box(
        origin=np.array([0.0, 0.0, 0.0]),
        size=np.array([0.2, 0.05, 0.05]),
        divisions=divisions,
    )


def _tet_box(divisions=(4, 2, 2)) -> TetMesh:
    """Tet mesh — used for plastic-active branch tests since the
    plastic Tet4 path is the only one wired in Phase 1."""
    return TetMesh.create_box(
        origin=np.array([0.0, 0.0, 0.0]),
        size=np.array([0.2, 0.05, 0.05]),
        divisions=divisions,
    )


@pytest.mark.parametrize("K", [1, 4, 8])
def test_constructor_state_arrays_sized_correctly(K):
    mesh = _hex_box()
    body = HybridCBPlasticBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        n_modes=6,
        n_regions=K,
        name="hybrid",
    )
    assert body.n_regions == K
    assert len(body.region_state) == K
    assert all(s == RegionState.ELASTIC for s in body.region_state)
    assert body.all_elastic()
    assert body.partition.element_region.shape == (mesh.n_elements,)
    assert body.eps_p.shape == (mesh.n_elements, 3, 3)
    assert np.linalg.norm(body.eps_p) == 0.0


def test_K1_elastic_matches_standalone_CB():
    """With K=1 and the body never leaving ELASTIC, the hybrid path
    delegates to CraigBamptonBody verbatim. Trajectories under identical
    init + identical forces should match to float tolerance."""
    mesh = _hex_box()
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    material = CorotationalElastic(young=1e6, poisson=0.3)
    dt = 5e-4

    standalone = CraigBamptonBody(
        mesh=mesh, material=material, density=1000.0,
        n_modes=8, fixed_nodes=fixed,
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.5, name="ref",
    )
    hybrid = HybridCBPlasticBody(
        mesh=mesh, material=material, density=1000.0,
        n_modes=8, fixed_nodes=fixed,
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.5, name="hyb", n_regions=1,
    )

    sv_ref = CraigBamptonSolver(bodies=[standalone])
    sv_hyb = CraigBamptonSolver(bodies=[hybrid._cb])
    sv_ref.initialize(dt=dt)
    hybrid.initialize(dt=dt)
    sv_hyb.initialize(dt=dt)

    for _ in range(50):
        sv_ref.step()
        sv_hyb.step()

    assert standalone.x is not None and hybrid.x is not None
    np.testing.assert_allclose(hybrid.x, standalone.x, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(hybrid.v, standalone.v, atol=1e-12, rtol=1e-12)


def test_active_branch_runs_and_grows_plastic_strain():
    """Force a region into PLASTIC_ACTIVE under a load that yields and
    confirm the full-FEM branch advances time and accumulates eps_p."""
    mesh = _tet_box()
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    body = HybridCBPlasticBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        density=1000.0, n_modes=8, fixed_nodes=fixed,
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.5, name="hyb",
        yield_stress=5e2, hardening=2e5, n_regions=2,
    )
    body.initialize(dt=5e-4)
    body.region_state[0] = RegionState.PLASTIC_ACTIVE   # manual trigger

    # Tip-pull: large -X load on the right face to push past yield.
    right = np.where(mesh.nodes[:, 0] > 0.2 - 1e-9)[0]
    n_dof = mesh.n_nodes * 3
    f_pull = np.zeros(n_dof)
    f_pull[right * 3 + 2] = -50.0 / max(1, len(right))   # -Z load at tip

    for _ in range(80):
        body.step(dt=5e-4, extra_forces={0: f_pull})

    # Body must have moved.
    assert body.x is not None
    disp = float(np.linalg.norm(body.x - mesh.nodes))
    assert disp > 1e-4, f"plastic-active branch did not deform: {disp:.6f}"
    # Plastic strain accumulated somewhere.
    assert np.linalg.norm(body.eps_p) > 1e-4


def test_active_then_back_to_elastic_preserves_state():
    """Run a few PLASTIC_ACTIVE steps, then flip back to ELASTIC and
    ensure the next CB step still works (q_r resync is correct)."""
    mesh = _tet_box()
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    body = HybridCBPlasticBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        density=1000.0, n_modes=6, fixed_nodes=fixed,
        gravity=np.array([0.0, 0.0, -9.81]),
        damping=0.5,
        yield_stress=5e2, hardening=1e5, n_regions=2,
    )
    body.initialize(dt=5e-4)
    body.region_state[0] = RegionState.PLASTIC_ACTIVE

    n_dof = mesh.n_nodes * 3
    right = np.where(mesh.nodes[:, 0] > 0.2 - 1e-9)[0]
    f_pull = np.zeros(n_dof)
    f_pull[right * 3 + 2] = -30.0 / max(1, len(right))
    for _ in range(40):
        body.step(dt=5e-4, extra_forces={0: f_pull})

    x_active = body.x.copy()
    body.region_state[0] = RegionState.ELASTIC
    # Removing the load + a few elastic steps must not crash.
    for _ in range(5):
        body.step(dt=5e-4)

    assert body.x is not None
    # State changed (CB still moves under gravity/damping).
    assert not np.allclose(body.x, x_active, atol=1e-12)
