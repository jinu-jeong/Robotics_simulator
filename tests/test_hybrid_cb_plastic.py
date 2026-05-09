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
from robosim.physics.fem.mesh import FEMesh
from robosim.physics.fem.hybrid import HybridCBPlasticBody, RegionState
from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver


def _hex_box(divisions=(4, 4, 4)) -> FEMesh:
    return FEMesh.create_hex_box(
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


def test_phase_2B_plastic_branch_not_yet_implemented():
    """Sanity guard: forcing a region into PLASTIC_ACTIVE before the
    branch exists should raise NotImplementedError, not silently fall
    through to the CB-only fast path. Removed once 2C-D lands."""
    mesh = _hex_box()
    body = HybridCBPlasticBody(
        mesh=mesh, n_modes=4, n_regions=2,
        material=CorotationalElastic(young=1e5, poisson=0.3),
    )
    body.initialize(dt=1e-3)
    body.region_state[0] = RegionState.PLASTIC_ACTIVE
    with pytest.raises(NotImplementedError):
        body.step(dt=1e-3)
