"""Integration test for FEM corotational J2 plasticity end-to-end.

Builds a small clamped bar (TetMesh), pulls one face, releases, and
checks that:
- Elastic body returns fully to rest.
- Plastic body retains permanent (non-zero) deformation after release.
- Plastic strain is bounded by physical magnitude (sanity).
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.fem.materials import CorotationalElastic, CorotationalPlastic
from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.solver import DeformableBody, FEMSolver


def _hex_to_5tet_bar(nx=4, ny=2, nz=2, lx=0.4, ly=0.05, lz=0.05) -> TetMesh:
    """Build a slender bar from hexahedral cells split into 5 tets each.

    Returns a :class:`TetMesh`. Bar long-axis is along +X with corner at
    origin; node (0..) at x=0 are clamped by the caller.
    """
    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    nodes = np.array([[x, y, z] for x in xs for y in ys for z in zs],
                     dtype=np.float64)

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    # 5-tet decomposition of a unit cube (standard, alternating per cell).
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [
                    nid(i,     j,     k),     nid(i + 1, j,     k),
                    nid(i + 1, j + 1, k),     nid(i,     j + 1, k),
                    nid(i,     j,     k + 1), nid(i + 1, j,     k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i,     j + 1, k + 1),
                ]
                if (i + j + k) % 2 == 0:
                    tets.append([v[0], v[1], v[3], v[4]])
                    tets.append([v[1], v[2], v[3], v[6]])
                    tets.append([v[1], v[5], v[4], v[6]])
                    tets.append([v[3], v[4], v[7], v[6]])
                    tets.append([v[1], v[3], v[4], v[6]])
                else:
                    tets.append([v[0], v[1], v[2], v[5]])
                    tets.append([v[0], v[2], v[3], v[7]])
                    tets.append([v[0], v[2], v[7], v[5]])
                    tets.append([v[0], v[5], v[7], v[4]])
                    tets.append([v[2], v[5], v[6], v[7]])
    return TetMesh(nodes=nodes, elements=np.asarray(tets, dtype=np.int64))


def _run_pull_then_release(material, n_pull=200, n_release=600,
                           pull_force=200.0, dt=5e-4):
    mesh = _hex_to_5tet_bar()
    # Clamp the x=0 face nodes.
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    # Right-face nodes (where the pull force is applied).
    right = np.where(mesh.nodes[:, 0] > 0.4 - 1e-9)[0]

    body = DeformableBody(
        name="bar", mesh=mesh, material=material,
        density=1000.0, fixed_nodes=fixed,
    )
    solver = FEMSolver(bodies=[body], dt=dt,
                       gravity=np.zeros(3), damping=0.5,
                       max_newton_iters=15)
    solver.initialize(dt)

    n_dof = body.mesh.n_nodes * 3
    f_pull = np.zeros(n_dof)
    for n in right:
        f_pull[n * 3] = pull_force / len(right)   # +X tension

    # Pull phase
    for _ in range(n_pull):
        solver.step(dt, extra_forces={0: f_pull})
    x_after_pull = body.x.copy()

    # Release phase: no external force, let it relax.
    for _ in range(n_release):
        solver.step(dt)

    return body, x_after_pull


def _max_x_displacement(body) -> float:
    return float((body.x[:, 0] - body.mesh.nodes[:, 0]).max())


@pytest.mark.parametrize("density", [1000.0])
def test_elastic_bar_returns_to_rest(density):
    """A purely elastic bar pulled and released should return to within
    a fraction of a millimetre of its rest configuration."""
    material = CorotationalElastic(young=1e6, poisson=0.3)
    body, x_pulled = _run_pull_then_release(
        material, n_pull=200, n_release=800, pull_force=80.0,
    )
    pulled_disp = _max_x_displacement(
        type("B", (), {"x": x_pulled, "mesh": body.mesh})()
    )
    final_disp = _max_x_displacement(body)

    # The bar must have actually moved during the pull (sanity).
    assert pulled_disp > 1e-4, f"pull did not deform: {pulled_disp:.6f}"
    # Elastic recovery: residual displacement small relative to peak.
    assert abs(final_disp) < 0.20 * pulled_disp


def test_plastic_bar_retains_permanent_deformation():
    """Same loading on a plastic bar should leave a clearly residual
    displacement after release. Uses linear isotropic hardening so the
    yield surface expands during loading — perfect plasticity reverse-
    yields back to the origin under elastic restoring force during
    release (Bauschinger-free) and would lose all residual deformation."""
    material = CorotationalPlastic(
        young=1e6, poisson=0.3, yield_stress=1e3, hardening=2e5,
    )
    body, x_pulled = _run_pull_then_release(
        material, n_pull=200, n_release=800, pull_force=40.0,
    )
    pulled_disp = _max_x_displacement(
        type("B", (), {"x": x_pulled, "mesh": body.mesh})()
    )
    final_disp = _max_x_displacement(body)

    assert pulled_disp > 1e-4, f"pull did not deform: {pulled_disp:.6f}"
    # Plastic body keeps a meaningful fraction of the loaded displacement.
    assert final_disp > 0.20 * pulled_disp, (
        f"expected residual deformation > 20 % of pulled "
        f"({pulled_disp:.6f}); got {final_disp:.6f}"
    )

    # eps_p should be non-zero somewhere.
    assert body.eps_p is not None
    assert np.linalg.norm(body.eps_p) > 1e-4


def test_plastic_below_yield_behaves_elastic():
    """Light pull that never crosses yield ⇒ same final state as elastic."""
    pull = 5.0  # very light
    body_e, _ = _run_pull_then_release(
        CorotationalElastic(young=1e6, poisson=0.3),
        n_pull=80, n_release=400, pull_force=pull,
    )
    body_p, _ = _run_pull_then_release(
        CorotationalPlastic(young=1e6, poisson=0.3,
                            yield_stress=1e6, hardening=0.0),
        n_pull=80, n_release=400, pull_force=pull,
    )
    # Both should be very close to rest after release.
    np.testing.assert_allclose(body_p.x, body_e.x, atol=1e-5)
    # And no plastic strain accumulated.
    assert np.linalg.norm(body_p.eps_p) < 1e-8
