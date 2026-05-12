"""Stage-4 parity: C++ CB step (anchored + free) matches the Python
reference to float round-off when both run against the **same** body.

Independent ``CraigBamptonBody`` instances would have differently-
signed eigenvectors (``scipy.sparse.linalg.eigsh`` is sign-
non-deterministic), so cross-body comparison would falsely trigger
divergence — the test re-runs the same body on both backends with
checkpointed state instead.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cpp():
    try:
        from robosim import _cpp
        assert hasattr(_cpp, "cb")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"robosim._cpp.cb not built: {e}")
    return _cpp.cb


@pytest.fixture
def anchored_body():
    from robosim.physics.fem.materials import CorotationalElastic
    from robosim.physics.fem.mesh import FEMesh
    from robosim.physics.fem.reduced import CraigBamptonBody

    mesh = FEMesh.create_hex_box(np.array([0.0, 0, 0]),
                                  np.array([0.2, 0.05, 0.05]),
                                  divisions=(4, 4, 4))
    fixed = np.where(mesh.nodes[:, 0] < 1e-9)[0]
    return CraigBamptonBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e6, poisson=0.3),
        density=1000.0, n_modes=6, fixed_nodes=fixed,
        gravity=np.array([0, 0, -9.81]),
        damping=0.5, name="anchored",
    )


@pytest.fixture
def free_body():
    from robosim.physics.fem.materials import CorotationalElastic
    from robosim.physics.fem.mesh import FEMesh
    from robosim.physics.fem.reduced import CraigBamptonBody

    mesh = FEMesh.create_hex_box(np.array([0.0, 0, 0]),
                                  np.array([0.08, 0.08, 0.08]),
                                  divisions=(4, 4, 4))
    return CraigBamptonBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e5, poisson=0.45),
        density=1000.0, n_modes=10,
        fixed_nodes=np.array([], dtype=np.int64),
        gravity=np.array([0, 0, -9.81]),
        damping=0.5, name="free",
    )


def _checkpoint_then_step(body, extra_forces, dt, use_cpp):
    from robosim.physics.fem.reduced import CraigBamptonSolver
    sv = CraigBamptonSolver(bodies=[body])
    sv.initialize(dt=dt)
    body._use_cpp_cb = use_cpp
    body._cpp_cb_ready = False
    sv.step(dt, extra_forces=extra_forces)
    return body.x.copy(), body.v.copy()


def test_anchored_one_step_parity(anchored_body, cpp):
    """One step from rest on the same anchored body — Python and C++
    paths must reach the same x, v within float roundoff."""
    body = anchored_body
    n_dof = body.mesh.n_nodes * 3
    right = np.where(body.mesh.nodes[:, 0] > 0.2 - 1e-9)[0]
    f = np.zeros(n_dof)
    f[right * 3 + 2] = -8.0 / max(1, len(right))
    extra = {0: f}

    # checkpoint at rest
    body.x = body.mesh.nodes.copy()
    body.v = np.zeros_like(body.mesh.nodes)
    x_save = body.x.copy(); v_save = body.v.copy()

    x_py, v_py = _checkpoint_then_step(body, extra, 5e-4, use_cpp=False)

    body.x = x_save; body.v = v_save
    x_cp, v_cp = _checkpoint_then_step(body, extra, 5e-4, use_cpp=True)

    np.testing.assert_allclose(x_py, x_cp, atol=1e-12)
    np.testing.assert_allclose(v_py, v_cp, atol=1e-10)


def test_anchored_many_steps_parity(anchored_body, cpp):
    """50 steps from rest under sustained load — drift bounded by the
    cumulative float-roundoff budget across single-step matvecs.
    Threshold chosen ~ 50 · 1e-12 · max(|x|) headroom."""
    body = anchored_body
    n_dof = body.mesh.n_nodes * 3
    right = np.where(body.mesh.nodes[:, 0] > 0.2 - 1e-9)[0]
    f = np.zeros(n_dof)
    f[right * 3 + 2] = -8.0 / max(1, len(right))
    extra = {0: f}

    body.x = body.mesh.nodes.copy()
    body.v = np.zeros_like(body.mesh.nodes)
    x_save = body.x.copy(); v_save = body.v.copy()

    from robosim.physics.fem.reduced import CraigBamptonSolver
    sv = CraigBamptonSolver(bodies=[body])
    sv.initialize(dt=5e-4)
    body._use_cpp_cb = False
    for _ in range(50): sv.step(5e-4, extra_forces=extra)
    x_py = body.x.copy(); v_py = body.v.copy()

    body.x = x_save; body.v = v_save
    body._use_cpp_cb = True; body._cpp_cb_ready = False
    sv.step(5e-4, extra_forces=extra)   # reinit / warmup
    body.x = x_save; body.v = v_save
    for _ in range(50): sv.step(5e-4, extra_forces=extra)
    x_cp = body.x.copy(); v_cp = body.v.copy()

    np.testing.assert_allclose(x_py, x_cp, atol=1e-9)
    np.testing.assert_allclose(v_py, v_cp, atol=1e-6)


def test_free_one_step_parity(free_body, cpp):
    """Free-body parity, single step from rest."""
    body = free_body
    body.x = body.mesh.nodes.copy()
    body.v = np.zeros_like(body.mesh.nodes)
    x_save = body.x.copy(); v_save = body.v.copy()

    x_py, v_py = _checkpoint_then_step(body, None, 5e-4, use_cpp=False)

    body.x = x_save; body.v = v_save
    x_cp, v_cp = _checkpoint_then_step(body, None, 5e-4, use_cpp=True)

    np.testing.assert_allclose(x_py, x_cp, atol=1e-12)
    np.testing.assert_allclose(v_py, v_cp, atol=1e-12)


def test_free_many_steps_parity(free_body, cpp):
    """Free-body parity over 20 steps under gravity. Earlier port
    diverged here from the missing mass-weighted projection; this
    test pins the regression so the bug stays fixed."""
    body = free_body
    body.x = body.mesh.nodes.copy()
    body.v = np.zeros_like(body.mesh.nodes)
    x_save = body.x.copy(); v_save = body.v.copy()

    from robosim.physics.fem.reduced import CraigBamptonSolver
    sv = CraigBamptonSolver(bodies=[body])
    sv.initialize(dt=5e-4)

    body._use_cpp_cb = False
    for _ in range(20): sv.step(5e-4)
    x_py = body.x.copy(); v_py = body.v.copy()

    body.x = x_save; body.v = v_save
    body._use_cpp_cb = True; body._cpp_cb_ready = False
    for _ in range(20): sv.step(5e-4)
    x_cp = body.x.copy(); v_cp = body.v.copy()

    assert not np.any(np.isnan(x_cp)), "C++ free-body produced NaN"
    np.testing.assert_allclose(x_py, x_cp, atol=1e-12)
    np.testing.assert_allclose(v_py, v_cp, atol=1e-11)
