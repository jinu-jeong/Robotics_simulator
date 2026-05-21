"""Tests for CraigBamptonBody.set_anchor_pose — moving-frame anchored CB.

The legacy anchored mode treats ``fixed_nodes`` as world-stationary
(R=I, t=0). For deformable robot links the anchor frame moves each step
with the parent link. set_anchor_pose(R, t) prescribes that frame; the
small-displacement CB linearisation runs in local (anchor-relative)
coordinates and reconstructs to world.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.physics.fem.mesh import FEMesh
from robosim.physics.fem.materials import CorotationalElastic
from robosim.physics.fem.reduced import CraigBamptonBody


def _make_body(use_cpp: bool = False):
    """Build a small anchored CB body: 4×4×4 cm box, bottom face fixed."""
    size = np.array([0.04, 0.04, 0.04])
    mesh = FEMesh.create_hex_box(
        origin=np.zeros(3), size=size, divisions=(2, 2, 2),
    )
    fixed = np.where(mesh.nodes[:, 2] < 1e-6)[0]
    body = CraigBamptonBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e5, poisson=0.30),
        density=1000.0,
        n_modes=4,
        fixed_nodes=fixed,
        gravity=np.zeros(3),      # gravity-free for clean anchor tests
        damping=0.0,
        name="test_anchored",
    )
    body._use_cpp_cb = use_cpp
    body.initialize(dt=0.001)
    return body, fixed


def test_default_anchor_is_world_stationary():
    """No set_anchor_pose call → fixed nodes stay at mesh.nodes."""
    body, fixed = _make_body()
    assert body._anchor_R is None and body._anchor_t is None
    # No forces → no motion expected
    for _ in range(5):
        body.step(dt=0.001)
    np.testing.assert_allclose(
        body.x[fixed], body.mesh.nodes[fixed], atol=1e-12,
    )


def test_identity_anchor_runs_and_keeps_fixed_at_reference():
    """set_anchor_pose(I, 0) runs without error and keeps fixed nodes
    at the reference position (matrix path's R=I, t=0 special case).

    Bit-for-bit parity with the stationary (legacy) path is NOT a
    requirement — they are intentionally separate code paths so the
    legacy callers stay untouched, and the matrix path goes through a
    few extra matmuls that introduce floating-point noise. We only
    require the moving path's R=I, t=0 case to be physically sensible.
    """
    body, fixed = _make_body()
    body.set_anchor_pose(np.eye(3), np.zeros(3))

    n_dof = body.mesh.n_nodes * 3
    rng = np.random.default_rng(42)
    for _ in range(5):
        f = rng.standard_normal(n_dof) * 1e-3
        body.step(dt=0.001, extra_forces={0: f.copy()})

    # No NaN/Inf
    assert np.all(np.isfinite(body.x))
    assert np.all(np.isfinite(body.v))
    # Fixed nodes stay clamped at the (identity-transformed) reference
    np.testing.assert_allclose(
        body.x[fixed], body.mesh.nodes[fixed], atol=1e-12,
    )


def test_translated_anchor_carries_fixed_nodes():
    """set_anchor_pose(I, t) → fixed nodes sit at mesh.nodes + t each step."""
    body, fixed = _make_body()
    t_world = np.array([0.50, 0.30, 0.10])
    body.set_anchor_pose(np.eye(3), t_world)
    # Initialise body state at the translated reference so there is no
    # spurious initial displacement (otherwise the modal coords would
    # ring back to the new equilibrium).
    body.x = body.mesh.nodes + t_world
    body.v = np.zeros_like(body.v)

    for _ in range(5):
        body.step(dt=0.001)

    expected = body.mesh.nodes[fixed] + t_world
    np.testing.assert_allclose(body.x[fixed], expected, atol=1e-12)


def test_rotated_anchor_carries_fixed_nodes():
    """set_anchor_pose(R, 0) → fixed nodes sit at R @ mesh.nodes each step."""
    body, fixed = _make_body()
    theta = 0.7
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    body.set_anchor_pose(R, np.zeros(3))
    body.x = body.mesh.nodes @ R.T
    body.v = np.zeros_like(body.v)

    for _ in range(5):
        body.step(dt=0.001)

    expected = body.mesh.nodes[fixed] @ R.T
    np.testing.assert_allclose(body.x[fixed], expected, atol=1e-10)


def test_set_anchor_pose_rejects_free_body():
    """Calling on a body with no fixed_nodes should raise ValueError."""
    size = np.array([0.04, 0.04, 0.04])
    mesh = FEMesh.create_hex_box(
        origin=np.zeros(3), size=size, divisions=(2, 2, 2),
    )
    body = CraigBamptonBody(
        mesh=mesh,
        material=CorotationalElastic(young=1e5, poisson=0.30),
        density=1000.0,
        n_modes=4,
        fixed_nodes=None,    # free body
        gravity=np.zeros(3),
        damping=0.0,
        name="test_free",
    )
    with pytest.raises(ValueError, match="anchored"):
        body.set_anchor_pose(np.eye(3), np.zeros(3))
