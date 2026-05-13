"""Stage 9 smoke test: RigidSceneStep API works for a single
free-falling box on a ground plane. Validates that one C++ ``step()``
call advances state in a way that matches the Python path within
penalty-contact tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def cpp():
    try:
        from robosim import _cpp
        assert hasattr(_cpp, "scene_step")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"robosim._cpp.scene_step not built: {e}")
    return _cpp


def test_scene_step_creation(cpp):
    ss = cpp.scene_step.RigidSceneStep()
    assert ss.n_robots() == 0
    assert ss.n_bodies() == 0


def test_falling_box_fast_path_matches_slow(cpp):
    """Drop a free box on the ground and compare 0.5 s of integration
    between the Python slow path and the C++ fast path. They share
    kernels but differ in dispatch order; the final CoM should agree
    to within penalty-contact tolerance."""
    import os
    from robosim import Scene, Box, Ground

    def _run(use_fast: bool):
        if use_fast:
            os.environ.pop("ROBOSIM_NO_FAST_RIGID", None)
        else:
            os.environ["ROBOSIM_NO_FAST_RIGID"] = "1"
        scene = Scene(dt=1e-3, substeps=5)
        scene.add(Ground(height=0.0))
        scene.add(Box(size=(0.10, 0.10, 0.10), mass=1.0,
                      pos=[0.0, 0.0, 0.5]).name("box"))
        scene.run(duration=0.5, headless=True)
        return scene._body_handles["box"]._model.q.copy()

    try:
        q_slow = _run(False)
        q_fast = _run(True)
    finally:
        os.environ.pop("ROBOSIM_NO_FAST_RIGID", None)
    np.testing.assert_allclose(q_slow[:3], q_fast[:3], atol=5e-3)


def test_scene_step_box_falls_to_ground(cpp):
    """A single free box dropped from z=0.5 with gravity should settle
    near z = h/2 (resting on the ground) after a short integration."""
    from robosim.model.factory import create_free_box
    from robosim.model._cpp_bridge import build_rbd_topology

    box = create_free_box(name="probe", size=(0.10, 0.10, 0.10), mass=1.0,
                          position=np.array([0.0, 0.0, 0.5]))
    topo = build_rbd_topology(box)

    ss = cpp.scene_step.RigidSceneStep()
    rid = ss.add_robot(topo, np.array([0.0, 0.0, -9.81]), is_free=True)
    ss.set_state(rid, box.q.copy(), box.qd.copy())

    # Box's collision geometry on the body_link (last link).
    box_link_idx = box.n_links - 1
    bid = ss.add_box(
        rid, box_link_idx,
        np.array([0.05, 0.05, 0.05]),
        np.eye(3),
        np.zeros(3),
    )
    assert bid >= 0

    ss.set_ground(0.0, np.array([0.0, 0.0, 1.0]))
    ss.set_contact_params(stiffness=1e5, damping=2e2,
                           mu=0.5, friction_eps=1e-3,
                           max_penetration=0.01)

    # Integrate for 1 s in 1000 substeps.
    ss.step(1e-3, 1000)

    q, qd = ss.get_state(rid)
    # The free_box is a 6-chain (tx, ty, tz, rx, ry, rz). z-position is
    # element 2.
    z = float(q[2])
    assert z > 0.0, f"box should be above ground, got z={z}"
    assert z < 0.1, f"box should be settled near ground, got z={z}"
