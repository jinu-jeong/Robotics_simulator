"""Stage-2 parity: C++ FK matches the Python BFS to float round-off.

We exercise the full arm6_gripper URDF (6-DOF arm + 2 mimic finger
joints) so the test covers all four ``JointType`` paths (REVOLUTE,
PRISMATIC, FIXED, plus mimic propagation via ``enforce_mimic``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


_URDF = (
    Path(__file__).resolve().parent.parent
    / "examples" / "pure_simulation" / "urdf" / "arm6_gripper" / "arm6_gripper.urdf"
)


@pytest.fixture(scope="module")
def cpp():
    try:
        from robosim import _cpp
        assert hasattr(_cpp, "kin")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"robosim._cpp.kin not built: {e}")
    return _cpp.kin


@pytest.fixture(scope="module")
def robot():
    from robosim.model.urdf_parser import parse_urdf
    if not _URDF.exists():
        pytest.skip(f"URDF not found: {_URDF}")
    return parse_urdf(str(_URDF))


def _fk_with_backend(robot, *, use_cpp: bool):
    robot.invalidate_fk_cache()
    robot._use_cpp_fk = use_cpp
    robot._cpp_topo = None
    return robot.forward_kinematics()


@pytest.fixture
def random_qs(robot):
    rng = np.random.default_rng(42)
    return [rng.uniform(-1.0, 1.0, size=robot.n_dof) for _ in range(10)]


def test_cpp_fk_matches_python_fk_random(robot, cpp, random_qs):
    for q in random_qs:
        robot.q[:] = q
        robot.enforce_mimic()
        fk_py = _fk_with_backend(robot, use_cpp=False)
        fk_cpp = _fk_with_backend(robot, use_cpp=True)
        for l, (a, b) in enumerate(zip(fk_py, fk_cpp)):
            np.testing.assert_allclose(
                a.rotation, b.rotation, atol=1e-12,
                err_msg=f"link {l} rotation mismatch at q={q}",
            )
            np.testing.assert_allclose(
                a.translation, b.translation, atol=1e-12,
                err_msg=f"link {l} translation mismatch at q={q}",
            )


def test_cpp_fk_at_home_pose_matches_python(robot):
    robot.q[:] = 0.0
    robot.enforce_mimic()
    fk_py = _fk_with_backend(robot, use_cpp=False)
    fk_cpp = _fk_with_backend(robot, use_cpp=True)
    for a, b in zip(fk_py, fk_cpp):
        np.testing.assert_allclose(a.rotation, b.rotation, atol=1e-12)
        np.testing.assert_allclose(a.translation, b.translation, atol=1e-12)


def test_topology_n_links_and_n_dof(robot, cpp):
    from robosim.model._cpp_bridge import build_topology
    topo = build_topology(robot)
    assert topo.n_links == robot.n_links
    assert topo.n_dof == robot.n_dof
    assert len(topo.joints) == robot.n_joints
    assert len(topo.root_links) >= 1
