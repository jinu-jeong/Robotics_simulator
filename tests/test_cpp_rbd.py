"""Stage-3 parity: C++ ABA + gravity_torques match the Python kernels
to float round-off (atol 1e-12). Exercises the arm6_gripper URDF
across random q/qd/tau, with and without an external link wrench."""

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
        assert hasattr(_cpp, "rbd")
    except (ImportError, AssertionError) as e:
        pytest.skip(f"robosim._cpp.rbd not built: {e}")
    return _cpp.rbd


@pytest.fixture(scope="module")
def robot():
    from robosim.model.urdf_parser import parse_urdf
    if not _URDF.exists():
        pytest.skip(f"URDF not found: {_URDF}")
    return parse_urdf(str(_URDF))


def _call_aba(robot, q, qd, tau, *, use_cpp, f_ext=None):
    from robosim.physics.rbd.algorithms import aba
    robot._use_cpp_rbd = use_cpp
    if use_cpp:
        robot._cpp_rbd_topo = None
    return aba(robot, q, qd, tau, f_ext=f_ext)


def _call_gravity(robot, q, *, use_cpp):
    from robosim.physics.rbd.algorithms import gravity_torques
    robot._use_cpp_rbd = use_cpp
    if use_cpp:
        robot._cpp_rbd_topo = None
    return gravity_torques(robot, q)


def test_aba_random_inputs_match(robot, cpp):
    rng = np.random.default_rng(11)
    for _ in range(10):
        q   = rng.uniform(-1.0, 1.0, size=robot.n_dof)
        qd  = rng.uniform(-0.5, 0.5, size=robot.n_dof)
        tau = rng.uniform(-3.0, 3.0, size=robot.n_dof)
        py  = _call_aba(robot, q, qd, tau, use_cpp=False)
        cpp_ = _call_aba(robot, q, qd, tau, use_cpp=True)
        np.testing.assert_allclose(py, cpp_, atol=1e-11)


def test_aba_with_external_wrench_matches(robot, cpp):
    rng = np.random.default_rng(13)
    q   = rng.uniform(-1, 1, robot.n_dof)
    qd  = rng.uniform(-0.5, 0.5, robot.n_dof)
    tau = rng.uniform(-2, 2, robot.n_dof)
    f_ext = {3: rng.uniform(-1, 1, 6),
             5: rng.uniform(-1, 1, 6)}
    py  = _call_aba(robot, q, qd, tau, use_cpp=False, f_ext=f_ext)
    cpp_ = _call_aba(robot, q, qd, tau, use_cpp=True,  f_ext=f_ext)
    np.testing.assert_allclose(py, cpp_, atol=1e-11)


def test_gravity_torques_random_inputs_match(robot, cpp):
    rng = np.random.default_rng(17)
    for _ in range(10):
        q = rng.uniform(-1, 1, robot.n_dof)
        py = _call_gravity(robot, q, use_cpp=False)
        cpp_ = _call_gravity(robot, q, use_cpp=True)
        np.testing.assert_allclose(py, cpp_, atol=1e-12)


def test_aba_zero_inputs_gives_only_gravity(robot, cpp):
    """ABA(q, 0, 0) is the integrated form of the gravity term — checks
    that the C++ and Python paths agree on this corner."""
    q = np.zeros(robot.n_dof)
    qd = np.zeros(robot.n_dof)
    tau = np.zeros(robot.n_dof)
    py = _call_aba(robot, q, qd, tau, use_cpp=False)
    cpp_ = _call_aba(robot, q, qd, tau, use_cpp=True)
    np.testing.assert_allclose(py, cpp_, atol=1e-12)
