"""Stage-1 parity tests: C++ transform helpers vs ``robosim/math/transforms.py``.

Every entry point in ``robosim._cpp.transform`` must reproduce the
NumPy reference to float64 round-off. Element-wise ``atol=1e-12``
catches algorithmic divergence; ``rtol=0`` ensures no silent sign or
ordering bugs.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.math.transforms import (
    Transform,
    cross3 as py_cross3,
    skew as py_skew,
    rotation_x as py_rx,
    rotation_y as py_ry,
    rotation_z as py_rz,
)


@pytest.fixture(scope="module")
def cpp():
    try:
        from robosim._cpp import transform as cppt
    except ImportError as e:
        pytest.skip(f"robosim._cpp.transform not built: {e}")
    return cppt


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


def _random_rotation(rng):
    """A valid SO(3) via QR of a random 3x3."""
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _random_axis_angle(rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(-np.pi, np.pi))
    return axis, angle


def test_cross3_matches_numpy(cpp, rng):
    for _ in range(20):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        np.testing.assert_allclose(cpp.cross3(a, b), py_cross3(a, b), atol=1e-12)


def test_skew_matches_numpy(cpp, rng):
    for _ in range(20):
        v = rng.standard_normal(3)
        np.testing.assert_allclose(cpp.skew(v), py_skew(v), atol=1e-12)


@pytest.mark.parametrize("angle", [-2.4, -0.7, 0.0, 0.3, 1.5, 3.1])
def test_rotation_axes_match_numpy(cpp, angle):
    np.testing.assert_allclose(cpp.rotation_x(angle), py_rx(angle), atol=1e-12)
    np.testing.assert_allclose(cpp.rotation_y(angle), py_ry(angle), atol=1e-12)
    np.testing.assert_allclose(cpp.rotation_z(angle), py_rz(angle), atol=1e-12)


def test_from_axis_angle_matches_numpy(cpp, rng):
    for _ in range(30):
        axis, angle = _random_axis_angle(rng)
        R_cpp = cpp.from_axis_angle(axis, angle)
        R_py = Transform.from_axis_angle(axis, angle).rotation
        np.testing.assert_allclose(R_cpp, R_py, atol=1e-12)


def test_from_axis_angle_zero_axis_returns_identity(cpp):
    R = cpp.from_axis_angle(np.zeros(3), 1.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_compose_Rt_matches_numpy(cpp, rng):
    for _ in range(30):
        R1, t1 = _random_rotation(rng), rng.standard_normal(3)
        R2, t2 = _random_rotation(rng), rng.standard_normal(3)
        R_cpp, t_cpp = cpp.compose_Rt(R1, t1, R2, t2)
        T_py = Transform(rotation=R1, translation=t1).compose(
            Transform(rotation=R2, translation=t2)
        )
        np.testing.assert_allclose(R_cpp, T_py.rotation, atol=1e-12)
        np.testing.assert_allclose(t_cpp, T_py.translation, atol=1e-12)


def test_inverse_Rt_matches_numpy(cpp, rng):
    for _ in range(30):
        R, t = _random_rotation(rng), rng.standard_normal(3)
        R_inv, t_inv = cpp.inverse_Rt(R, t)
        T = Transform(rotation=R, translation=t).inverse()
        np.testing.assert_allclose(R_inv, T.rotation, atol=1e-12)
        np.testing.assert_allclose(t_inv, T.translation, atol=1e-12)


def test_inverse_undoes_compose(cpp, rng):
    """Algebraic round-trip: T ∘ T⁻¹ = identity."""
    for _ in range(20):
        R, t = _random_rotation(rng), rng.standard_normal(3)
        R_inv, t_inv = cpp.inverse_Rt(R, t)
        R_id, t_id = cpp.compose_Rt(R, t, R_inv, t_inv)
        np.testing.assert_allclose(R_id, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(t_id, np.zeros(3), atol=1e-12)


def test_apply_point_and_vector_match_numpy(cpp, rng):
    for _ in range(20):
        R, t = _random_rotation(rng), rng.standard_normal(3)
        p = rng.standard_normal(3)
        T = Transform(rotation=R, translation=t)
        np.testing.assert_allclose(cpp.apply_point(R, t, p),
                                   T.apply_point(p), atol=1e-12)
        np.testing.assert_allclose(cpp.apply_vector(R, p),
                                   T.apply_vector(p), atol=1e-12)
