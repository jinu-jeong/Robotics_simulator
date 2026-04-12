"""Tests for SE(3) transforms and spatial algebra."""

import math

import numpy as np
import pytest

from robosim.math.transforms import (
    Transform,
    rotation_x,
    rotation_y,
    rotation_z,
    skew,
    unskew,
)
from robosim.math.spatial import (
    SpatialInertia,
    spatial_cross_force,
    spatial_cross_motion,
    spatial_transform_force,
    spatial_transform_motion,
)


class TestTransform:
    def test_identity(self):
        T = Transform.identity()
        np.testing.assert_array_almost_equal(T.rotation, np.eye(3))
        np.testing.assert_array_almost_equal(T.translation, np.zeros(3))

    def test_inverse_identity(self):
        T = Transform.identity()
        T_inv = T.inverse()
        np.testing.assert_array_almost_equal(T_inv.to_matrix(), np.eye(4))

    def test_inverse_general(self):
        T = Transform.from_rpy(0.3, 0.5, -0.2)
        T.translation = np.array([1.0, 2.0, 3.0])
        T_inv = T.inverse()
        result = T.compose(T_inv)
        np.testing.assert_array_almost_equal(result.to_matrix(), np.eye(4), decimal=10)

    def test_compose_associativity(self):
        T1 = Transform.from_rpy(0.1, 0.2, 0.3)
        T1.translation = np.array([1, 0, 0])
        T2 = Transform.from_rpy(-0.1, 0.4, 0.0)
        T2.translation = np.array([0, 1, 0])
        T3 = Transform.from_rpy(0.0, 0.0, 0.5)
        T3.translation = np.array([0, 0, 1])

        # (T1 @ T2) @ T3 == T1 @ (T2 @ T3)
        left = (T1 @ T2) @ T3
        right = T1 @ (T2 @ T3)
        np.testing.assert_array_almost_equal(
            left.to_matrix(), right.to_matrix(), decimal=10
        )

    def test_apply_point(self):
        T = Transform.from_translation(np.array([1, 2, 3]))
        p = np.array([0, 0, 0])
        result = T.apply_point(p)
        np.testing.assert_array_almost_equal(result, [1, 2, 3])

    def test_apply_vector_translation_invariant(self):
        T = Transform.from_translation(np.array([10, 20, 30]))
        v = np.array([1, 0, 0])
        result = T.apply_vector(v)
        np.testing.assert_array_almost_equal(result, [1, 0, 0])

    def test_rotation_x_90(self):
        T = Transform(rotation=rotation_x(math.pi / 2))
        p = np.array([0, 1, 0])
        result = T.apply_point(p)
        np.testing.assert_array_almost_equal(result, [0, 0, 1], decimal=10)

    def test_rotation_z_90(self):
        T = Transform(rotation=rotation_z(math.pi / 2))
        p = np.array([1, 0, 0])
        result = T.apply_point(p)
        np.testing.assert_array_almost_equal(result, [0, 1, 0], decimal=10)

    def test_to_from_matrix_roundtrip(self):
        T = Transform.from_rpy(0.3, -0.4, 0.7)
        T.translation = np.array([1.5, -2.3, 0.8])
        mat = T.to_matrix()
        T2 = Transform.from_matrix(mat)
        np.testing.assert_array_almost_equal(T.rotation, T2.rotation, decimal=12)
        np.testing.assert_array_almost_equal(T.translation, T2.translation, decimal=12)

    def test_quaternion_roundtrip(self):
        T = Transform.from_rpy(0.5, -0.3, 1.2)
        q = T.to_quaternion()
        T2 = Transform.from_quaternion(q)
        np.testing.assert_array_almost_equal(T.rotation, T2.rotation, decimal=10)

    def test_from_axis_angle(self):
        # 90 degrees about Z
        T = Transform.from_axis_angle(np.array([0, 0, 1]), math.pi / 2)
        p = np.array([1, 0, 0])
        result = T.apply_point(p)
        np.testing.assert_array_almost_equal(result, [0, 1, 0], decimal=10)

    def test_rpy_roundtrip(self):
        roll, pitch, yaw = 0.3, 0.5, -0.8
        T = Transform.from_rpy(roll, pitch, yaw)
        r2, p2, y2 = T.to_rpy()
        assert abs(r2 - roll) < 1e-10
        assert abs(p2 - pitch) < 1e-10
        assert abs(y2 - yaw) < 1e-10

    def test_matmul_operator(self):
        T1 = Transform.from_translation(np.array([1, 0, 0]))
        T2 = Transform.from_translation(np.array([0, 2, 0]))
        T3 = T1 @ T2
        np.testing.assert_array_almost_equal(T3.translation, [1, 2, 0])


class TestSkew:
    def test_skew_cross_product(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        cross = skew(a) @ b
        np.testing.assert_array_almost_equal(cross, np.cross(a, b))

    def test_unskew_roundtrip(self):
        v = np.array([1.5, -2.3, 0.8])
        result = unskew(skew(v))
        np.testing.assert_array_almost_equal(result, v)


class TestSpatialInertia:
    def test_zero_mass(self):
        I = SpatialInertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3)))
        mat = I.to_matrix()
        np.testing.assert_array_almost_equal(mat, np.zeros((6, 6)))

    def test_point_mass_at_origin(self):
        m = 2.0
        I = SpatialInertia(mass=m, com=np.zeros(3), inertia=np.zeros((3, 3)))
        mat = I.to_matrix()
        # Lower-right block should be m * I_3
        np.testing.assert_array_almost_equal(mat[3:, 3:], m * np.eye(3))
        # Upper-left block should be zero (no rotational inertia, CoM at origin)
        np.testing.assert_array_almost_equal(mat[:3, :3], np.zeros((3, 3)))

    def test_from_matrix_roundtrip(self):
        I_orig = SpatialInertia(
            mass=3.0,
            com=np.array([0.1, 0.2, 0.3]),
            inertia=np.diag([0.5, 0.6, 0.7]),
        )
        mat = I_orig.to_matrix()
        I_recovered = SpatialInertia.from_matrix(mat)
        assert abs(I_recovered.mass - I_orig.mass) < 1e-10
        np.testing.assert_array_almost_equal(I_recovered.com, I_orig.com, decimal=10)
        np.testing.assert_array_almost_equal(
            I_recovered.inertia, I_orig.inertia, decimal=10
        )

    def test_addition(self):
        I1 = SpatialInertia(mass=1.0, com=np.zeros(3), inertia=np.eye(3))
        I2 = SpatialInertia(mass=2.0, com=np.zeros(3), inertia=2 * np.eye(3))
        I_sum = I1 + I2
        assert abs(I_sum.mass - 3.0) < 1e-10

    def test_spatial_inertia_symmetry(self):
        I = SpatialInertia(
            mass=2.5,
            com=np.array([0.1, -0.05, 0.2]),
            inertia=np.array([[0.5, 0.01, 0.02], [0.01, 0.6, 0.03], [0.02, 0.03, 0.7]]),
        )
        mat = I.to_matrix()
        np.testing.assert_array_almost_equal(mat, mat.T, decimal=12)


class TestSpatialTransforms:
    def test_identity_transform_motion(self):
        X = spatial_transform_motion(Transform.identity())
        np.testing.assert_array_almost_equal(X, np.eye(6))

    def test_identity_transform_force(self):
        X = spatial_transform_force(Transform.identity())
        np.testing.assert_array_almost_equal(X, np.eye(6))

    def test_cross_motion_antisymmetric(self):
        v = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
        crm = spatial_cross_motion(v)
        # crm is not symmetric — but crm + crm^T should relate to crf
        crf = spatial_cross_force(v)
        np.testing.assert_array_almost_equal(crf, -crm.T)
