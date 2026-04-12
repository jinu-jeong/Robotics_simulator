"""SE(3) rigid body transforms: rotation + translation.

Conventions:
- Rotation matrices are (3,3) in SO(3)
- Quaternions are [w, x, y, z] (scalar-first)
- Homogeneous matrices are (4,4)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Transform:
    """SE(3) rigid body transform: rotation + translation.

    Represents the transform T that maps a point p_body in the body frame
    to the world frame: p_world = R @ p_body + t
    """

    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)

    @staticmethod
    def identity() -> Transform:
        return Transform()

    @staticmethod
    def from_matrix(mat: np.ndarray) -> Transform:
        """Create from a (4,4) homogeneous matrix."""
        mat = np.asarray(mat, dtype=np.float64)
        return Transform(rotation=mat[:3, :3].copy(), translation=mat[:3, 3].copy())

    @staticmethod
    def from_rotation(R: np.ndarray) -> Transform:
        return Transform(rotation=R)

    @staticmethod
    def from_translation(t: np.ndarray) -> Transform:
        return Transform(translation=t)

    @staticmethod
    def from_rpy(roll: float, pitch: float, yaw: float) -> Transform:
        """Create from roll-pitch-yaw (XYZ extrinsic = ZYX intrinsic)."""
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ],
        ])
        return Transform(rotation=R)

    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> Transform:
        """Create from axis-angle rotation (Rodrigues' formula)."""
        axis = np.asarray(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return Transform.identity()
        axis = axis / norm
        K = skew(axis)
        R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
        return Transform(rotation=R)

    @staticmethod
    def from_quaternion(q: np.ndarray) -> Transform:
        """Create from quaternion [w, x, y, z]."""
        q = np.asarray(q, dtype=np.float64)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])
        return Transform(rotation=R)

    def to_matrix(self) -> np.ndarray:
        """Convert to (4,4) homogeneous matrix."""
        mat = np.eye(4)
        mat[:3, :3] = self.rotation
        mat[:3, 3] = self.translation
        return mat

    def to_quaternion(self) -> np.ndarray:
        """Extract quaternion [w, x, y, z] from rotation matrix."""
        R = self.rotation
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def to_rpy(self) -> tuple[float, float, float]:
        """Extract roll-pitch-yaw angles."""
        R = self.rotation
        pitch = math.atan2(-R[2, 0], math.sqrt(R[0, 0]**2 + R[1, 0]**2))

        if abs(math.cos(pitch)) > 1e-10:
            roll = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            yaw = 0.0

        return roll, pitch, yaw

    def compose(self, other: Transform) -> Transform:
        """Compose transforms: self * other (apply other first, then self)."""
        R = self.rotation @ other.rotation
        t = self.rotation @ other.translation + self.translation
        return Transform(rotation=R, translation=t)

    def __matmul__(self, other: Transform) -> Transform:
        """T1 @ T2 = compose."""
        return self.compose(other)

    def inverse(self) -> Transform:
        """Compute the inverse transform."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return Transform(rotation=R_inv, translation=t_inv)

    def apply_point(self, p: np.ndarray) -> np.ndarray:
        """Transform a point: R @ p + t."""
        return self.rotation @ np.asarray(p) + self.translation

    def apply_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate a vector (translation-invariant): R @ v."""
        return self.rotation @ np.asarray(v)

    def __repr__(self) -> str:
        r, p, y = self.to_rpy()
        t = self.translation
        return (
            f"Transform(rpy=[{math.degrees(r):.1f}, {math.degrees(p):.1f}, "
            f"{math.degrees(y):.1f}]°, t=[{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}])"
        )


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from a 3-vector. [v]_x such that [v]_x @ w = v x w."""
    v = np.asarray(v, dtype=np.float64)
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


def unskew(S: np.ndarray) -> np.ndarray:
    """Extract 3-vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rotation_x(angle: float) -> np.ndarray:
    """Rotation matrix about the X axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle: float) -> np.ndarray:
    """Rotation matrix about the Y axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_z(angle: float) -> np.ndarray:
    """Rotation matrix about the Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
