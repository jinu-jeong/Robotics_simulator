"""Spatial algebra for rigid body dynamics (Featherstone conventions).

Spatial vectors are 6D vectors in Plucker coordinates:
  - Motion vectors (twists):  [angular(3); linear(3)]
  - Force vectors (wrenches): [torque(3); force(3)]

Spatial inertia is a 6x6 SPD matrix relating motion to momentum.

Reference: Roy Featherstone, "Rigid Body Dynamics Algorithms", Springer 2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robosim.math.transforms import Transform, skew


@dataclass
class SpatialInertia:
    """6x6 spatial inertia of a rigid body.

    Defined by mass, center of mass (in body frame), and rotational
    inertia about the center of mass.
    """

    mass: float
    com: np.ndarray  # (3,) center of mass in body frame
    inertia: np.ndarray  # (3,3) rotational inertia about CoM, in body frame

    def __post_init__(self):
        self.com = np.asarray(self.com, dtype=np.float64).reshape(3)
        self.inertia = np.asarray(self.inertia, dtype=np.float64).reshape(3, 3)
        self._cached_matrix: np.ndarray | None = None

    def to_matrix(self) -> np.ndarray:
        """Convert to 6x6 spatial inertia matrix about the body frame origin.

        Using the parallel axis theorem (generalized):
        I_spatial = | I_rot + m*[c]x*[c]x^T   m*[c]x |
                    | m*[c]x^T                 m*I_3  |

        where [c]x is the skew-symmetric matrix of the CoM vector.
        Cached after first call (SpatialInertia is conceptually immutable).
        """
        cached = getattr(self, "_cached_matrix", None)
        if cached is not None:
            return cached

        m = self.mass
        c = self.com
        cx = skew(c)
        I_rot = self.inertia

        I_spatial = np.zeros((6, 6))
        I_spatial[:3, :3] = I_rot + m * (cx @ cx.T)
        I_spatial[:3, 3:] = m * cx
        I_spatial[3:, :3] = m * cx.T
        I_spatial[3:, 3:] = m * np.eye(3)
        self._cached_matrix = I_spatial
        return I_spatial

    @staticmethod
    def from_matrix(I_sp: np.ndarray) -> SpatialInertia:
        """Extract mass, CoM, and inertia from a 6x6 spatial inertia matrix."""
        m = I_sp[3, 3]
        if m < 1e-15:
            return SpatialInertia(mass=0.0, com=np.zeros(3), inertia=np.zeros((3, 3)))
        cx = I_sp[:3, 3:] / m
        c = np.array([cx[2, 1], cx[0, 2], cx[1, 0]])
        I_rot = I_sp[:3, :3] - m * (cx @ cx.T)
        return SpatialInertia(mass=m, com=c, inertia=I_rot)

    def __add__(self, other: SpatialInertia) -> SpatialInertia:
        """Add two spatial inertias (combine bodies at the same frame)."""
        result_mat = self.to_matrix() + other.to_matrix()
        return SpatialInertia.from_matrix(result_mat)

    def __repr__(self) -> str:
        return (
            f"SpatialInertia(mass={self.mass:.4f}, "
            f"com=[{self.com[0]:.4f}, {self.com[1]:.4f}, {self.com[2]:.4f}])"
        )


def spatial_transform_force(T: Transform) -> np.ndarray:
    """6x6 spatial force transform matrix X* for transform T.

    Maps a spatial force from frame B to frame A, where T: A <- B.
    X* = | R     [t]x @ R |
         | 0     R        |
    """
    R = T.rotation
    tx = skew(T.translation)
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[:3, 3:] = tx @ R
    X[3:, 3:] = R
    return X


def spatial_transform_motion(T: Transform) -> np.ndarray:
    """6x6 spatial motion transform matrix X for transform T.

    Maps a spatial motion vector from frame B to frame A, where T: A <- B.
    X = | R        0 |
        | [t]x @ R R |
    """
    R = T.rotation
    tx = skew(T.translation)
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, :3] = tx @ R
    X[3:, 3:] = R
    return X


def spatial_cross_motion(v: np.ndarray) -> np.ndarray:
    """Spatial cross product operator for motion vectors: [v]x.

    Inlined skew construction to avoid two allocations + reshape per call.
    """
    w0, w1, w2 = v[0], v[1], v[2]
    l0, l1, l2 = v[3], v[4], v[5]
    X = np.zeros((6, 6))
    # [omega]x in upper-left and lower-right
    X[0, 1] = -w2; X[0, 2] =  w1
    X[1, 0] =  w2; X[1, 2] = -w0
    X[2, 0] = -w1; X[2, 1] =  w0
    X[3, 4] = -w2; X[3, 5] =  w1
    X[4, 3] =  w2; X[4, 5] = -w0
    X[5, 3] = -w1; X[5, 4] =  w0
    # [v_lin]x in lower-left
    X[3, 1] = -l2; X[3, 2] =  l1
    X[4, 0] =  l2; X[4, 2] = -l0
    X[5, 0] = -l1; X[5, 1] =  l0
    return X


def spatial_cross_force(v: np.ndarray) -> np.ndarray:
    """Spatial cross product operator for force vectors: [v]x*.

    crf(v) @ f = v x_f f (spatial force cross product).
    crf(v) = -crm(v)^T
    """
    return -spatial_cross_motion(v).T


def motion_subspace_revolute(axis: np.ndarray) -> np.ndarray:
    """Joint motion subspace S for a revolute joint.

    Returns a (6, 1) matrix: [axis; 0].
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    S = np.zeros((6, 1))
    S[:3, 0] = axis
    return S


def motion_subspace_prismatic(axis: np.ndarray) -> np.ndarray:
    """Joint motion subspace S for a prismatic joint.

    Returns a (6, 1) matrix: [0; axis].
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    S = np.zeros((6, 1))
    S[3:, 0] = axis
    return S


def joint_transform(joint_type: str, axis: np.ndarray, q: float) -> Transform:
    """Compute the transform across a joint given its configuration.

    Parameters
    ----------
    joint_type : one of 'revolute', 'prismatic', 'fixed'
    axis : (3,) joint axis in local frame
    q : joint position (angle in rad for revolute, displacement for prismatic)
    """
    if joint_type == "fixed":
        return Transform.identity()
    elif joint_type == "revolute":
        return Transform.from_axis_angle(axis, q)
    elif joint_type == "prismatic":
        return Transform.from_translation(axis * q)
    else:
        raise ValueError(f"Unknown joint type: {joint_type}")
