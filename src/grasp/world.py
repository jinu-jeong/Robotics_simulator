"""World placement of the two fingers, the object and the table.

Finger local frame is the project convention (x = length, y = width, z = up).
The jaws open along gripper ``y``: the left inner face sits at ``y = −g/2``,
the right inner face at ``y = +g/2``. The right finger is a mirror of the left
through the plane ``y = 0``.

When ``T_ee`` is omitted the jaws live in world coordinates with a scalar
``+z`` lift (Stages A / B). Stage C passes the flange pose so the same
offsets are expressed in the EE frame.
"""

from __future__ import annotations

import numpy as np

from ..geometry.finger import FingerGeometry


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4×4 pose to ``(N, 3)`` points."""
    pts = np.asarray(pts, float).reshape(-1, 3)
    return pts @ T[:3, :3].T + T[:3, 3]


def transform_vectors(T: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """Rotate ``(N, 3)`` vectors by the linear part of ``T``."""
    vecs = np.asarray(vecs, float).reshape(-1, 3)
    return vecs @ T[:3, :3].T


class ParallelJawWorld:
    def __init__(self, geometry: FingerGeometry) -> None:
        self.g = geometry
        self.W = float(geometry.width)

    def jaw_ee(self, nodes_local: np.ndarray, opening: float, side: str) -> np.ndarray:
        """Finger nodes in the gripper EE frame (opening along ``y``)."""
        n = np.asarray(nodes_local, float).copy()
        if side == "left":
            n[:, 1] = n[:, 1] - self.W - 0.5 * opening
        elif side == "right":
            n[:, 1] = -n[:, 1] + self.W + 0.5 * opening
        else:
            raise ValueError(f"side must be left/right, got {side!r}")
        return n

    def left_nodes(
        self,
        nodes_local: np.ndarray,
        opening: float,
        lift: float = 0.0,
        T_ee: np.ndarray | None = None,
    ) -> np.ndarray:
        n = self.jaw_ee(nodes_local, opening, "left")
        if T_ee is None:
            n[:, 2] = n[:, 2] + lift
            return n
        return transform_points(T_ee, n)

    def right_nodes(
        self,
        nodes_local: np.ndarray,
        opening: float,
        lift: float = 0.0,
        T_ee: np.ndarray | None = None,
    ) -> np.ndarray:
        n = self.jaw_ee(nodes_local, opening, "right")
        if T_ee is None:
            n[:, 2] = n[:, 2] + lift
            return n
        return transform_points(T_ee, n)

    @staticmethod
    def left_disp(u_local: np.ndarray, T_ee: np.ndarray | None = None) -> np.ndarray:
        u = np.asarray(u_local, float).copy()
        return u if T_ee is None else transform_vectors(T_ee, u)

    @staticmethod
    def right_disp(u_local: np.ndarray, T_ee: np.ndarray | None = None) -> np.ndarray:
        u = np.asarray(u_local, float).copy()
        u[:, 1] *= -1.0
        return u if T_ee is None else transform_vectors(T_ee, u)

    def left_contact_world(
        self,
        contact_local: np.ndarray,
        opening: float,
        lift: float = 0.0,
        T_ee: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.left_nodes(np.asarray(contact_local, float).reshape(1, 3), opening, lift, T_ee)[0]

    def right_contact_world(
        self,
        contact_local: np.ndarray,
        opening: float,
        lift: float = 0.0,
        T_ee: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.right_nodes(np.asarray(contact_local, float).reshape(1, 3), opening, lift, T_ee)[0]
