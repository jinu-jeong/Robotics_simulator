"""Runtime handles — returned by Scene.add(), provide access to physics state.

Each handle wraps the underlying physics objects but exposes a clean,
mode-independent interface.  Handles are *live*: properties like ``.com``
and ``.q`` always reflect the current simulation state.

Hierarchy::

    RobotHandle               — articulated robot (any mode)
    RigidBodyHandle           — free-floating rigid body
    FEMBodyHandle             — Hex8 FEM deformable body
    CBBodyHandle              — Craig-Bampton reduced-order body
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robosim.model.robot import Robot as RobotModel
    from robosim.physics.rbd.solver import RBDSolver
    from robosim.physics.fem.solver import FEMSolver, DeformableBody
    from robosim.physics.fem.reduced import CraigBamptonSolver, CraigBamptonBody
    from robosim.control.phases import Trajectory


# ══════════════════════════════════════════════════════════════════════════════
# Robot
# ══════════════════════════════════════════════════════════════════════════════

class RobotHandle:
    """Handle for an articulated robot in the scene.

    Properties
    ----------
    q   : (n_dof,) current joint positions
    dq  : (n_dof,) current joint velocities
    fk  : forward-kinematics — dict[link_name, Transform] or single Transform

    Methods
    -------
    set_target(q)  — update PD controller desired joint positions
    trajectory()   — create a phase-based Trajectory bound to this robot
    """

    def __init__(
        self,
        name:       str,
        model:      "RobotModel",
        solver:     "RBDSolver",
        controller  = None,   # JointPD
    ):
        self._name       = name
        self._model      = model
        self._solver     = solver
        self._controller = controller

    # ── identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_dof(self) -> int:
        return self._model.n_dof

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def q(self) -> np.ndarray:
        return self._model.q.copy()

    @property
    def dq(self) -> np.ndarray:
        return self._model.qd.copy()

    def fk(self, link_name: str | None = None):
        """Forward kinematics.

        Parameters
        ----------
        link_name : if given, return the single Transform for that link;
                    otherwise return a dict mapping all link names → Transform.
        """
        transforms = self._model.forward_kinematics()
        if link_name is None:
            return {self._model.links[i].name: transforms[i]
                    for i in range(len(transforms))}
        return transforms[self._model.link_index(link_name)]

    # ── control ───────────────────────────────────────────────────────────────

    def set_target(self, q) -> None:
        """Directly update the PD controller's target joint positions."""
        if self._controller is not None:
            self._controller.target_q = np.asarray(q, dtype=float)

    def trajectory(self) -> "Trajectory":
        """Create a phase-based Trajectory bound to this robot."""
        from robosim.control.phases import Trajectory
        return Trajectory(self)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        q_str  = np.array2string(self._model.q, precision=3, suppress_small=True)
        ctrl   = f"\n  controller : {self._controller!r}" if self._controller else ""
        return (f"RobotHandle '{self._name}'\n"
                f"  n_dof : {self.n_dof}\n"
                f"  q     : {q_str}"
                f"{ctrl}")


# ══════════════════════════════════════════════════════════════════════════════
# Rigid body
# ══════════════════════════════════════════════════════════════════════════════

class RigidBodyHandle:
    """Handle for a free-floating rigid body."""

    def __init__(self, name: str, model, solver):
        self._name   = name
        self._model  = model
        self._solver = solver

    @property
    def name(self) -> str:
        return self._name

    @property
    def com(self) -> np.ndarray:
        # For a free-floating 6-DOF box (create_free_box), q[:3] = [tx, ty, tz]
        # which directly gives the CoM world position.
        return self._model.q[:3].copy()

    @property
    def q(self) -> np.ndarray:
        return self._model.q.copy()

    @property
    def bottom_z(self) -> float:
        """Z-coordinate of the lowest point of the bounding box."""
        # The terminal link (links[-1]) carries the collision geometry.
        half_z = float(self._model.links[-1].collisions[0].geometry.size[2]) / 2.0
        return float(self.com[2]) - half_z

    def is_lifted(self, threshold: float = 0.05) -> bool:
        return self.bottom_z > threshold

    def __repr__(self) -> str:
        c = self.com
        return (f"RigidBodyHandle '{self._name}'\n"
                f"  com : ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# Deformable base
# ══════════════════════════════════════════════════════════════════════════════

class _DeformableHandle:
    """Common interface for FEM and CB body handles."""

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    # Overridden by subclasses
    @property
    def nodes(self) -> np.ndarray:
        raise NotImplementedError

    # Derived properties (same for FEM and CB)

    @property
    def com(self) -> np.ndarray:
        return self.nodes.mean(axis=0)

    @property
    def bottom_z(self) -> float:
        return float(self.nodes[:, 2].min())

    @property
    def extents(self) -> np.ndarray:
        """Axis-aligned bounding box extents (lx, ly, lz) in metres."""
        ns = self.nodes
        return ns.max(axis=0) - ns.min(axis=0)

    def is_lifted(self, threshold: float = 0.05) -> bool:
        return self.bottom_z > threshold


# ══════════════════════════════════════════════════════════════════════════════
# FEM body
# ══════════════════════════════════════════════════════════════════════════════

class FEMBodyHandle(_DeformableHandle):
    """Handle for an FEM (Hex8) deformable body.

    Extra properties vs rigid
    -------------------------
    nodes    : (N, 3) current node world positions
    velocity : (N, 3) current node velocities
    extents  : (3,)   AABB extents  [X, Y, Z]  in metres
    """

    def __init__(self, name: str, body: "DeformableBody", solver: "FEMSolver"):
        super().__init__(name)
        self._body   = body
        self._solver = solver

    @property
    def nodes(self) -> np.ndarray:
        return self._body.x.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._body.v.copy()

    def __repr__(self) -> str:
        if self._body.x is None:
            return f"FEMBodyHandle '{self._name}'  (not yet initialized)"
        ext = self.extents * 1e3
        c   = self.com
        return (f"FEMBodyHandle '{self._name}'\n"
                f"  nodes   : {self._body.x.shape[0]}\n"
                f"  com     : ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})\n"
                f"  extents : X={ext[0]:.1f}  Y={ext[1]:.1f}  Z={ext[2]:.1f} mm\n"
                f"  bottom_z: {self.bottom_z:+.4f} m")


# ══════════════════════════════════════════════════════════════════════════════
# CB body
# ══════════════════════════════════════════════════════════════════════════════

class CBBodyHandle(_DeformableHandle):
    """Handle for a Craig-Bampton reduced-order deformable body.

    Extra properties
    ----------------
    modal_coords : reduced modal coordinate vector
    """

    def __init__(self, name: str, body: "CraigBamptonBody", solver: "CraigBamptonSolver"):
        super().__init__(name)
        self._body   = body
        self._solver = solver

    @property
    def nodes(self) -> np.ndarray:
        return self._body.x.copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._body.v.copy()

    @property
    def modal_coords(self) -> np.ndarray:
        return self._body.q_r.copy()

    def __repr__(self) -> str:
        if self._body.x is None:
            return f"CBBodyHandle '{self._name}'  (not yet initialized)"
        ext = self.extents * 1e3
        c   = self.com
        n_modes = len(self._body.q_r) if self._body.q_r is not None else "?"
        return (f"CBBodyHandle '{self._name}'\n"
                f"  nodes        : {self._body.x.shape[0]}\n"
                f"  modal_coords : {n_modes}\n"
                f"  com          : ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})\n"
                f"  extents      : X={ext[0]:.1f}  Y={ext[1]:.1f}  Z={ext[2]:.1f} mm\n"
                f"  bottom_z     : {self.bottom_z:+.4f} m")
