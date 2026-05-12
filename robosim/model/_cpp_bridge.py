"""Bridge: build a :class:`robosim._cpp.kin.Topology` from a Python
:class:`Robot` and dispatch :meth:`Robot.forward_kinematics` through it
when the C++ extension is available.

The C++ path is a *strict* drop-in for the pure-Python reference —
same return shape (list of :class:`Transform`), same caching contract
(invalidated by ``q`` change via the snapshot compare in
:meth:`Robot.forward_kinematics`). Anything beyond FK still goes
through Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robosim.math.transforms import Transform
from robosim.model.joint import JointType

def _import_cpp_with_autobuild():
    """Probe ``robosim._cpp``; on first failure trigger the one-shot
    in-place build via :func:`robosim.ensure_cpp_built`. Subsequent
    calls are cheap (the helper caches its result)."""
    try:
        from robosim import _cpp
        return _cpp, True
    except ImportError:
        pass
    from robosim._build_helper import ensure_cpp_built
    if ensure_cpp_built(verbose=True):
        from robosim import _cpp
        return _cpp, True
    return None, False


_cpp, _CPP_LOADED = _import_cpp_with_autobuild()
HAVE_CPP = _CPP_LOADED and hasattr(_cpp, "kin")
HAVE_CPP_RBD = _CPP_LOADED and hasattr(_cpp, "rbd")

if TYPE_CHECKING:
    from robosim.model.robot import Robot


def _joint_kind(jt: JointType):
    """Map our :class:`JointType` enum to the C++ ``JointKind``."""
    if jt == JointType.FIXED:
        return _cpp.kin.JointKind.FIXED
    if jt in (JointType.REVOLUTE, JointType.CONTINUOUS):
        return _cpp.kin.JointKind.REVOLUTE
    if jt == JointType.PRISMATIC:
        return _cpp.kin.JointKind.PRISMATIC
    # Anything we don't know about (planar/floating/…): treat as fixed
    # — Python FK would fail on these too in the joint_transform call,
    # so the behaviour matches.
    return _cpp.kin.JointKind.FIXED


def build_topology(robot: "Robot"):
    """Construct + return a :class:`Topology` mirroring the Robot.

    Layout invariant the C++ FK relies on: joints are emitted in the
    same order Python stores them, which (as built by ``Robot.build``)
    already has parent links coming before children for any sane URDF.
    The roots (no inbound joint) are seeded to identity inside the
    C++ pass — no explicit traversal order is needed beyond
    parent-before-child.
    """
    if not HAVE_CPP:
        raise RuntimeError("robosim._cpp.kin not available")

    topo = _cpp.kin.Topology()
    topo.n_links = robot.n_links
    topo.n_dof = robot.n_dof

    js = []
    for j_idx, joint in enumerate(robot.joints):
        spec = _cpp.kin.JointSpec()
        spec.parent_link = robot._link_name_to_idx[joint.parent_link]
        spec.child_link = robot._link_name_to_idx[joint.child_link]
        spec.kind = _joint_kind(joint.joint_type)
        spec.dof_index = (robot._dof_index[j_idx]
                          if joint.num_dof > 0 else -1)
        # axis: store unit axis as float64 (3,)
        axis = np.asarray(joint.axis, dtype=np.float64).reshape(3)
        n2 = float(axis @ axis)
        if n2 > 0 and abs(n2 - 1.0) > 1e-9:
            axis = axis / np.sqrt(n2)
        spec.axis = axis
        spec.origin_R = np.ascontiguousarray(joint.origin.rotation,
                                             dtype=np.float64)
        spec.origin_t = np.ascontiguousarray(joint.origin.translation,
                                             dtype=np.float64)
        js.append(spec)

    topo.joints = js

    # Roots: every link that isn't pointed at by any joint.child_link.
    parents = set(robot._link_name_to_idx[j.child_link] for j in robot.joints)
    topo.root_links = [i for i in range(robot.n_links) if i not in parents]

    return topo


def build_rbd_topology(robot: "Robot"):
    """Construct an :class:`RbdTopology` (Topology + spatial inertias +
    tree order) for use by ABA / gravity_torques in C++."""
    if not HAVE_CPP_RBD:
        raise RuntimeError("robosim._cpp.rbd not available")

    topo = _cpp.rbd.RbdTopology()
    topo.n_links = robot.n_links
    topo.n_dof = robot.n_dof

    js = []
    for j_idx, joint in enumerate(robot.joints):
        spec = _cpp.kin.JointSpec()
        spec.parent_link = robot._link_name_to_idx[joint.parent_link]
        spec.child_link = robot._link_name_to_idx[joint.child_link]
        spec.kind = _joint_kind(joint.joint_type)
        spec.dof_index = (robot._dof_index[j_idx]
                          if joint.num_dof > 0 else -1)
        axis = np.asarray(joint.axis, dtype=np.float64).reshape(3)
        n2 = float(axis @ axis)
        if n2 > 0 and abs(n2 - 1.0) > 1e-9:
            axis = axis / np.sqrt(n2)
        spec.axis = axis
        spec.origin_R = np.ascontiguousarray(joint.origin.rotation,
                                             dtype=np.float64)
        spec.origin_t = np.ascontiguousarray(joint.origin.translation,
                                             dtype=np.float64)
        js.append(spec)
    topo.joints = js

    # Parent / joint-for-link tables, mirroring Robot._parent and
    # Robot._joint_for_link (built by Robot.build()).
    topo.parent_link    = list(robot._parent)
    topo.joint_for_link = list(robot._joint_for_link)

    # Tree order: BFS from roots, excluding roots. Matches
    # ``robosim/physics/rbd/algorithms.py::_build_tree_order``.
    from collections import deque
    order: list[int] = []
    queue = deque()
    for i in range(robot.n_links):
        if robot._parent[i] == -1:
            queue.append(i)
    while queue:
        link_idx = queue.popleft()
        for child_idx in robot._children[link_idx]:
            order.append(child_idx)
            queue.append(child_idx)
    topo.tree_order = order

    # Per-link 6×6 spatial inertias, packed row-major as (n_links, 36).
    inertias = np.zeros((robot.n_links, 36), dtype=np.float64)
    for l in range(robot.n_links):
        I6 = robot.links[l].inertial.to_matrix()
        inertias[l] = np.ascontiguousarray(I6).ravel()
    topo.link_inertias = inertias

    parents = set(robot._link_name_to_idx[j.child_link]
                  for j in robot.joints)
    topo.root_links = [i for i in range(robot.n_links) if i not in parents]
    return topo


def aba_via_cpp(robot: "Robot", topo, q, qd, tau, gravity, f_ext=None):
    """ABA: returns qdd (n_dof,). Inputs are contiguous float64 numpy.

    ``f_ext`` mirrors the Python signature — a dict ``{link_idx: (6,)
    wrench}``. We densify it into a (n_links*6,) row-major buffer for
    the C++ kernel. Length-0 buffer = no external forces.
    """
    g = np.ascontiguousarray(gravity, dtype=np.float64).reshape(3)
    q  = np.ascontiguousarray(q,  dtype=np.float64)
    qd = np.ascontiguousarray(qd, dtype=np.float64)
    tau = np.ascontiguousarray(tau, dtype=np.float64)

    if f_ext:
        f_flat = np.zeros(robot.n_links * 6, dtype=np.float64)
        for li, w in f_ext.items():
            f_flat[li * 6 : li * 6 + 6] = np.asarray(w, dtype=np.float64).reshape(6)
    else:
        f_flat = np.empty(0, dtype=np.float64)
    return _cpp.rbd.aba(topo, q, qd, tau, g, f_flat)


def gravity_torques_via_cpp(robot: "Robot", topo, q, gravity):
    g = np.ascontiguousarray(gravity, dtype=np.float64).reshape(3)
    q = np.ascontiguousarray(q, dtype=np.float64)
    return _cpp.rbd.gravity_torques(topo, q, g)


def forward_kinematics_via_cpp(robot: "Robot", topo) -> list[Transform]:
    """Run the C++ FK kernel and wrap the (R, t) pair per link back
    into Python :class:`Transform` objects."""
    q = np.ascontiguousarray(robot.q, dtype=np.float64)
    R_flat, t_arr = _cpp.kin.forward_kinematics(topo, q)
    # R_flat: (n_links, 9) row-major flatten of each 3×3
    # t_arr:  (n_links, 3)
    out: list[Transform] = []
    for l in range(robot.n_links):
        R = R_flat[l].reshape(3, 3)
        # Cheap path: skip Transform.__post_init__ validation.
        out.append(Transform._fast(np.ascontiguousarray(R),
                                   np.ascontiguousarray(t_arr[l])))
    return out
