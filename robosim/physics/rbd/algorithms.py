"""Featherstone's algorithms for articulated rigid body dynamics.

Implements:
- RNEA: Recursive Newton-Euler Algorithm (inverse dynamics)
- ABA:  Articulated Body Algorithm (forward dynamics)
- CRBA: Composite Rigid Body Algorithm (mass matrix)

Reference: Roy Featherstone, "Rigid Body Dynamics Algorithms", Springer 2008.

Convention: spatial vectors are 6D [angular(3); linear(3)] (Featherstone).
Link 0 is root. Parent of root is the "world" (index -1).
"""

from __future__ import annotations

import numpy as np

from robosim.math.spatial import (
    SpatialInertia,
    spatial_cross_force,
    spatial_cross_motion,
    joint_transform,
    motion_subspace_revolute,
    motion_subspace_prismatic,
)
from robosim.math.transforms import Transform
from robosim.model.joint import JointType
from robosim.model.robot import Robot


def _get_motion_subspace(robot: Robot, j_idx: int) -> np.ndarray:
    """Get 6xn motion subspace for joint j_idx."""
    return robot.joint_motion_subspace(j_idx)


def _spatial_transform_matrix(T: Transform) -> np.ndarray:
    """Compute the 6x6 spatial motion transform X such that v_A = X @ v_B.

    Inlined skew product for the (3:6, 0:3) block to avoid a separate
    skew() allocation + matmul (this is the hottest matrix builder in
    ABA/RNEA, called O(n_links) times per substep).
    """
    R = T.rotation
    p = T.translation
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    # px @ R where px = [[0,-p2,p1],[p2,0,-p0],[-p1,p0,0]]
    p0, p1, p2 = p[0], p[1], p[2]
    R0 = R[0]; R1 = R[1]; R2 = R[2]
    X[3, :3] = -p2 * R1 + p1 * R2
    X[4, :3] =  p2 * R0 - p0 * R2
    X[5, :3] = -p1 * R0 + p0 * R1
    return X


def _spatial_transform_force_matrix(T: Transform) -> np.ndarray:
    """Compute the 6x6 spatial force transform X* = X^{-T}.

    X* = | R     [p]x R |
         | 0     R      |
    """
    R = T.rotation
    p = T.translation
    from robosim.math.transforms import skew
    px = skew(p)
    Xf = np.zeros((6, 6))
    Xf[:3, :3] = R
    Xf[:3, 3:] = px @ R
    Xf[3:, 3:] = R
    return Xf


def _build_tree_order(robot: Robot) -> list[int]:
    """Get link indices in parent-first (BFS) order, excluding root."""
    from collections import deque
    order = []
    queue = deque()
    # Find root(s)
    for i in range(robot.n_links):
        if robot.parent_index(i) == -1:
            queue.append(i)
    while queue:
        link_idx = queue.popleft()
        for child_idx in robot.children_indices(link_idx):
            order.append(child_idx)
            queue.append(child_idx)
    return order


def rnea(
    robot: Robot,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    f_ext: dict[int, np.ndarray] | None = None,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Recursive Newton-Euler Algorithm — Inverse Dynamics.

    Computes joint torques tau given positions, velocities, accelerations.
    tau = M(q) @ qdd + C(q, qd) - J^T @ f_ext

    Parameters
    ----------
    robot : Robot model
    q : (n_dof,) joint positions
    qd : (n_dof,) joint velocities
    qdd : (n_dof,) joint accelerations
    f_ext : optional dict mapping link_idx -> (6,) spatial force in link frame
    gravity : (3,) gravity vector (defaults to robot.gravity)

    Returns
    -------
    tau : (n_dof,) joint torques
    """
    n_links = robot.n_links
    n_dof = robot.n_dof

    if gravity is None:
        gravity = robot.gravity

    # Save and set state
    old_q, old_qd = robot.q.copy(), robot.qd.copy()
    robot.q = q.copy()
    robot.qd = qd.copy()

    # Allocate
    v = [np.zeros(6) for _ in range(n_links)]       # spatial velocity
    a = [np.zeros(6) for _ in range(n_links)]       # spatial acceleration
    f = [np.zeros(6) for _ in range(n_links)]       # spatial force
    X_parent = [np.eye(6) for _ in range(n_links)]  # transform to parent
    I_link = [np.zeros((6, 6)) for _ in range(n_links)]  # spatial inertia

    # Compute spatial inertias
    for i in range(n_links):
        I_link[i] = robot.links[i].inertial.to_matrix()

    # Gravity as base acceleration (a_0 = -gravity in spatial coords)
    a_grav = np.zeros(6)
    a_grav[3:] = -gravity

    order = _build_tree_order(robot)

    # Initialize root links with gravity acceleration
    for i in range(n_links):
        if robot.parent_index(i) == -1:
            a[i] = a_grav.copy()

    # --- Pass 1: Forward recursion (compute velocities and accelerations) ---
    Xs = robot.joint_spatial_transforms()
    for i in order:
        parent_idx = robot.parent_index(i)
        j_idx = robot.joint_index_for_link(i)

        Xi = Xs[j_idx]
        X_parent[i] = Xi

        # Motion subspace and joint velocity
        S = _get_motion_subspace(robot, j_idx)
        qi_d = robot.get_joint_qd(j_idx)

        # Velocity: v_i = X_i @ v_parent + S * qd
        v[i] = Xi @ v[parent_idx]
        if S.shape[1] > 0:
            v[i] += S.ravel() * qi_d

        # Acceleration: a_i = X_i @ a_parent + S * qdd + v_i x S * qd
        dof_offset = robot._dof_index[j_idx]
        a[i] = Xi @ a[parent_idx]
        if S.shape[1] > 0:
            qdd_i = qdd[dof_offset]
            a[i] += S.ravel() * qdd_i
            a[i] += spatial_cross_motion(v[i]) @ (S.ravel() * qi_d)

    # --- Pass 2: Backward recursion (compute forces and torques) ---
    for i in reversed(order):
        f[i] = I_link[i] @ a[i] + spatial_cross_force(v[i]) @ (I_link[i] @ v[i])

        # Subtract external forces
        if f_ext and i in f_ext:
            f[i] -= f_ext[i]

    # Propagate forces to parents
    for i in reversed(order):
        parent_idx = robot.parent_index(i)
        if parent_idx >= 0:
            f[parent_idx] += X_parent[i].T @ f[i]

    # Extract torques
    tau = np.zeros(n_dof)
    for i in order:
        j_idx = robot.joint_index_for_link(i)
        S = _get_motion_subspace(robot, j_idx)
        if S.shape[1] > 0:
            dof_offset = robot._dof_index[j_idx]
            tau[dof_offset] = S.ravel() @ f[i]

    # Restore state
    robot.q = old_q
    robot.qd = old_qd
    return tau


def aba(
    robot: Robot,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    f_ext: dict[int, np.ndarray] | None = None,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Articulated Body Algorithm — Forward Dynamics.

    Computes joint accelerations qdd given positions, velocities, torques.
    qdd = M(q)^{-1} @ (tau - C(q, qd))

    Parameters
    ----------
    robot : Robot model
    q : (n_dof,) joint positions
    qd : (n_dof,) joint velocities
    tau : (n_dof,) joint torques
    f_ext : optional dict mapping link_idx -> (6,) spatial force in link frame
    gravity : (3,) gravity vector (defaults to robot.gravity)

    Returns
    -------
    qdd : (n_dof,) joint accelerations
    """
    n_links = robot.n_links
    n_dof = robot.n_dof

    if gravity is None:
        gravity = robot.gravity

    old_q, old_qd = robot.q.copy(), robot.qd.copy()
    robot.q = q.copy()
    robot.qd = qd.copy()

    # Allocate
    v = [np.zeros(6) for _ in range(n_links)]
    c = [np.zeros(6) for _ in range(n_links)]       # velocity-dependent bias
    I_A = [np.zeros((6, 6)) for _ in range(n_links)]  # articulated inertia
    p_A = [np.zeros(6) for _ in range(n_links)]       # articulated bias force
    X_parent = [np.eye(6) for _ in range(n_links)]
    S_list = [np.zeros((6, 0)) for _ in range(n_links)]
    # For articulated body quantities per joint
    U = [np.zeros(6) for _ in range(n_links)]
    d = [0.0 for _ in range(n_links)]
    u = [0.0 for _ in range(n_links)]

    a_grav = np.zeros(6)
    a_grav[3:] = -gravity

    order = _build_tree_order(robot)

    # Compute spatial inertias
    for i in range(n_links):
        I_A[i] = robot.links[i].inertial.to_matrix().copy()

    # --- Pass 1: Forward recursion (velocities, bias forces) ---
    Xs = robot.joint_spatial_transforms()
    for i in order:
        parent_idx = robot.parent_index(i)
        j_idx = robot.joint_index_for_link(i)

        Xi = Xs[j_idx]
        X_parent[i] = Xi

        S = _get_motion_subspace(robot, j_idx)
        S_list[i] = S

        v[i] = Xi @ v[parent_idx]
        qi_d = robot.get_joint_qd(j_idx)
        if S.shape[1] > 0:
            v[i] += S.ravel() * qi_d

        # Velocity product: c_i = v_i x S * qd (Coriolis)
        c[i] = np.zeros(6)
        if S.shape[1] > 0:
            c[i] = spatial_cross_motion(v[i]) @ (S.ravel() * qi_d)

        # Bias force: p_A = v x* (I * v)
        p_A[i] = spatial_cross_force(v[i]) @ (I_A[i] @ v[i])

        # Subtract external forces
        if f_ext and i in f_ext:
            p_A[i] -= f_ext[i]

    # --- Pass 2: Backward recursion (articulated inertias) ---
    for i in reversed(order):
        j_idx = robot.joint_index_for_link(i)
        S = S_list[i]

        if S.shape[1] > 0:
            s = S.ravel()
            U[i] = I_A[i] @ s
            d[i] = float(s @ U[i])

            dof_offset = robot._dof_index[j_idx]
            u[i] = float(tau[dof_offset]) - float(s @ p_A[i])

            if abs(d[i]) > 1e-15:
                I_a = I_A[i] - np.outer(U[i], U[i]) / d[i]
                p_a = p_A[i] + I_a @ c[i] + U[i] * (u[i] / d[i])
            else:
                I_a = I_A[i].copy()
                p_a = p_A[i] + I_a @ c[i]
        else:
            I_a = I_A[i].copy()
            p_a = p_A[i] + I_a @ c[i]

        parent_idx = robot.parent_index(i)
        if parent_idx >= 0:
            Xp = X_parent[i]
            I_A[parent_idx] += Xp.T @ I_a @ Xp
            p_A[parent_idx] += Xp.T @ p_a

    # --- Pass 3: Forward recursion (accelerations) ---
    a = [np.zeros(6) for _ in range(n_links)]
    qdd = np.zeros(n_dof)

    # Initialize root links with gravity acceleration
    for i in range(n_links):
        if robot.parent_index(i) == -1:
            a[i] = a_grav.copy()

    for i in order:
        parent_idx = robot.parent_index(i)
        j_idx = robot.joint_index_for_link(i)
        S = S_list[i]

        a_prime = X_parent[i] @ a[parent_idx] + c[i]

        if S.shape[1] > 0 and abs(d[i]) > 1e-15:
            s = S.ravel()
            dof_offset = robot._dof_index[j_idx]
            qdd_i = (u[i] - float(U[i] @ a_prime)) / d[i]
            qdd[dof_offset] = qdd_i
            a[i] = a_prime + s * qdd_i
        else:
            a[i] = a_prime

    robot.q = old_q
    robot.qd = old_qd
    return qdd


def crba(
    robot: Robot,
    q: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Composite Rigid Body Algorithm — Mass Matrix.

    Computes the joint-space mass matrix M(q).

    Parameters
    ----------
    robot : Robot model
    q : (n_dof,) joint positions

    Returns
    -------
    M : (n_dof, n_dof) symmetric positive definite mass matrix
    """
    n_links = robot.n_links
    n_dof = robot.n_dof

    old_q = robot.q.copy()
    robot.q = q.copy()

    I_C = [np.zeros((6, 6)) for _ in range(n_links)]
    X_parent = [np.eye(6) for _ in range(n_links)]

    order = _build_tree_order(robot)

    for i in range(n_links):
        I_C[i] = robot.links[i].inertial.to_matrix().copy()

    Xs = robot.joint_spatial_transforms()
    for i in order:
        j_idx = robot.joint_index_for_link(i)
        X_parent[i] = Xs[j_idx]

    # Backward pass: accumulate composite inertias
    for i in reversed(order):
        parent_idx = robot.parent_index(i)
        if parent_idx >= 0:
            Xp = X_parent[i]
            I_C[parent_idx] += Xp.T @ I_C[i] @ Xp

    # Compute mass matrix
    M = np.zeros((n_dof, n_dof))

    for i in order:
        j_idx = robot.joint_index_for_link(i)
        S_i = _get_motion_subspace(robot, j_idx)
        if S_i.shape[1] == 0:
            continue

        dof_i = robot._dof_index[j_idx]
        s_i = S_i.ravel()  # (6,)
        F = I_C[i] @ s_i   # (6,)

        # Diagonal block
        M[dof_i, dof_i] = float(s_i @ F)

        # Off-diagonal: walk up to root
        j = i
        while True:
            parent_idx = robot.parent_index(j)
            if parent_idx < 0:
                break
            F = X_parent[j].T @ F
            j = parent_idx

            j_idx_p = robot.joint_index_for_link(j)
            if j_idx_p < 0:
                break
            S_j = _get_motion_subspace(robot, j_idx_p)
            if S_j.shape[1] == 0:
                continue

            dof_j = robot._dof_index[j_idx_p]
            val = float(S_j.ravel() @ F)
            M[dof_i, dof_j] = val
            M[dof_j, dof_i] = val

    robot.q = old_q
    return M


def gravity_torques(
    robot: Robot,
    q: np.ndarray,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute gravity compensation torques: tau_g = RNEA(q, 0, 0)."""
    n_dof = robot.n_dof
    return rnea(robot, q, np.zeros(n_dof), np.zeros(n_dof), gravity=gravity)
