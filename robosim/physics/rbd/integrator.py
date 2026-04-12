"""Time integration schemes for rigid body dynamics."""

from __future__ import annotations

import numpy as np

from robosim.model.joint import JointType
from robosim.model.robot import Robot
from robosim.physics.rbd.algorithms import aba


def semi_implicit_euler(
    robot: Robot,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    dt: float,
    f_ext: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Semi-implicit (symplectic) Euler integration.

    1. Compute accelerations: qdd = ABA(q, qd, tau)
    2. Update velocities: qd_new = qd + qdd * dt
    3. Update positions: q_new = q + qd_new * dt  (use new velocity!)

    This is symplectic and provides much better energy behavior than
    explicit Euler for conservative systems.

    Parameters
    ----------
    robot : Robot model
    q : (n_dof,) current joint positions
    qd : (n_dof,) current joint velocities
    tau : (n_dof,) applied joint torques
    dt : time step
    f_ext : optional external spatial forces per link

    Returns
    -------
    q_new, qd_new : updated positions and velocities
    """
    # Add joint damping torques
    tau_total = tau.copy()
    for j_idx, joint in enumerate(robot.joints):
        if joint.num_dof > 0:
            dof_offset = robot._dof_index[j_idx]
            tau_total[dof_offset] -= joint.damping * qd[dof_offset]

    # Forward dynamics
    qdd = aba(robot, q, qd, tau_total, f_ext=f_ext)

    # Symplectic Euler: velocity first, then position with new velocity
    qd_new = qd + qdd * dt
    q_new = q + qd_new * dt

    # Enforce joint limits (clamp + zero velocity at limit)
    for j_idx, joint in enumerate(robot.joints):
        if joint.num_dof == 0 or joint.limits is None:
            continue
        if joint.joint_type in (JointType.CONTINUOUS,):
            continue  # no limits for continuous joints

        dof_offset = robot._dof_index[j_idx]
        if q_new[dof_offset] < joint.limits.lower:
            q_new[dof_offset] = joint.limits.lower
            qd_new[dof_offset] = max(0.0, qd_new[dof_offset])
        elif q_new[dof_offset] > joint.limits.upper:
            q_new[dof_offset] = joint.limits.upper
            qd_new[dof_offset] = min(0.0, qd_new[dof_offset])

    # Enforce mimic constraints
    if robot.has_mimic:
        # Temporarily set state so enforce_mimic works on q/qd arrays
        q_save, qd_save = robot._q, robot._qd
        robot._q, robot._qd = q_new, qd_new
        robot.enforce_mimic()
        q_new, qd_new = robot._q, robot._qd
        robot._q, robot._qd = q_save, qd_save

    return q_new, qd_new
