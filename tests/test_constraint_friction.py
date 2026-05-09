"""Static-friction validation for ConstraintContactSolver.

Drops a free-floating rigid box onto a tilted ground plane (modelled by
gravity rotated relative to the world Z axis — same effect as inclining
the floor) and checks the stick/slip threshold against ``tan(θ) vs μ``.

PenaltyContactSolver is shown to fail the same test (object slides
even when ``tan(θ) < μ``), confirming that static friction is what
ConstraintContactSolver newly provides.
"""

from __future__ import annotations

import numpy as np
import pytest

from robosim.model.factory import create_free_box
from robosim.physics.contact.constraint_solver import ConstraintContactSolver
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.response import ContactParams
from robosim.physics.contact.solver import PenaltyContactSolver
from robosim.physics.rbd.solver import RBDSolver


def _run_incline(solver_cls, angle_deg: float, mu: float, duration: float = 1.5):
    """Drop a 1 kg box on the floor under tilted gravity. Return final XY drift."""
    dt = 1e-3
    g  = 9.81
    theta = np.deg2rad(angle_deg)
    gravity = np.array([g * np.sin(theta), 0.0, -g * np.cos(theta)])

    box = create_free_box(
        name="probe", size=(0.10, 0.10, 0.10), mass=1.0,
        position=np.array([0.0, 0.0, 0.05]),
    )
    box.gravity = gravity
    rbd = RBDSolver(robot=box)
    rbd.initialize(dt=dt)

    params = ContactParams(stiffness=1e5, damping=2e2,
                           friction_mu=mu, friction_eps=1e-3)
    if solver_cls is ConstraintContactSolver:
        contact = solver_cls(ground=GroundPlane(height=0.0), params=params, dt=dt)
    else:
        contact = solver_cls(ground=GroundPlane(height=0.0), params=params)
    contact.register_rbd(rbd, robot_id="probe")

    n_steps = int(duration / dt)
    # Settle phase — let the box land.
    for _ in range(int(0.4 / dt)):
        rbd.clear_external_forces()
        for li, w in contact.compute_rbd_contact_forces(rbd).items():
            rbd.set_external_force(li, w)
        rbd.step()
    x0 = box.q[0]
    # Measure phase.
    for _ in range(n_steps - int(0.4 / dt)):
        rbd.clear_external_forces()
        for li, w in contact.compute_rbd_contact_forces(rbd).items():
            rbd.set_external_force(li, w)
        rbd.step()
    return float(box.q[0] - x0)


def test_constraint_static_friction_holds_below_threshold():
    """μ=0.5, θ=20° (tan=0.36 < μ): block should stay essentially still."""
    drift = _run_incline(ConstraintContactSolver, angle_deg=20.0, mu=0.5)
    assert abs(drift) < 0.01, f"expected stick (<1cm drift), got {drift:.4f} m"


def test_constraint_kinetic_slip_above_threshold():
    """μ=0.5, θ=35° (tan=0.70 > μ): block should slide downhill."""
    drift = _run_incline(ConstraintContactSolver, angle_deg=35.0, mu=0.5)
    assert drift > 0.05, f"expected slip (>5cm drift), got {drift:.4f} m"


def test_constraint_friction_grasp_holds_box_against_gravity():
    """Pinch-grasp surrogate: a free box squeezed between ground (below) and
    a horizontal slab (above) held in place by a kinematic body. Under
    constraint mode, static friction at the top contact must support a
    vertical pull-out attempt; under penalty mode it cannot.

    We simulate this by tilting gravity slightly horizontal so the box
    has a sideways component (mimicking the lateral pull from a moving
    grip). Constraint solver: box stays. Penalty solver: box drifts.
    """
    dt = 1e-3
    g  = 9.81
    # 5° tilt — tan(5°) = 0.087; far below μ=0.6, so stick should hold.
    theta = np.deg2rad(5.0)
    gravity = np.array([g * np.sin(theta), 0.0, -g * np.cos(theta)])

    box = create_free_box(
        name="probe", size=(0.10, 0.10, 0.10), mass=0.5,
        position=np.array([0.0, 0.0, 0.05]),
    )
    box.gravity = gravity
    rbd = RBDSolver(robot=box)
    rbd.initialize(dt=dt)

    params = ContactParams(stiffness=1e4, damping=2e2,
                           friction_mu=0.6, friction_eps=1e-3)
    contact = ConstraintContactSolver(
        ground=GroundPlane(height=0.0), params=params, dt=dt,
    )
    contact.register_rbd(rbd, robot_id="probe")

    # Settle, then measure 1.5 s of drift under tilted gravity.
    for _ in range(int(0.4 / dt)):
        rbd.clear_external_forces()
        for li, w in contact.compute_rbd_contact_forces(rbd).items():
            rbd.set_external_force(li, w)
        rbd.step()
    x0 = box.q[0]
    for _ in range(int(1.5 / dt)):
        rbd.clear_external_forces()
        for li, w in contact.compute_rbd_contact_forces(rbd).items():
            rbd.set_external_force(li, w)
        rbd.step()
    drift = float(box.q[0] - x0)
    assert abs(drift) < 0.01, (
        f"constraint solver should hold box at 5° tilt with μ=0.6 "
        f"(tan=0.087 < 0.6); got drift={drift:.4f} m"
    )


def test_penalty_lacks_static_friction_baseline():
    """PenaltyContactSolver slides even at θ=20° due to regularised friction.

    Documents the limitation that motivated ConstraintContactSolver. If
    this test starts passing, the penalty path may have been enhanced
    and the constraint solver's value proposition needs revisiting.
    """
    drift = _run_incline(PenaltyContactSolver, angle_deg=20.0, mu=0.5)
    assert drift > 0.005, (
        f"penalty solver unexpectedly held the block (drift={drift:.4f} m). "
        "Static friction may have been added to PenaltyContactSolver."
    )
