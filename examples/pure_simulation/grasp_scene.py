"""Grasp demo via the Scene API.

Robot picks up a box (rigid / FEM / CB modes) in ~40 lines of user code:
  Scene → add robot + box → register contact/grip → build trajectory → run

Uses penalty contact + kinematic grip lock (Scene's default). The
constraint-based Coulomb-cone solver is also available — see
``Scene(contact_solver="constraint")`` and tests/test_constraint_friction.py
— but it's not exposed here because penalty + grip-lock is the cleaner
visual for this demo.

Usage:
  python examples/grasp_scene.py              # rigid mode, GUI
  python examples/grasp_scene.py --mode fem --headless
  python examples/grasp_scene.py --mode cb  --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim import Scene, Robot, Box, Ground, FEM, CB, Rigid, JointPD

# ── URDF path ─────────────────────────────────────────────────────────────────
_URDF = str(Path(__file__).parent / "urdf" / "arm6_gripper" / "arm6_gripper.urdf")

# ── Grasp poses ───────────────────────────────────────────────────────────────
HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
# Finger target (q[4]=-0.011 with q[5]=+0.011 via mimic) puts the finger
# inner face right at the box +Y surface for this CLOSE pose.  Going further
# (e.g. -0.020) commands the PD to drive the fingers THROUGH the box →
# visible "fork-skewer" penetration during LIFT.  -0.011 still triggers the
# kinematic grip threshold (q[4] < 0.003) without overshooting.
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.011,  0.011])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.011,  0.011])
# RELEASE: same arm pose as the LIFT hold but fingers wide open (HOME
# finger q = ±0.040). Crossing ``release_val=0.020`` snaps the kinematic
# grip off and the box inherits the palm's velocity at that instant.
RELEASE_Q  = np.array([ 0.0, -0.878,  2.080, -1.202,  0.040, -0.040])

_LIFT_START_T = 1.5 + 1.2 + 2.5 + 1.5 + 1.5   # = 8.2 s
# Trajectory totals: HOME+FOLD+APPROACH+NEAR+CLOSE+LIFT+RELEASE = 13.7 s.
# After RELEASE the trajectory holds and physics runs free (box falls).
_RELEASE_END_T = _LIFT_START_T + 3.0 + 0.5     # = 11.7 s
_DROP_END_T    = _RELEASE_END_T + 2.5          # let the box settle

# ── Physics mode → contact params ────────────────────────────────────────────
_CONTACT_PARAMS = {
    "fem": dict(k=1e3,  c=30.0),
    "cb":  dict(k=1e4,  c=50.0),
    "cb-coarse":  dict(k=1e4,  c=50.0),
    "rigid": {},
}

# ── Physics mode → box physics spec ──────────────────────────────────────────
def _box_physics(mode: str):
    if mode == "fem":
        return FEM(young=1e5, poisson=0.45, mesh=(8, 8, 8), dt_scale=5)
    if mode == "cb":
        # mesh=(6,6,6) — at (4,4,4) the 20 mm node spacing is so coarse that
        # only the finger's mid-row (z=4 cm) sits inside its 20 mm collision
        # height, so the finger appears to skewer the box between rows.
        # (6,6,6) gives 3 rows inside the slab (with the SLAB_PAD); (8,8,8)
        # is similar but doubles the CB basis size for marginal gain.
        return CB(young=1.1e5, poisson=0.45, mesh=(6, 6, 6), n_modes=10)
    if mode == "cb-coarse":
        # Same physics, coarser mesh. CB step scales cubically with
        # n_nodes — (4,4,4) is ~7× faster than (6,6,6). Fidelity drops
        # but RL workloads may not care.
        return CB(young=1.1e5, poisson=0.45, mesh=(4, 4, 4), n_modes=10)
    return Rigid()


def main(mode: str = "rigid", headless: bool = False) -> None:
    # ── Build scene ───────────────────────────────────────────────────────────
    scene = Scene(dt=0.001, gravity=[0, 0, -9.81], substeps=10)

    ground = scene.add(Ground(height=0.0))

    robot = scene.add(
        Robot(_URDF)
        .controller(JointPD(
            # Arm joints stiff (large links, gravity-compensated load).
            # Finger joints intentionally soft: a stiff finger PD overpowers
            # the contact reaction and drives the fingertips visibly through
            # the box mesh.  kp≈800 leaves enough grip force for the
            # kinematic grip threshold to fire while keeping penetration <5 mm.
            kp=[280.0, 480.0, 200.0, 75.0,  800,  800],
            kd=[ 35.0,  50.0,  20.0,  2.0,   30,   30],
        ))
        .initial_q(HOME_Q)
        .name("arm")
    )

    box = scene.add(
        Box(size=0.08, mass=1.0, pos=[0.6, 0.0, 0.04])
        .physics(_box_physics(mode))
        .name("box")
    )

    # ── Contact + grip ────────────────────────────────────────────────────────
    if mode != "rigid":
        scene.contact(
            robot, box,
            links=["left_finger", "right_finger"],
            **_CONTACT_PARAMS[mode],
        )
    scene.grip(
        robot, box,
        trigger_q=4,
        trigger_val=0.003,
        lift_start_t=_LIFT_START_T,
        # Fingers must cross 20 mm back open before the grip drops; with
        # the RELEASE trajectory target of 40 mm and a soft finger PD,
        # this fires roughly midway through the RELEASE phase.
        release_val=0.020,
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    traj = robot.trajectory()
    traj.phase("HOME",     target=HOME_Q,     duration=1.5)
    traj.phase("FOLD",     target=FOLD_Q,     duration=1.2)
    traj.phase("APPROACH", target=APPROACH_Q, duration=2.5)
    traj.phase("NEAR",     target=NEAR_Q,     duration=1.5)
    traj.phase("CLOSE",    target=CLOSE_Q,    duration=1.5)
    traj.phase("LIFT",     target=LIFT_Q,     duration=3.0)
    # Open fingers — box drops onto the ground, bounces, settles.
    traj.phase("RELEASE",  target=RELEASE_Q,  duration=0.5)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(scene)
    scene.run(
        trajectories=traj,
        # Total run = lift_start + lift + release-ramp + free-fall settle.
        duration=_DROP_END_T,
        viewer=not headless,
        headless=headless,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp demo — Scene API")
    parser.add_argument("--mode",     default="rigid",
                        choices=["rigid", "fem", "cb", "cb-coarse"])
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(mode=args.mode, headless=args.headless)
