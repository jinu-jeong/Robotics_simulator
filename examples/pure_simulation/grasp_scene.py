"""Grasp demo via the Scene API.

Robot picks up a box (rigid / FEM / CB) with optional deformable CB fingers:
  Scene → add robot + box → register contact/grip → build trajectory → run

Uses penalty contact + kinematic grip lock (Scene's default). The
constraint-based Coulomb-cone solver is also available — see
``Scene(contact_solver="constraint")`` and tests/test_constraint_friction.py
— but it's not exposed here because penalty + grip-lock is the cleaner
visual for this demo.

Usage:
  python examples/pure_simulation/grasp_scene.py
  python examples/pure_simulation/grasp_scene.py --object rigid --finger cb
  python examples/pure_simulation/grasp_scene.py --object cb --finger cb --headless
  python examples/pure_simulation/grasp_scene.py --mode fem --headless   # legacy
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
# Finger close target: joint_left origin at palm tip (x=0.025, y=0.043);
# visual half-width is 0.005, so the finger inner face sits at
# Y = 0.043 + q[4] - 0.005 (palm frame).  The box's +Y face is at
# palm-Y 0.040 (box half-extent 0.04, centered on palm Y=0).
# Solving inner_face == +Y face gives q[4] ≈ +0.002.  We slightly
# undershoot at q[4]=-0.002 so the finger PD still drives it just
# 4 mm past the surface (enough load for the kinematic grip threshold
# q[4] < 0.003 to fire and lock the grip on), without the previous
# 13 mm "fork through tofu" overshoot the old -0.011 target produced.
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.002,  0.002])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.002,  0.002])
# RELEASE: same arm pose as the LIFT hold but fingers wide open (HOME
# finger q = ±0.040). Crossing ``release_val=0.020`` snaps the kinematic
# grip off and the box inherits the palm's velocity at that instant.
RELEASE_Q  = np.array([ 0.0, -0.878,  2.080, -1.202,  0.040, -0.040])

_LIFT_START_T = 1.5 + 1.2 + 2.5 + 1.5 + 1.5   # = 8.2 s
# Trajectory totals: HOME+FOLD+APPROACH+NEAR+CLOSE+LIFT+RELEASE = 13.7 s.
# After RELEASE the trajectory holds and physics runs free (box falls).
_RELEASE_END_T = _LIFT_START_T + 3.0 + 0.5     # = 11.7 s
_DROP_END_T    = _RELEASE_END_T + 2.5          # let the box settle

_FINGER_LINKS = ["left_finger", "right_finger"]

# ── Contact params ────────────────────────────────────────────────────────────
_OBJECT_CONTACT_PARAMS = {
    "fem": dict(k=1e3,  c=30.0),
    "cb":  dict(k=1e4,  c=50.0),
}
# Softer finger↔rigid penalty so the CB mesh can bend (cantilever visual).
_FINGER_RIGID_CONTACT = dict(k=300.0, c=8.0)

# ── Object physics ────────────────────────────────────────────────────────────
def _box_physics(object_mode: str):
    if object_mode == "fem":
        return FEM(young=1e5, poisson=0.45, mesh=(8, 8, 8), dt_scale=5)
    if object_mode == "cb":
        return CB(young=1.1e5, poisson=0.45, mesh=(6, 6, 6), n_modes=10)
    return Rigid()


def main(
    object_mode: str = "rigid",
    finger_mode: str = "rigid",
    headless: bool = False,
) -> None:
    # ── Build scene ───────────────────────────────────────────────────────────
    scene = Scene(dt=0.001, gravity=[0, 0, -9.81], substeps=10)

    ground = scene.add(Ground(height=0.0))

    robot = scene.add(
        Robot(_URDF)
        .controller(JointPD(
            kp=[280.0, 480.0, 200.0, 75.0,  800,  800],
            kd=[ 35.0,  50.0,  20.0,  2.0,   30,   30],
        ))
        .initial_q(HOME_Q)
        .name("arm")
    )

    box = scene.add(
        Box(size=0.08, mass=1.0, pos=[0.625, 0.0, 0.04])
        .physics(_box_physics(object_mode))
        .name("box")
    )

    # ── Contact + deformable fingers ──────────────────────────────────────────
    if object_mode != "rigid":
        scene.contact(
            robot, box,
            links=_FINGER_LINKS,
            **_OBJECT_CONTACT_PARAMS[object_mode],
        )

    if finger_mode == "cb":
        # Soft CB prong: anchored at palm (-X face), free tip bends under load.
        scene.attach_deformable_gripper(
            robot, links=_FINGER_LINKS,
            divisions=(4, 2, 2),
            young=3.0e4,
            poisson=0.45,
            n_modes=8,
            damping=0.12,
        )
        if object_mode == "rigid":
            scene.contact_deformable_gripper(
                robot, box,
                **_FINGER_RIGID_CONTACT,
            )

    scene.grip(
        robot, box,
        trigger_q=4,
        trigger_val=0.003,
        lift_start_t=_LIFT_START_T,
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
    traj.phase("RELEASE",  target=RELEASE_Q,  duration=0.5)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(scene)
    scene.run(
        trajectories=traj,
        duration=_DROP_END_T,
        viewer=not headless,
        headless=headless,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp demo — Scene API")
    parser.add_argument(
        "--object", default=None, choices=["rigid", "fem", "cb"],
        help="physics model for the grasped box",
    )
    parser.add_argument(
        "--finger", default=None, choices=["rigid", "cb"],
        help="physics model for gripper fingers (cb = deformable Craig-Bampton)",
    )
    parser.add_argument(
        "--mode", default=None, choices=["rigid", "fem", "cb"],
        help="legacy alias: sets --object and auto-enables deformable fingers "
             "for fem/cb (use --object/--finger instead)",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    if args.mode is not None:
        object_mode = args.object or args.mode
        if args.finger is not None:
            finger_mode = args.finger
        else:
            finger_mode = "cb" if args.mode in ("fem", "cb") else "rigid"
    else:
        object_mode = args.object or "rigid"
        finger_mode = args.finger or "rigid"

    main(object_mode=object_mode, finger_mode=finger_mode, headless=args.headless)
