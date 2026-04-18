"""Grasp demo — rewritten with the Scene API.

Same scenario as grasp_demo.py but expressed in ~40 lines of user code:
  Scene → add robot + box → register contact/grip → build trajectory → run

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
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.020,  0.020])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.020,  0.020])

_LIFT_START_T = 1.5 + 1.2 + 2.5 + 1.5 + 1.5   # = 8.2 s

# ── Physics mode → contact params ────────────────────────────────────────────
_CONTACT_PARAMS = {
    "fem": dict(k=1e3,  c=30.0),
    "cb":  dict(k=1e4,  c=50.0),
    "rigid": {},
}

# ── Physics mode → box physics spec ──────────────────────────────────────────
def _box_physics(mode: str):
    if mode == "fem":
        return FEM(young=1e5, poisson=0.45, mesh=(8, 8, 8), dt_scale=5)
    if mode == "cb":
        return CB(young=1e5, poisson=0.45, mesh=(4, 4, 4), n_modes=10)
    return Rigid()


def main(mode: str = "rigid", headless: bool = False) -> None:
    # ── Build scene ───────────────────────────────────────────────────────────
    scene = Scene(dt=0.001, gravity=[0, 0, -9.81], substeps=10)

    ground = scene.add(Ground(height=0.0))

    robot = scene.add(
        Robot(_URDF)
        .controller(JointPD(
            kp=[280.0, 480.0, 200.0, 75.0, 5e4, 5e4],
            kd=[ 35.0,  50.0,  20.0,  2.0, 100, 100],
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
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    traj = robot.trajectory()
    traj.phase("HOME",     target=HOME_Q,     duration=1.5)
    traj.phase("FOLD",     target=FOLD_Q,     duration=1.2)
    traj.phase("APPROACH", target=APPROACH_Q, duration=2.5)
    traj.phase("NEAR",     target=NEAR_Q,     duration=1.5)
    traj.phase("CLOSE",    target=CLOSE_Q,    duration=1.5)
    traj.phase("LIFT",     target=LIFT_Q,     duration=3.0)

    # ── Run ───────────────────────────────────────────────────────────────────
    print(scene)
    scene.run(
        trajectories=traj,
        duration=_LIFT_START_T + 3.2,
        viewer=not headless,
        headless=headless,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grasp demo — Scene API")
    parser.add_argument("--mode",     default="rigid", choices=["rigid", "fem", "cb"])
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(mode=args.mode, headless=args.headless)
