"""Grasp demo with a plastic box — Scene API.

Robot picks up a soft box; the constitutive model is corotational
small-strain J2 plasticity, so the finger contacts dent the box
permanently. Watch the surface near the gripper jaws — once σ_eq
crosses σ_Y the deformation stops disappearing on release.

Same trajectory + arm setup as ``grasp_scene.py``; the only difference
is ``.physics(FEMPlastic(...))`` in place of ``FEM(...)`` / ``CB(...)``.

Usage::

    python examples/pure_simulation/grasp_scene_plastic.py            # GUI
    python examples/pure_simulation/grasp_scene_plastic.py --headless # diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim import Scene, Robot, Box, Ground, FEMPlastic, JointPD


_URDF = str(Path(__file__).parent / "urdf" / "arm6_gripper" / "arm6_gripper.urdf")

# ── Grasp poses (mirrors grasp_scene.py) ─────────────────────────────────────
HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
# Plastic box yields easily — push the fingers a bit deeper to be sure
# the visible dent forms (σ_eq must clear σ_Y_eff).
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.013,  0.013])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.013,  0.013])

_LIFT_START_T = 1.5 + 1.2 + 2.5 + 1.5 + 1.5   # = 8.2 s


def main(headless: bool = False) -> None:
    scene = Scene(dt=0.001, gravity=[0, 0, -9.81], substeps=10)

    scene.add(Ground(height=0.0))

    robot = scene.add(
        Robot(_URDF)
        .controller(JointPD(
            kp=[280.0, 480.0, 200.0, 75.0,  800,  800],
            kd=[ 35.0,  50.0,  20.0,  2.0,   30,   30],
        ))
        .initial_q(HOME_Q)
        .name("arm")
    )

    # Soft, easily-yielding box. yield_stress small enough that the
    # gripper crushes it; hardening high enough that the deformation
    # is retained on release (perfect plasticity would let elastic
    # rebound erase it via reverse yielding).
    box = scene.add(
        Box(size=0.08, mass=0.5, pos=[0.6, 0.0, 0.04])
        .physics(FEMPlastic(
            young=5e4, poisson=0.45,
            yield_stress=5e2, hardening=8e4,
            mesh=(4, 4, 4), dt_scale=5,
        ))
        .name("box")
    )

    # FEM-mesh contact uses position projection internally, so no k/c
    # tuning is needed — the runner reads body.x and snaps boundary
    # nodes on the finger side of each contact pair.
    scene.contact(robot, box, links=["left_finger", "right_finger"])
    scene.grip(
        robot, box,
        trigger_q=4, trigger_val=0.003,
        lift_start_t=_LIFT_START_T,
    )

    traj = robot.trajectory()
    traj.phase("HOME",     target=HOME_Q,     duration=1.5)
    traj.phase("FOLD",     target=FOLD_Q,     duration=1.2)
    traj.phase("APPROACH", target=APPROACH_Q, duration=2.5)
    traj.phase("NEAR",     target=NEAR_Q,     duration=1.5)
    traj.phase("CLOSE",    target=CLOSE_Q,    duration=1.5)
    traj.phase("LIFT",     target=LIFT_Q,     duration=3.0)

    print(scene)
    print("[plastic] σ_Y =", box._body.material.yield_stress,
          "  H =", box._body.material.hardening)
    scene.run(
        trajectories=traj,
        duration=_LIFT_START_T + 3.2,
        viewer=not headless,
        headless=headless,
    )

    # Plastic-state diagnostic: report accumulated eps_p norm + the
    # per-element maximum equivalent plastic strain. ε_p_eq itself is
    # √(2/3) ‖eps_p‖_F per element; max across elements gives the "most
    # plastically deformed spot".
    body = box._body
    if body.eps_p is None:
        print("\n[plastic] no eps_p storage (material not plastic?)")
    else:
        eps_p_eq = np.sqrt(2.0 / 3.0) * np.linalg.norm(body.eps_p, axis=(1, 2))
        print(f"\n[plastic] ‖eps_p‖ (sum of all elements) = "
              f"{float(np.linalg.norm(body.eps_p)):.3f}")
        print(f"[plastic] max per-element ε_p_eq          = "
              f"{float(eps_p_eq.max()):.4f}  (~{eps_p_eq.max()*100:.1f}% strain)")
        print(f"[plastic] yielded elements (ε_p_eq > 1e-4) = "
              f"{int((eps_p_eq > 1e-4).sum())} / {eps_p_eq.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plastic grasp demo (Scene API)")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    main(headless=args.headless)
