"""MPM grasp-drop — robot picks a rigid box and drops it on a clay pile.

Same arm + grasp sequence as ``examples/grasp_demo.py --mode rigid``, but
with two additions:

  1. An MPM clay pile placed off to one side on the floor.
  2. Two extra trajectory phases after ``LIFT``:
     * ``MOVE``   — rotates the pan joint so the palm ends up directly
                    above the pile.
     * ``RELEASE`` — opens the fingers; the rigid box falls under
                    gravity, collides with the clay (one-way via
                    :class:`KinematicBoxCollider`), and the pile deforms.

The Taichi GGUI view uses :class:`SimViewer` (same style as
grasp_demo) with URDF meshes for the arm + box, and an MPM particle
cloud for the pile.

Usage:
  python examples/mpm_grasp_drop.py
  python examples/mpm_grasp_drop.py --material clay   # default
  python examples/mpm_grasp_drop.py --material sand
  python examples/mpm_grasp_drop.py --material metal
  python examples/mpm_grasp_drop.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from robosim.model.factory import create_free_box
from robosim.model.urdf_parser import parse_urdf
from robosim.physics.contact.detection import GroundPlane
from robosim.physics.contact.response import ContactParams
from robosim.physics.contact.solver import ContactSolver
from robosim.physics.mpm.boundary import KinematicBoxCollider
from robosim.physics.mpm.grid import Grid
from robosim.physics.mpm.materials import (
    DruckerPragerPlastic, NeoHookean, VonMisesPlastic,
)
from robosim.physics.mpm.solver import BoxBC, MPMSolver, sample_box_particles
from robosim.physics.rbd.algorithms import gravity_torques
from robosim.physics.rbd.solver import RBDSolver


_URDF = Path(__file__).parent / "urdf" / "arm6_gripper" / "arm6_gripper.urdf"

# ── Grasp poses (same q layout as grasp_demo) ────────────────────────────────
HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.020,  0.020])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.020,  0.020])
# New: swing over to the clay pile (pan = −1.0 rad ≈ −57°).
MOVE_Q     = np.array([-1.0, -0.878,  2.080, -1.202, -0.020,  0.020])
# New: open fingers at the drop location (release grip).
RELEASE_Q  = np.array([-1.0, -0.878,  2.080, -1.202,  0.040, -0.040])

_PHASES = [
    (HOME_Q,     1.5),
    (FOLD_Q,     1.2),
    (APPROACH_Q, 2.5),
    (NEAR_Q,     1.5),
    (CLOSE_Q,    1.5),
    (LIFT_Q,     3.0),
    (MOVE_Q,     2.5),
    (RELEASE_Q,  1.0),
]
_PHASE_NAMES = ["HOME", "FOLD", "APPROACH", "NEAR", "CLOSE", "LIFT",
                "MOVE", "RELEASE", "SETTLE"]

_KP = np.array([280.0, 480.0, 200.0,  75.0, 5e4,  5e4 ])
_KD = np.array([ 35.0,  50.0,  20.0,   2.0, 100., 100.])


# ── Clay pile geometry (placed under the MOVE palm position) ─────────────────

# Pan = −1.0 → palm ends up near (0.28, −0.44, 0.22). Put the pile there.
PILE_CENTER_XY = np.array([0.28, -0.44])
PILE_SIZE      = np.array([0.16, 0.16, 0.10])           # x, y, z extent
PILE_N         = 8                                        # n particles / axis
PILE_DENSITY   = 1400.0

# MPM domain covers only the pile + enough space around it for the box to
# enter/exit during the drop. Oversized domains kill performance — every
# MPM step touches every grid cell. Arm workspace is irrelevant here.
_PAD         = 0.12       # metres of clearance around the pile in xy/z
DOMAIN_LOWER = np.array([
    PILE_CENTER_XY[0] - PILE_SIZE[0] / 2 - _PAD,
    PILE_CENTER_XY[1] - PILE_SIZE[1] / 2 - _PAD,
    0.0,
])
DOMAIN_UPPER = np.array([
    PILE_CENTER_XY[0] + PILE_SIZE[0] / 2 + _PAD,
    PILE_CENTER_XY[1] + PILE_SIZE[1] / 2 + _PAD,
    PILE_SIZE[2] + _PAD,
])
DX           = 0.025


def make_material(kind: str):
    if kind == "clay":
        return VonMisesPlastic(young=1.5e5, poisson=0.3, yield_stress=8e2)
    if kind == "sand":
        return DruckerPragerPlastic(young=1e5, poisson=0.3,
                                    friction_angle=np.deg2rad(40.0))
    if kind == "metal":
        return VonMisesPlastic(young=2e5, poisson=0.3, yield_stress=3e3)
    if kind == "elastic":
        return NeoHookean(young=8e4, poisson=0.3)
    raise ValueError(kind)


def _smooth(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _control(robot, q_target: np.ndarray) -> np.ndarray:
    q_err  = q_target - robot.q
    qd_err = -robot.qd
    g_tau  = gravity_torques(robot, robot.q)
    return _KP * q_err + _KD * qd_err + g_tau


def _height_colors(z: np.ndarray, zmax: float) -> np.ndarray:
    t = np.clip(z / max(zmax, 1e-6), 0, 1).astype(np.float32)
    return np.column_stack([0.3 + 0.5*t, 0.5 + 0.3*t, 1.0 - 0.4*t])


# ═══════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════

def run(material: str = "clay", headless: bool = False) -> None:
    dt       = 0.001
    substeps = 10

    # ── Arm ───────────────────────────────────────────────────────
    robot = parse_urdf(_URDF)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.q[:]   = HOME_Q
    robot.enforce_mimic()
    robot.qd[:]  = 0.0
    arm_solver = RBDSolver(robot=robot)
    arm_solver.initialize(dt=dt)

    # ── Rigid box (the thing the arm picks up) ────────────────────
    BOX_SIZE   = np.array([0.080, 0.080, 0.080])
    BOX_CENTRE = np.array([0.600, 0.000, 0.040])
    BOX_COLOR  = np.array([0.85, 0.15, 0.15])
    box_robot  = create_free_box(
        name="target_box", size=tuple(BOX_SIZE),
        mass=1.0, position=BOX_CENTRE.copy(),
        color=np.append(BOX_COLOR, 1.0),
    )
    box_robot.gravity = np.array([0.0, 0.0, -9.81])
    box_solver = RBDSolver(robot=box_robot)
    box_solver.initialize(dt=dt)

    # ── Rigid ground contact (arm ↔ ground, box ↔ ground, arm ↔ box) ──
    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=ContactParams(stiffness=1e3, damping=50, friction_mu=0.0),
    )
    contact.register_rbd(arm_solver, robot_id="arm")
    contact.register_rbd(box_solver, robot_id="box")
    for arm_link in ["base_link", "link_1", "link_2", "link_3", "link_4",
                     "palm_link"]:
        contact.add_cross_filter("arm", arm_link, "box", "target_box_body")

    # ── MPM pile ─────────────────────────────────────────────────
    pile_lower = np.array([
        PILE_CENTER_XY[0] - PILE_SIZE[0] / 2,
        PILE_CENTER_XY[1] - PILE_SIZE[1] / 2,
        0.0,
    ])
    pile_upper = pile_lower + PILE_SIZE
    mpm_particles = sample_box_particles(pile_lower, pile_upper,
                                         PILE_N, PILE_DENSITY)
    mpm_grid = Grid.from_bounds(DOMAIN_LOWER, DOMAIN_UPPER, DX, pad=3)
    # Box pose that the MPM solver uses — lives outside the MPM domain
    # until the rigid box actually falls onto the pile.
    mpm_collider = KinematicBoxCollider(
        center=BOX_CENTRE.copy(),
        half_extent=BOX_SIZE * 0.5,
        velocity=np.zeros(3),
    )
    mpm_solver = MPMSolver(
        particles=mpm_particles, grid=mpm_grid,
        material=make_material(material),
        gravity=np.array([0.0, 0.0, -9.81]),
        bcs=[BoxBC(lower=DOMAIN_LOWER,
                   upper=np.array([DOMAIN_UPPER[0], DOMAIN_UPPER[1], 10.0]),
                   mode="slip")],
        colliders=[mpm_collider],
    )

    # ── Phase interpolation state ────────────────────────────────
    phase_idx     = 0
    phase_start_t = 0.0
    q_phase_start = HOME_Q.copy()
    q_target      = HOME_Q.copy()

    def advance_phases(t: float) -> None:
        nonlocal phase_idx, phase_start_t, q_phase_start
        if phase_idx >= len(_PHASES):
            q_target[:] = _PHASES[-1][0]
            return
        tgt, dur = _PHASES[phase_idx]
        alpha = _smooth((t - phase_start_t) / dur)
        q_target[:] = q_phase_start + alpha * (tgt - q_phase_start)
        if (t - phase_start_t) >= dur:
            phase_idx    += 1
            phase_start_t = t
            q_phase_start = tgt.copy()

    # ── Constraint grip (arm ↔ box kinematic coupling) ───────────
    _LIFT_START_T   = sum(d for _, d in _PHASES[:5])
    _RELEASE_START  = sum(d for _, d in _PHASES[:7])   # after MOVE ends
    grip_active     = [False]
    grip_palm_pos0  = np.zeros(3)
    grip_box_pos0   = np.zeros(3)
    grip_box_rot0   = np.zeros(3)
    grip_offset     = np.zeros(3)
    prev_palm_pos   = [None]
    palm_idx        = robot.link_index("palm_link")

    box_ground_xy  = BOX_CENTRE[:2].copy()
    box_ground_rot = np.zeros(3)

    sim_time = [0.0]

    def step_once() -> None:
        nonlocal grip_box_pos0, grip_box_rot0, grip_offset
        dt_step = dt
        advance_phases(sim_time[0])

        # Arm PD + gravity comp + ground/box contact
        arm_solver.clear_external_forces()
        arm_solver.tau = _control(robot, q_target)
        for li, w in contact.compute_rbd_contact_forces(arm_solver).items():
            arm_solver.set_external_force(li, w)
        arm_solver.step()
        robot.enforce_mimic()

        fk = robot.forward_kinematics()
        palm_pos = fk[palm_idx].translation.copy()

        # Rigid box physics + grip
        box_solver.clear_external_forces()
        for li, w in contact.compute_rbd_contact_forces(box_solver).items():
            box_solver.set_external_force(li, w)
        box_solver.step()

        if not grip_active[0] and box_robot.q[2] < 0.045:
            # Hold the box on the ground until the first grip activation
            # so it doesn't wander under ground-contact noise.
            if sim_time[0] < _LIFT_START_T:
                box_robot.q[:2]   = box_ground_xy
                box_robot.qd[:2]  = 0.0
                box_robot.q[3:6]  = box_ground_rot
                box_robot.qd[3:6] = 0.0

        # Grip on/off. Active when fingers closed AND we haven't reached
        # the release phase yet. Once release starts (fingers reopening),
        # grip turns off permanently and the box follows free dynamics.
        fingers_closed = robot.q[4] < 0.003
        if (sim_time[0] >= _LIFT_START_T
                and sim_time[0] < _RELEASE_START
                and fingers_closed):
            if not grip_active[0]:
                grip_active[0]   = True
                grip_palm_pos0[:] = palm_pos
                grip_box_pos0 = box_robot.q[:3].copy()
                grip_box_rot0 = box_robot.q[3:6].copy()
                grip_offset   = grip_box_pos0 - grip_palm_pos0
            target_pos = palm_pos + grip_offset
            if prev_palm_pos[0] is not None:
                palm_vel = (palm_pos - prev_palm_pos[0]) / dt_step
            else:
                palm_vel = np.zeros(3)
            box_robot.q[:3]   = target_pos
            box_robot.qd[:3]  = palm_vel
            box_robot.q[3:6]  = grip_box_rot0
            box_robot.qd[3:6] = 0.0
        elif sim_time[0] >= _RELEASE_START and grip_active[0]:
            # Release: hand off to free dynamics with the current palm vel.
            grip_active[0] = False

        # Feed the rigid box pose into the MPM collider and step MPM.
        new_center = box_robot.q[:3].copy()
        mpm_collider.velocity = (new_center - mpm_collider.center) / dt_step
        mpm_collider.center = new_center
        try:
            mpm_solver.step(dt_step)
        except IndexError:
            pass  # particle left the grid — skip MPM this substep

        prev_palm_pos[0] = palm_pos
        sim_time[0] += dt_step

    # ── Headless: run a finite duration, print summary ───────────
    if headless:
        total_dur = sum(d for _, d in _PHASES) + 2.0   # + settle time
        total_steps = int(total_dur / dt) + substeps
        t_wall0 = time.time()
        for fr in range(total_steps // substeps + 1):
            for _ in range(substeps):
                step_once()
            if fr % 200 == 0:
                p = mpm_particles
                com_box = box_robot.q[:3]
                z_top = float(p.x[:, 2].max())
                xy_ext = float(np.ptp(p.x[:, :2], axis=0).mean())
                ph = _PHASE_NAMES[min(phase_idx, len(_PHASE_NAMES) - 1)]
                print(f"t={sim_time[0]:6.3f}s  phase={ph:<8}  "
                      f"box=({com_box[0]:+.3f},{com_box[1]:+.3f},{com_box[2]:+.3f})  "
                      f"pile z_top={z_top:.3f}  xy_ext={xy_ext:.3f}")
        print(f"\nWall time: {time.time()-t_wall0:.2f} s")
        return

    # ── GUI ──────────────────────────────────────────────────────
    from robosim.viz.viewer import SimViewer
    from robosim.viz.scene_renderer import RobotRenderer

    viewer = SimViewer(
        title=f"RoboSim — MPM Grasp-Drop [{material.upper()}]",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    arm_renderer = RobotRenderer(robot, viewer)
    arm_renderer.setup()
    box_renderer = RobotRenderer(box_robot, viewer)
    box_renderer.setup()

    viewer.add_particles(
        "mpm", mpm_particles.x, radius=0.008,
        per_vertex_color=_height_colors(
            mpm_particles.x[:, 2], PILE_SIZE[2] * 2.0,
        ),
    )

    step_count = [0]

    def frame_step(frame: int) -> None:
        for _ in range(substeps):
            step_once()
            step_count[0] += 1

        arm_renderer.update()
        box_renderer.update()
        viewer.update_particles(
            "mpm", mpm_particles.x,
            per_vertex_color=_height_colors(
                mpm_particles.x[:, 2], PILE_SIZE[2] * 2.0,
            ),
        )

        ph = _PHASE_NAMES[min(phase_idx, len(_PHASE_NAMES) - 1)]
        box_com = box_robot.q[:3]
        p = mpm_particles
        z_top = float(p.x[:, 2].max())
        xy_ext = float(np.ptp(p.x[:, :2], axis=0).mean())
        viewer.add_text(
            f"t = {sim_time[0]:6.3f} s    Phase: {ph}    [{material.upper()}]\n"
            f"Box CoM : ({box_com[0]:+.3f}, {box_com[1]:+.3f}, {box_com[2]:+.3f})\n"
            f"Pile z_top: {z_top:.3f}    xy extent: {xy_ext:.3f}\n"
            f"grip: {'ON' if grip_active[0] else 'off'}    "
            f"q_fing: {robot.q[4]:+.3f} m\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(frame_step)
    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--material", choices=["clay", "sand", "metal", "elastic"],
                    default="clay")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if not args.headless:
        import taichi as ti
        ti.init(arch=ti.metal)

    run(material=args.material, headless=args.headless)


if __name__ == "__main__":
    main()
