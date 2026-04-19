"""MPM grasp-drop — robot picks a box and drops it on a clay pile.

Same arm + grasp sequence as ``examples/grasp_demo.py``, with two extra
phases after ``LIFT``:

  - ``MOVE``    — pan joint swings the lifted box over the clay pile.
  - ``RELEASE`` — fingers reopen; grip constraint turns off; the box
                  falls under gravity, collides with the clay, and the
                  pile deforms.

Box physics modes (``--mode``):

  - ``rigid`` (default)  — 6-DOF free-floating RBD box.
  - ``cb``               — Craig-Bampton reduced deformable. The box
                           wobbles/squishes on impact; the MPM solver
                           sees the box via its live AABB (one-way
                           kinematic coupling, same as rigid mode).

In both modes the MPM pile reacts through a
:class:`KinematicBoxCollider` fed the current box pose each substep.

Usage:
  python examples/mpm_grasp_drop.py                        # rigid / clay
  python examples/mpm_grasp_drop.py --mode cb              # deformable box
  python examples/mpm_grasp_drop.py --mode cb --material sand
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

# ── Grasp poses ──────────────────────────────────────────────────────────────
HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.020,  0.020])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.020,  0.020])
MOVE_Q     = np.array([-1.0, -0.878,  2.080, -1.202, -0.020,  0.020])
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

# Finger geometry (matches grasp_demo — must mirror the URDF collision box).
_F_HX, _F_HY, _F_HZ = 0.040, 0.005, 0.010
_K_FINGER = 1e4
_C_FINGER = 50.0

# ── Clay pile geometry (under the MOVE palm position, pan = −1.0 rad) ───────
PILE_CENTER_XY = np.array([0.28, -0.44])
PILE_SIZE      = np.array([0.16, 0.16, 0.10])
PILE_N         = 8
PILE_DENSITY   = 1400.0

_PAD         = 0.12
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
DX = 0.025


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


def _stress_color(vm: np.ndarray, vmax: float) -> np.ndarray:
    """Jet colormap on von Mises stress (matches grasp_demo CB mode)."""
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


def _apply_ground_projection(body, mu: float = 0.9) -> None:
    below = body.x[:, 2] < 0.0
    if not below.any():
        return
    body.x[below, 2] = 0.0
    body.v[below, 2] = np.maximum(body.v[below, 2], 0.0)
    if mu > 0.0:
        body.v[below, :2] *= max(0.0, 1.0 - mu * 0.1)


def _compute_finger_forces(bx, bv, arm_fk, robot):
    """Per-node penalty contact between arm fingers and a CB box.

    Mirrors grasp_demo._compute_finger_forces. Returns
    ``(f_fem, l_idx, w_left, r_idx, w_right)``.
    """
    n = bx.shape[0]
    f_fem   = np.zeros(n * 3)
    w_left  = np.zeros(6)
    w_right = np.zeros(6)
    l_idx = robot.link_index("left_finger")
    r_idx = robot.link_index("right_finger")
    _geom_off = np.array([0.020, 0.0, 0.020])
    p_l = arm_fk[l_idx].translation + _geom_off
    p_r = arm_fk[r_idx].translation + _geom_off
    T_l_inv = arm_fk[l_idx].inverse()
    T_r_inv = arm_fk[r_idx].inverse()
    for i in range(n):
        xi, vi = bx[i], bv[i]
        if abs(xi[0] - p_l[0]) < _F_HX and abs(xi[2] - p_l[2]) < _F_HZ:
            pen = xi[1] - (p_l[1] - _F_HY)
            if pen > 0.0:
                fn = _K_FINGER * pen + _C_FINGER * max(0.0, vi[1])
                f_fem[i * 3 + 1] -= fn
                f_react_world  = np.array([0.0, fn, 0.0])
                f_react_link   = T_l_inv.apply_vector(f_react_world)
                p_contact_link = T_l_inv.apply_point(xi)
                w_left[:3] += np.cross(p_contact_link, f_react_link)
                w_left[3:] += f_react_link
        if abs(xi[0] - p_r[0]) < _F_HX and abs(xi[2] - p_r[2]) < _F_HZ:
            pen = (p_r[1] + _F_HY) - xi[1]
            if pen > 0.0:
                fn = _K_FINGER * pen + _C_FINGER * max(0.0, -vi[1])
                f_fem[i * 3 + 1] += fn
                f_react_world  = np.array([0.0, -fn, 0.0])
                f_react_link   = T_r_inv.apply_vector(f_react_world)
                p_contact_link = T_r_inv.apply_point(xi)
                w_right[:3] += np.cross(p_contact_link, f_react_link)
                w_right[3:] += f_react_link
    return f_fem, l_idx, w_left, r_idx, w_right


# ═══════════════════════════════════════════════════════════════════
# Main driver
# ═══════════════════════════════════════════════════════════════════

def run(mode: str = "rigid",
        material: str = "clay",
        headless: bool = False,
        n_modes: int = 10) -> None:
    assert mode in ("rigid", "cb"), f"unknown mode: {mode}"
    dt       = 0.001
    substeps = 10

    BOX_SIZE   = np.array([0.080, 0.080, 0.080])
    BOX_CENTRE = np.array([0.600, 0.000, 0.040])
    BOX_COLOR  = (np.array([0.85, 0.15, 0.15]) if mode == "rigid"
                  else np.array([0.20, 0.55, 0.90]))

    # ── Arm ───────────────────────────────────────────────────────
    robot = parse_urdf(_URDF)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.q[:]   = HOME_Q
    robot.enforce_mimic()
    robot.qd[:]  = 0.0
    arm_solver = RBDSolver(robot=robot)
    arm_solver.initialize(dt=dt)

    # ── Box (rigid or CB) ─────────────────────────────────────────
    box_robot = box_solver = None
    cb_body = cb_solver = surf_faces = None

    if mode == "rigid":
        box_robot = create_free_box(
            name="target_box", size=tuple(BOX_SIZE),
            mass=1.0, position=BOX_CENTRE.copy(),
            color=np.append(BOX_COLOR, 1.0),
        )
        box_robot.gravity = np.array([0.0, 0.0, -9.81])
        box_solver = RBDSolver(robot=box_robot)
        box_solver.initialize(dt=dt)
    else:  # cb
        from robosim.physics.fem.materials import CorotationalElastic
        from robosim.physics.fem.mesh import FEMesh
        from robosim.physics.fem.reduced import CraigBamptonBody, CraigBamptonSolver
        from robosim.physics.fem.assembly import batch_von_mises  # noqa: F401

        box_origin = BOX_CENTRE - BOX_SIZE * 0.5
        box_origin[2] = 0.0
        mesh = FEMesh.create_hex_box(origin=box_origin, size=BOX_SIZE,
                                     divisions=(4, 4, 4))
        density  = 1.0 / float(np.prod(BOX_SIZE))
        material_fem = CorotationalElastic(young=1e5, poisson=0.45)
        print(f"\n  Building Craig-Bampton basis (n_modes={n_modes}) …")
        t_b0 = time.time()
        cb_body = CraigBamptonBody(
            mesh=mesh, material=material_fem, density=density,
            n_modes=n_modes,
            gravity=np.array([0.0, 0.0, -9.81]),
            damping=0.5, name="cb_box",
        )
        print(f"  CB basis built in {time.time()-t_b0:.2f} s "
              f"(reduced DOFs: {cb_body._n_r})")
        cb_solver = CraigBamptonSolver(bodies=[cb_body])
        cb_solver.initialize(dt=dt)
        surf_faces = mesh.extract_surface()

    # ── Contact (arm ↔ ground; rigid box ↔ ground + arm ↔ rigid box) ──
    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=ContactParams(stiffness=1e3, damping=50, friction_mu=0.0),
    )
    contact.register_rbd(arm_solver, robot_id="arm")
    if mode == "rigid":
        contact.register_rbd(box_solver, robot_id="box")
        for arm_link in ["base_link", "link_1", "link_2", "link_3",
                         "link_4", "palm_link"]:
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

    # ── Phase + grip state ───────────────────────────────────────
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

    _LIFT_START_T  = sum(d for _, d in _PHASES[:5])
    _RELEASE_START = sum(d for _, d in _PHASES[:7])
    grip_active    = [False]
    grip_palm_pos0 = np.zeros(3)
    grip_box_pos0  = np.zeros(3)
    grip_box_rot0  = np.zeros(3)
    grip_offset    = np.zeros(3)
    grip_body_x0   = [None]
    prev_palm_pos  = [None]
    palm_idx       = robot.link_index("palm_link")

    _l_idx = robot.link_index("left_finger")
    _r_idx = robot.link_index("right_finger")
    cached_w_left  = np.zeros(6)
    cached_w_right = np.zeros(6)

    box_ground_xy = BOX_CENTRE[:2].copy()
    sim_time = [0.0]

    def _box_com() -> np.ndarray:
        if mode == "rigid":
            return box_robot.q[:3].copy()
        return cb_body.x.mean(axis=0)

    def _box_collider_pose() -> tuple[np.ndarray, np.ndarray]:
        """Centre + half-extent to drive the MPM collider.

        For CB we track the centre of mass but keep a **fixed** half-extent
        equal to the undeformed box size. Using the live AABB makes the
        collider wobble during deformation, which translates into huge
        prescribed grid velocities when a face moves a few mm per step —
        enough to blow up the MLS-MPM F-integration on impact. The CoM
        alone is enough to drive contact; the lost deformation nuance in
        the collider shape is not visible anyway (the pile reacts on a
        coarser length scale than the box's sub-mm wobble).
        """
        if mode == "rigid":
            return box_robot.q[:3].copy(), BOX_SIZE * 0.5
        return cb_body.x.mean(axis=0), BOX_SIZE * 0.5

    def step_once() -> None:
        nonlocal grip_box_pos0, grip_box_rot0, grip_offset
        nonlocal cached_w_left, cached_w_right
        advance_phases(sim_time[0])

        # ── Arm ──
        arm_solver.clear_external_forces()
        arm_solver.tau = _control(robot, q_target)
        for li, w in contact.compute_rbd_contact_forces(arm_solver).items():
            arm_solver.set_external_force(li, w)
        # Cached 1-step-lag CB finger reaction wrenches (applied before arm
        # step, computed at the *end* of the previous box step).
        if mode == "cb" and not grip_active[0]:
            for li, w_cached in [(_l_idx, cached_w_left),
                                 (_r_idx, cached_w_right)]:
                if li in arm_solver._f_ext:
                    arm_solver._f_ext[li] = arm_solver._f_ext[li] + w_cached
                else:
                    arm_solver.set_external_force(li, w_cached)
        arm_solver.step()
        robot.enforce_mimic()

        fk = robot.forward_kinematics()
        palm_pos = fk[palm_idx].translation.copy()

        fingers_closed = robot.q[4] < 0.003
        gripping_now = (sim_time[0] >= _LIFT_START_T
                        and sim_time[0] < _RELEASE_START
                        and fingers_closed)

        if mode == "rigid":
            # Rigid box physics
            box_solver.clear_external_forces()
            for li, w in contact.compute_rbd_contact_forces(box_solver).items():
                box_solver.set_external_force(li, w)
            box_solver.step()

            if not grip_active[0] and box_robot.q[2] < 0.045 and sim_time[0] < _LIFT_START_T:
                box_robot.q[:2]   = box_ground_xy
                box_robot.qd[:2]  = 0.0
                box_robot.q[3:6]  = 0.0
                box_robot.qd[3:6] = 0.0

            if gripping_now:
                if not grip_active[0]:
                    grip_active[0]   = True
                    grip_palm_pos0[:] = palm_pos
                    grip_box_pos0 = box_robot.q[:3].copy()
                    grip_box_rot0 = box_robot.q[3:6].copy()
                    grip_offset   = grip_box_pos0 - grip_palm_pos0
                target_pos = palm_pos + grip_offset
                palm_vel = ((palm_pos - prev_palm_pos[0]) / dt
                            if prev_palm_pos[0] is not None else np.zeros(3))
                box_robot.q[:3]   = target_pos
                box_robot.qd[:3]  = palm_vel
                box_robot.q[3:6]  = grip_box_rot0
                box_robot.qd[3:6] = 0.0
            elif sim_time[0] >= _RELEASE_START and grip_active[0]:
                grip_active[0] = False

        else:  # cb
            if gripping_now:
                # Kinematic grip. Skip cb_solver.step entirely — otherwise
                # its internal reduced-coordinate integrator keeps
                # accumulating gravity even though we overwrite x/v, so at
                # release q_r_dot is wildly non-zero and the body flies off.
                if not grip_active[0]:
                    grip_active[0]    = True
                    grip_palm_pos0[:] = palm_pos
                    grip_body_x0[0]   = cb_body.x.copy()
                    # Re-seat CB internal state so that on future free
                    # steps it starts from "no internal motion".
                    if cb_body.q_r is not None:
                        cb_body.q_r[:] = 0.0
                delta_palm = palm_pos - grip_palm_pos0
                cb_body.x[:] = grip_body_x0[0] + delta_palm
                palm_vel = ((palm_pos - prev_palm_pos[0]) / dt
                            if prev_palm_pos[0] is not None else np.zeros(3))
                cb_body.v[:] = palm_vel
                cached_w_left[:]  = 0.0
                cached_w_right[:] = 0.0
            else:
                # Free dynamics. Skip finger penalty contact once the
                # release phase has started — the box has just been
                # unclasped, so it's still geometrically inside the
                # finger slab and ``_compute_finger_forces`` would inject
                # a huge spurious impulse ejecting it sideways.
                if sim_time[0] >= _RELEASE_START and grip_active[0]:
                    grip_active[0] = False
                if sim_time[0] < _RELEASE_START:
                    f_fem, l_i, w_l, r_i, w_r = _compute_finger_forces(
                        cb_body.x, cb_body.v, fk, robot,
                    )
                    cb_solver.step(dt=dt, extra_forces={0: f_fem})
                    cached_w_left, cached_w_right = w_l, w_r
                else:
                    cb_solver.step(dt=dt)
                    cached_w_left[:]  = 0.0
                    cached_w_right[:] = 0.0
                _apply_ground_projection(cb_body)

                # Ground XY hold before grip kicks in (box sits on floor).
                if sim_time[0] < _LIFT_START_T:
                    com_xy   = cb_body.x[:, :2].mean(axis=0)
                    delta_xy = box_ground_xy - com_xy
                    if np.linalg.norm(delta_xy) > 1e-6:
                        cb_body.x[:, :2] += delta_xy
                        cb_body.v[:, :2]  = 0.0

        # ── MPM step driven by the current box pose ──
        new_center, new_half = _box_collider_pose()
        mpm_collider.velocity = (new_center - mpm_collider.center) / dt
        mpm_collider.center = new_center
        mpm_collider.half_extent = new_half
        try:
            mpm_solver.step(dt)
        except (IndexError, np.linalg.LinAlgError):
            # Runaway particle escaped the grid OR the plastic projection
            # couldn't converge (rare — usually a single bad F_trial).
            # Skip this MPM tick and carry on; arm physics is unaffected.
            pass

        prev_palm_pos[0] = palm_pos
        sim_time[0] += dt

    # ── Headless ─────────────────────────────────────────────────
    if headless:
        total_dur   = sum(d for _, d in _PHASES) + 2.0
        total_steps = int(total_dur / dt) + substeps
        t_wall0 = time.time()
        for fr in range(total_steps // substeps + 1):
            for _ in range(substeps):
                step_once()
            if fr % 200 == 0:
                com = _box_com()
                z_top = float(mpm_particles.x[:, 2].max())
                xy_ext = float(np.ptp(mpm_particles.x[:, :2], axis=0).mean())
                ph = _PHASE_NAMES[min(phase_idx, len(_PHASE_NAMES) - 1)]
                print(f"t={sim_time[0]:6.3f}s  phase={ph:<8}  "
                      f"box=({com[0]:+.3f},{com[1]:+.3f},{com[2]:+.3f})  "
                      f"pile z_top={z_top:.3f}  xy_ext={xy_ext:.3f}")
        print(f"\nWall time: {time.time()-t_wall0:.2f} s")
        return

    # ── GUI ──────────────────────────────────────────────────────
    from robosim.viz.viewer import SimViewer
    from robosim.viz.scene_renderer import RobotRenderer

    viewer = SimViewer(
        title=f"RoboSim — MPM Grasp-Drop [{mode.upper()} / {material.upper()}]",
        window_size=(1280, 800),
        background=(0.08, 0.08, 0.10),
    )
    viewer.initialize()

    arm_renderer = RobotRenderer(robot, viewer)
    arm_renderer.setup()

    if mode == "rigid":
        box_renderer = RobotRenderer(box_robot, viewer)
        box_renderer.setup()
    else:
        viewer.add_mesh("cb_box", cb_body.x, surf_faces,
                        color=BOX_COLOR, opacity=1.0)

    viewer.add_particles(
        "mpm", mpm_particles.x, radius=0.008,
        per_vertex_color=_height_colors(
            mpm_particles.x[:, 2], PILE_SIZE[2] * 2.0,
        ),
    )

    def frame_step(frame: int) -> None:
        for _ in range(substeps):
            step_once()

        arm_renderer.update()
        vm_max = 0.0
        if mode == "rigid":
            box_renderer.update()
        else:
            viewer.update_mesh_vertices("cb_box", cb_body.x)
            try:
                from robosim.physics.fem.assembly import batch_von_mises
                vm = batch_von_mises(
                    cb_body.mesh, cb_body.x, cb_body.material,
                    cb_body._dN_list, cb_body._volumes,
                )
                vm_max = max(float(vm.max()), 1.0)
                viewer.update_mesh_color("cb_box", _stress_color(vm, vm_max))
            except Exception:
                pass

        viewer.update_particles(
            "mpm", mpm_particles.x,
            per_vertex_color=_height_colors(
                mpm_particles.x[:, 2], PILE_SIZE[2] * 2.0,
            ),
        )

        ph = _PHASE_NAMES[min(phase_idx, len(_PHASE_NAMES) - 1)]
        com = _box_com()
        z_top = float(mpm_particles.x[:, 2].max())
        xy_ext = float(np.ptp(mpm_particles.x[:, :2], axis=0).mean())
        stress_line = (f"\nsigma_vm peak: {vm_max:.0f} Pa"
                       if mode == "cb" and vm_max > 0 else "")
        viewer.add_text(
            f"t = {sim_time[0]:6.3f} s    Phase: {ph}    "
            f"[{mode.upper()} / {material.upper()}]\n"
            f"Box CoM : ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})\n"
            f"Pile z_top: {z_top:.3f}    xy extent: {xy_ext:.3f}"
            f"{stress_line}\n"
            f"grip: {'ON' if grip_active[0] else 'off'}    "
            f"q_fing: {robot.q[4]:+.3f} m\n"
            f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
        )

    viewer.add_callback(frame_step)
    viewer.show()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["rigid", "cb"], default="rigid",
                    help="box physics: rigid RBD (default) or CB deformable")
    ap.add_argument("--material", choices=["clay", "sand", "metal", "elastic"],
                    default="clay")
    ap.add_argument("--n-modes", type=int, default=10,
                    help="CB normal modes to keep (cb mode only)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    if not args.headless:
        import taichi as ti
        ti.init(arch=ti.metal)

    run(mode=args.mode, material=args.material,
        headless=args.headless, n_modes=args.n_modes)


if __name__ == "__main__":
    main()
