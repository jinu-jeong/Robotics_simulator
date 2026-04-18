"""Grasp Demo — arm6_gripper picks up a box (rigid / FEM / CB modes).

Robot: arm6_gripper (4 revolute arm joints + 2 prismatic finger joints)
       Approach-aligned gripper (80×10×20 mm fingers extend along world X,
       50×150×40 mm palm).  Wrist compensation keeps palm local-X = world X
       throughout all phases (fingers always point toward the box).

Box physics modes (``--mode``):
  rigid   (default)  RBD free-floating box, SAT box-box contact.
  fem                Hex8 finite-element deformable body (FEMSolver).
  cb                 Craig-Bampton reduced-order deformable body (fastest
                     deformable mode — one LU factorisation at startup).

Grasp sequence (in simulation time):
  HOME     (0.0 – 1.5 s)  Arm straight up, fingers open
  FOLD     (1.5 – 2.7 s)  Fold elbow up to keep arc compact
  APPROACH (2.7 – 5.2 s)  Swing arm down to box height
  NEAR     (5.2 – 6.7 s)  Advance palm to box (24 mm overlap = 30 % tip grasp)
  CLOSE    (6.7 – 8.2 s)  Close fingers (penalty contact holds them)
  LIFT     (8.2 – 11.2 s) Raise arm — kinematic constraint lifts the box

Usage:
  python examples/grasp_demo.py
  python examples/grasp_demo.py --mode fem
  python examples/grasp_demo.py --mode cb --n-modes 15
  python examples/grasp_demo.py --mode cb --headless
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
from robosim.physics.rbd.algorithms import gravity_torques
from robosim.physics.rbd.solver import RBDSolver

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

_URDF = Path(__file__).parent / "urdf" / "arm6_gripper" / "arm6_gripper.urdf"

# ═══════════════════════════════════════════════════════════════════
# Grasp poses — q = [pan, shoulder, elbow, wrist, q_left, q_right]
#
# Wrist compensation: q_wrist = -(q_shoulder + q_elbow) keeps the palm
# horizontal throughout the motion.
# ═══════════════════════════════════════════════════════════════════

HOME_Q     = np.array([ 0.0, -1.571,  0.000, +1.571,  0.040, -0.040])
FOLD_Q     = np.array([ 0.0, -1.400,  2.000, -0.600,  0.040, -0.040])
APPROACH_Q = np.array([ 0.0, -0.365,  2.047, -1.682,  0.040, -0.040])
NEAR_Q     = np.array([ 0.0, -0.370,  1.909, -1.539,  0.040, -0.040])
CLOSE_Q    = np.array([ 0.0, -0.370,  1.909, -1.539, -0.020,  0.020])
LIFT_Q     = np.array([ 0.0, -0.878,  2.080, -1.202, -0.020,  0.020])

_PHASES = [
    (HOME_Q,     1.5),
    (FOLD_Q,     1.2),
    (APPROACH_Q, 2.5),
    (NEAR_Q,     1.5),
    (CLOSE_Q,    1.5),
    (LIFT_Q,     3.0),
]


# ═══════════════════════════════════════════════════════════════════
# Smooth-step helper
# ═══════════════════════════════════════════════════════════════════

def _smooth(t: float) -> float:
    """Quintic smooth-step: 0 → 1 with zero 1st AND 2nd derivatives at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ═══════════════════════════════════════════════════════════════════
# PD gains
# ═══════════════════════════════════════════════════════════════════

_KP = np.array([280.0,  480.0,  200.0,   75.0,   5e4,    5e4  ])
_KD = np.array([ 35.0,   50.0,   20.0,    2.0,  100.0,  100.0])

# Finger contact half-extents and spring gains (overridden per-mode in run())
_F_HX, _F_HY, _F_HZ = 0.040, 0.005, 0.010   # matches actual finger geometry (80×10×20 mm)
_K_FINGER = 1e4    # N/m
_C_FINGER  = 50.0  # N·s/m


def _control(robot, q_target: np.ndarray) -> np.ndarray:
    """PD + gravity compensation.  Returns joint-space torques."""
    q_err  = q_target - robot.q
    qd_err = -robot.qd
    g_tau  = gravity_torques(robot, robot.q)
    return _KP * q_err + _KD * qd_err + g_tau


# ═══════════════════════════════════════════════════════════════════
# Deformable-body contact helpers  (used by FEM + CB modes)
# ═══════════════════════════════════════════════════════════════════

def _apply_ground_projection(body, mu_friction: float = 0.0) -> None:
    """Position-projection ground contact: push nodes above z=0, kill sinking.

    Both FEMSolver and CraigBamptonSolver absorb external state changes
    before integrating, so direct modification of ``body.x`` / ``body.v``
    is safe.  Position-projection is far more stable than spring forces
    for the free-floating CB mode (rotation-Kabsch basis amplifies
    penalty impulses otherwise).
    """
    below = body.x[:, 2] < 0.0
    if not below.any():
        return
    body.x[below, 2] = 0.0
    body.v[below, 2] = np.maximum(body.v[below, 2], 0.0)
    if mu_friction > 0.0:
        body.v[below, :2] *= max(0.0, 1.0 - mu_friction * 0.1)


def _apply_finger_projection(body, arm_fk, robot, dt):
    """Position-projection finger contact for FEM bodies.

    Mirrors _apply_ground_projection: after the FEM step, any node that has
    moved inside a finger slab is projected to the finger's inner face and its
    inward velocity component is killed.  The impulse (velocity change × mass)
    is accumulated as a Newton 3rd-law reaction wrench on each finger link.

    This avoids injecting large penalty forces into the implicit FEM solver,
    which causes numerical blow-up when the finger closes 20+ mm past the box
    face (kinematic trajectory is not limited by contact in FEM mode).

    Returns
    -------
    l_idx, w_left, r_idx, w_right  – same signature as _compute_finger_forces
    """
    l_idx = robot.link_index("left_finger")
    r_idx = robot.link_index("right_finger")

    _geom_off = np.array([0.020, 0.0, 0.020])
    p_l = arm_fk[l_idx].translation + _geom_off
    p_r = arm_fk[r_idx].translation + _geom_off
    T_l_inv = arm_fk[l_idx].inverse()
    T_r_inv = arm_fk[r_idx].inverse()

    inner_l = p_l[1] - _F_HY   # +Y face of left finger (facing box)
    inner_r = p_r[1] + _F_HY   # −Y face of right finger (facing box)

    # Per-node mass from the lumped mass matrix diagonal (X-DOF = Y-DOF = Z-DOF)
    M_diag  = body._M.diagonal()
    m_nodes = M_diag[0::3]       # mass for X-DOF of each node ≡ node mass

    w_left  = np.zeros(6)
    w_right = np.zeros(6)

    for i in range(body.x.shape[0]):
        xi = body.x[i]

        # Skip nodes outside the finger slab (X-extent and Z-extent)
        if not (abs(xi[0] - p_l[0]) < _F_HX and abs(xi[2] - p_l[2]) < _F_HZ):
            continue

        mi = m_nodes[i]

        # ── Left finger: project nodes with xi[1] > inner_l ──────────
        if xi[1] > inner_l:
            body.x[i, 1] = inner_l                  # push to finger surface
            v_old = body.v[i, 1]
            body.v[i, 1] = min(v_old, 0.0)          # kill +Y (inward) velocity
            dv = body.v[i, 1] - v_old               # velocity change (≤ 0)
            # Reaction on arm: Newton 3rd law (impulse / dt = force)
            f_react_world  = np.array([0.0, -mi * dv / dt, 0.0])
            f_react_link   = T_l_inv.apply_vector(f_react_world)
            p_contact_link = T_l_inv.apply_point(xi)
            tau_link = np.cross(p_contact_link, f_react_link)
            w_left[:3] += tau_link
            w_left[3:]  += f_react_link

        # ── Right finger: project nodes with xi[1] < inner_r ─────────
        if xi[1] < inner_r:
            body.x[i, 1] = inner_r
            v_old = body.v[i, 1]
            body.v[i, 1] = max(v_old, 0.0)          # kill −Y (inward) velocity
            dv = body.v[i, 1] - v_old               # velocity change (≥ 0)
            f_react_world  = np.array([0.0, -mi * dv / dt, 0.0])
            f_react_link   = T_r_inv.apply_vector(f_react_world)
            p_contact_link = T_r_inv.apply_point(xi)
            tau_link = np.cross(p_contact_link, f_react_link)
            w_right[:3] += tau_link
            w_right[3:]  += f_react_link

    return l_idx, w_left, r_idx, w_right


def _compute_finger_forces(bx, bv, arm_fk, robot):
    """Per-node penalty contact — used for CB mode only.

    CB's modal integrator handles large stiffness without blow-up (the
    reduced basis naturally damps high-frequency modes).  Returns:
      f_fem   – flat (3n,) force array  (extra_forces to CB solver)
      l_idx, w_left, r_idx, w_right  – Newton 3rd-law arm wrenches
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
        xi = bx[i]
        vi = bv[i]

        if abs(xi[0] - p_l[0]) < _F_HX and abs(xi[2] - p_l[2]) < _F_HZ:
            pen = xi[1] - (p_l[1] - _F_HY)
            if pen > 0.0:
                fn = _K_FINGER * pen + _C_FINGER * max(0.0, vi[1])
                f_fem[i * 3 + 1] -= fn
                f_react_world  = np.array([0.0, fn, 0.0])
                f_react_link   = T_l_inv.apply_vector(f_react_world)
                p_contact_link = T_l_inv.apply_point(xi)
                tau_link = np.cross(p_contact_link, f_react_link)
                w_left[:3] += tau_link
                w_left[3:]  += f_react_link

        if abs(xi[0] - p_r[0]) < _F_HX and abs(xi[2] - p_r[2]) < _F_HZ:
            pen = (p_r[1] + _F_HY) - xi[1]
            if pen > 0.0:
                fn = _K_FINGER * pen + _C_FINGER * max(0.0, -vi[1])
                f_fem[i * 3 + 1] += fn
                f_react_world  = np.array([0.0, -fn, 0.0])
                f_react_link   = T_r_inv.apply_vector(f_react_world)
                p_contact_link = T_r_inv.apply_point(xi)
                tau_link = np.cross(p_contact_link, f_react_link)
                w_right[:3] += tau_link
                w_right[3:]  += f_react_link

    return f_fem, l_idx, w_left, r_idx, w_right


def _stress_color(vm: np.ndarray, vmax: float) -> np.ndarray:
    """Map per-node von Mises stress to an RGB jet colormap.

    Same colour ramp used by examples/drop_test.py for consistency:
      blue → cyan → green → yellow → red as stress grows.
    """
    t = np.clip(vm / max(vmax, 1.0), 0, 1)
    r = np.clip(1.5 - np.abs(t - 0.75) * 4.0, 0, 1)
    g = np.clip(1.5 - np.abs(t - 0.50) * 4.0, 0, 1)
    b = np.clip(1.5 - np.abs(t - 0.25) * 4.0, 0, 1)
    return np.column_stack([r, g, b]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Main simulation
# ═══════════════════════════════════════════════════════════════════

def run(mode: str = "rigid", headless: bool = False, n_modes: int = 10) -> None:
    """Run the grasp demo in one of three box-physics modes.

    Parameters
    ----------
    mode : "rigid" | "fem" | "cb"
    headless : skip GUI if True
    n_modes : CB normal modes to keep (only used when mode=="cb")
    """
    assert mode in ("rigid", "fem", "cb"), f"unknown mode: {mode}"

    # Mode-dependent finger contact stiffness
    global _K_FINGER, _C_FINGER
    if mode == "fem":
        _K_FINGER = 1e3   # with _PEN_CAP, max per-node force = 5 N → stable
        _C_FINGER  = 30.0
    elif mode == "cb":
        _K_FINGER = 1e4
        _C_FINGER  = 50.0

    dt       = 0.001
    substeps = 10

    # Box geometry: 8×8×8 cm cube at x=0.600, resting on ground (bottom at z=0)
    BOX_SIZE     = np.array([0.080, 0.080, 0.080])
    BOX_CENTRE   = np.array([0.600, 0.000, 0.040])
    BOX_MASS     = 1.0
    BOX_COLOR    = (np.array([0.85, 0.15, 0.15]) if mode != "cb"
                    else np.array([0.20, 0.55, 0.90]))  # blue for CB

    # ── Arm (rigid body, all modes) ───────────────────────────────
    robot = parse_urdf(_URDF)
    robot.gravity = np.array([0.0, 0.0, -9.81])
    robot.q[:]   = HOME_Q
    robot.enforce_mimic()
    robot.qd[:]  = 0.0

    arm_solver = RBDSolver(robot=robot)
    arm_solver.initialize(dt=dt)

    # ═══════════════ Mode-dependent box construction ═══════════════
    box_robot     = None   # only used in rigid mode
    box_solver    = None
    fem_body      = None   # used in FEM/CB mode
    fem_solver    = None
    cb_body       = None
    cb_solver     = None
    surf_faces    = None
    fem_dt        = dt
    fem_every     = 1

    if mode == "rigid":
        # 6-DOF free-floating RBD box
        box_robot = create_free_box(
            name="target_box",
            size=tuple(BOX_SIZE),
            mass=BOX_MASS,
            position=BOX_CENTRE.copy(),
            color=np.append(BOX_COLOR, 1.0),
        )
        box_robot.gravity = np.array([0.0, 0.0, -9.81])
        box_solver = RBDSolver(robot=box_robot)
        box_solver.initialize(dt=dt)

    else:  # fem or cb
        from robosim.physics.fem.assembly import batch_von_mises
        from robosim.physics.fem.materials import CorotationalElastic
        from robosim.physics.fem.mesh import FEMesh

        # Hex8 mesh with lower corner at (x-hx, y-hy, 0)
        box_origin = BOX_CENTRE - BOX_SIZE * 0.5
        box_origin[2] = 0.0   # sit on ground
        # FEM: fine mesh (8×8×8) — position-projection contact is stable at any density
        # CB : coarse mesh (4×4×4) — penalty contact needs k_element >> k_contact;
        #       larger elements (20 mm) give k_elem ≈ 2000 N/m > k=1e4 OK with modal basis
        mesh = FEMesh.create_hex_box(
            origin=box_origin,
            size=BOX_SIZE,
            divisions=(8, 8, 8) if mode == "fem" else (4, 4, 4),
        )
        box_density = BOX_MASS / float(np.prod(BOX_SIZE))
        material    = CorotationalElastic(young=1e5, poisson=0.45)
        surf_faces  = mesh.extract_surface()

        if mode == "fem":
            from robosim.physics.fem.solver import DeformableBody, FEMSolver
            fem_body = DeformableBody(
                name="fem_box", mesh=mesh, material=material,
                density=box_density,
            )
            # FEM runs at 5 ms (vs arm 1 ms) with 1-iter linearised step;
            # Newton iters are too slow at 1 ms for interactive rates.
            fem_dt    = 0.005
            fem_every = max(1, round(fem_dt / dt))
            fem_solver = FEMSolver(
                bodies=[fem_body],
                gravity=np.array([0.0, 0.0, -9.81]),
                damping=0.5,
                max_newton_iters=1,
            )
            fem_solver.initialize(dt=fem_dt)

        else:  # cb
            from robosim.physics.fem.reduced import (
                CraigBamptonBody, CraigBamptonSolver,
            )
            print(f"\n  Building Craig-Bampton basis  "
                  f"(n_nodes={mesh.n_nodes}, n_modes={n_modes}) …")
            t_b0 = time.time()
            cb_body = CraigBamptonBody(
                mesh=mesh, material=material, density=box_density,
                n_modes=n_modes,
                gravity=np.array([0.0, 0.0, -9.81]),
                damping=0.5, name="cb_box",
            )
            print(f"  CB basis built in {time.time()-t_b0:.2f} s  "
                  f"(reduced DOFs: {cb_body._n_r})")
            cb_solver = CraigBamptonSolver(bodies=[cb_body])
            cb_solver.initialize(dt=dt)

    # ── Contact solver ────────────────────────────────────────────
    contact = ContactSolver(
        ground=GroundPlane(height=0.0),
        params=ContactParams(stiffness=1e3, damping=50, friction_mu=0.0),
    )
    contact.register_rbd(arm_solver, robot_id="arm")

    if mode == "rigid":
        contact.register_rbd(box_solver, robot_id="box")
        for arm_link in ["base_link", "link_1", "link_2",
                         "link_3", "link_4", "palm_link"]:
            contact.add_cross_filter("arm", arm_link, "box", "target_box_body")
    # FEM / CB: finger contact is handled by _compute_finger_forces (penalty
    # forces injected as extra_forces before each FEM/CB solver step).

    # ── Visualisation ─────────────────────────────────────────────
    if not headless:
        import taichi as ti  # noqa: F401
        from robosim.viz.scene_renderer import RobotRenderer
        from robosim.viz.viewer import SimViewer

        title = f"RoboSim — Grasp Demo [{mode.upper()}]  (arm6_gripper)"
        if mode == "cb":
            title += f"  n_modes={n_modes}"
        viewer = SimViewer(title=title, window_size=(1280, 800))
        viewer.initialize()

        arm_renderer = RobotRenderer(robot, viewer)
        arm_renderer.setup()

        if mode == "rigid":
            box_renderer = RobotRenderer(box_robot, viewer)
            box_renderer.setup()
        elif mode == "fem":
            viewer.add_mesh("fem_box", fem_body.x, surf_faces,
                            color=BOX_COLOR, opacity=1.0)
        else:  # cb
            viewer.add_mesh("cb_box", cb_body.x, surf_faces,
                            color=BOX_COLOR, opacity=1.0)

    # ── Phase interpolation state ────────────────────────────────
    sim_time       = 0.0
    phase_idx      = 0
    phase_start_t  = 0.0
    q_phase_start  = HOME_Q.copy()
    q_target       = HOME_Q.copy()

    def _advance_phases(t: float) -> None:
        nonlocal phase_idx, phase_start_t, q_phase_start, q_target
        if phase_idx >= len(_PHASES):
            q_target[:] = _PHASES[-1][0]
            return
        tgt_q, dur = _PHASES[phase_idx]
        alpha = _smooth((t - phase_start_t) / dur)
        q_target[:] = q_phase_start + alpha * (tgt_q - q_phase_start)
        if (t - phase_start_t) >= dur:
            phase_idx     += 1
            phase_start_t  = t
            q_phase_start  = tgt_q.copy()

    _phase_names = ["HOME", "FOLD", "APPROACH", "NEAR", "CLOSE", "LIFT", "HOLD"]

    # ── Helpers for reporting current box state in a unified way ─
    def _box_com() -> np.ndarray:
        if mode == "rigid":
            return box_robot.q[:3].copy()
        if mode == "fem":
            return fem_body.x.mean(axis=0)
        return cb_body.x.mean(axis=0)

    def _box_bottom_z() -> float:
        if mode == "rigid":
            return float(box_robot.q[2]) - 0.040  # centre - half extent
        if mode == "fem":
            return float(fem_body.x[:, 2].min())
        return float(cb_body.x[:, 2].min())

    # ── Constraint grip state ───────────────────────────────────
    _LIFT_START_T   = sum(d for _, d in _PHASES[:5])
    _grip_active    = False
    _grip_palm_pos0 = np.zeros(3)
    _grip_box_pos0  = np.zeros(3)     # rigid mode only
    _grip_box_rot0  = np.zeros(3)     # rigid mode only
    _grip_offset    = np.zeros(3)     # rigid mode only
    _grip_body_x0   = None            # FEM/CB mode: snapshot of all node pos
    _prev_palm_pos  = None
    palm_idx        = robot.link_index("palm_link")

    # Ground-hold: freeze XY while box is on the ground
    _box_ground_xy = BOX_CENTRE[:2].copy()
    _box_ground_rot = np.zeros(3)

    # (ground friction is now a fixed 0.9 in _apply_ground_projection)

    # ── Debug tracking ──────────────────────────────────────────
    _prev_phase_idx = -1
    _lift_debug_t   = 0.0
    _LIFT_DEBUG_INTERVAL = 0.5

    def _debug_state(label: str) -> None:
        com   = _box_com()
        z_lo  = _box_bottom_z()
        lift  = max(0.0, z_lo)
        print(f"\n{'─'*60}")
        print(f"  [{label}]  t = {sim_time:.3f} s   mode={mode}")
        print(f"  Box CoM      : ({com[0]:+.4f}, {com[1]:+.4f}, {com[2]:+.4f})")
        print(f"  Box bottom Z : {z_lo:+.4f} m   (lifted {lift:.4f} m)")
        n_cf = len(contact._last_forces)
        print(f"  Arm contacts : {n_cf}")
        if mode != "rigid":
            body = fem_body if mode == "fem" else cb_body
            y_ext = float(body.x[:, 1].max() - body.x[:, 1].min())
            x_ext = float(body.x[:, 0].max() - body.x[:, 0].min())
            z_ext = float(body.x[:, 2].max() - body.x[:, 2].min())
            print(f"  Box extents  : X={x_ext*1e3:.1f}mm  Y={y_ext*1e3:.1f}mm  Z={z_ext*1e3:.1f}mm"
                  f"  (rest: 80/80/80 mm)")
        print(f"  q = {robot.q}")
        print(f"{'─'*60}")

    # ── Per-mode box step ───────────────────────────────────────
    _arm_step_count = 0

    # Cache reaction wrenches from FEM finger contact (1-step lag).
    # Computed by _compute_finger_forces(); applied to arm_solver next substep.
    _cached_l_idx   = robot.link_index("left_finger")
    _cached_r_idx   = robot.link_index("right_finger")
    _cached_w_left  = np.zeros(6)
    _cached_w_right = np.zeros(6)

    def _step_box_physics(fk, palm_pos):
        """Advance the box one physics step; mode-specific."""
        nonlocal _grip_active, _grip_palm_pos0, _grip_box_pos0
        nonlocal _grip_box_rot0, _grip_offset, _grip_body_x0
        nonlocal _prev_palm_pos, _arm_step_count
        nonlocal _cached_l_idx, _cached_r_idx, _cached_w_left, _cached_w_right

        if mode == "rigid":
            # Contact forces on box (reaction from arm + ground)
            box_solver.clear_external_forces()
            box_wrenches = contact.compute_rbd_contact_forces(box_solver)
            for li, w in box_wrenches.items():
                box_solver.set_external_force(li, w)
            box_solver.step()

            # Ground-hold
            if not _grip_active and box_robot.q[2] < 0.045:
                box_robot.q[:2]   = _box_ground_xy
                box_robot.qd[:2]  = 0.0
                box_robot.q[3:6]  = _box_ground_rot
                box_robot.qd[3:6] = 0.0

            # Constraint grip at LIFT
            if sim_time >= _LIFT_START_T and robot.q[4] < 0.003:
                if not _grip_active:
                    _grip_active    = True
                    _grip_palm_pos0 = palm_pos.copy()
                    _grip_box_pos0  = box_robot.q[:3].copy()
                    _grip_box_rot0  = box_robot.q[3:6].copy()
                    _grip_offset    = _grip_box_pos0 - _grip_palm_pos0

                target_pos = palm_pos + _grip_offset
                if target_pos[2] > _grip_box_pos0[2]:
                    if _prev_palm_pos is not None:
                        palm_vel = (palm_pos - _prev_palm_pos) / dt
                    else:
                        palm_vel = np.zeros(3)
                    box_robot.q[:3]   = target_pos
                    box_robot.qd[:3]  = palm_vel
                    box_robot.q[3:6]  = _grip_box_rot0
                    box_robot.qd[3:6] = 0.0

        else:
            # ---- FEM / CB ----
            body    = fem_body if mode == "fem" else cb_body
            solver  = fem_solver if mode == "fem" else cb_solver
            this_dt = fem_dt    if mode == "fem" else dt

            _arm_step_count += 1
            run_fem_step = (mode == "cb") or (_arm_step_count % fem_every == 0)

            if run_fem_step:
                if not _grip_active:
                    if mode == "fem":
                        # ── FEM: position-projection contact (post-step) ──────
                        # Step first (no extra forces), then project nodes back
                        # to finger surfaces.  Avoids injecting large forces into
                        # the implicit solver (which blows up with 1 Newton iter
                        # when fingers ghost 20+ mm past the box face).
                        solver.step(dt=this_dt)
                        l_i, w_l, r_i, w_r = _apply_finger_projection(
                            body, fk, robot, this_dt,
                        )
                    else:
                        # ── CB: penalty contact (pre-step) ────────────────────
                        # CB's modal integrator handles large stiffness stably.
                        f_fem, l_i, w_l, r_i, w_r = _compute_finger_forces(
                            body.x, body.v, fk, robot,
                        )
                        solver.step(dt=this_dt, extra_forces={0: f_fem})

                    _cached_l_idx, _cached_w_left  = l_i, w_l
                    _cached_r_idx, _cached_w_right = r_i, w_r
                    _apply_ground_projection(body)
                else:
                    # Grip active: no finger forces, only ground
                    solver.step(dt=this_dt)
                    _apply_ground_projection(body)
                    _cached_w_left[:]  = 0.0
                    _cached_w_right[:] = 0.0

            # Ground-hold (XY drift suppression while box is on ground)
            if not _grip_active:
                com_xy   = body.x[:, :2].mean(axis=0)
                delta_xy = _box_ground_xy - com_xy
                if np.linalg.norm(delta_xy) > 1e-6:
                    body.x[:, :2] += delta_xy
                    body.v[:, :2]  = 0.0

            # Constraint grip at LIFT
            if sim_time >= _LIFT_START_T and robot.q[4] < 0.003:
                if not _grip_active:
                    _grip_active    = True
                    _grip_palm_pos0 = palm_pos.copy()
                    _grip_body_x0   = body.x.copy()

                delta_palm = palm_pos - _grip_palm_pos0
                body.x = _grip_body_x0 + delta_palm   # rigid translation

                if _prev_palm_pos is not None:
                    palm_vel = (palm_pos - _prev_palm_pos) / dt
                else:
                    palm_vel = np.zeros(3)
                if palm_vel[2] > 0.0:
                    body.v[:] = palm_vel

    # ── Main step callback ──────────────────────────────────────
    def step(frame: int = 0) -> None:
        nonlocal sim_time, _prev_palm_pos, _prev_phase_idx, _lift_debug_t

        for _ in range(substeps):
            _advance_phases(sim_time)

            # Arm PD + gravity comp + ground contact
            arm_solver.clear_external_forces()
            arm_solver.tau = _control(robot, q_target)
            arm_wrenches = contact.compute_rbd_contact_forces(arm_solver)
            for li, w in arm_wrenches.items():
                arm_solver.set_external_force(li, w)

            # FEM/CB mode: apply cached reaction wrenches from last FEM contact
            # step (1-step lag — standard practice, avoids circular dependency).
            # Populated by _compute_finger_forces() in _step_box_physics().
            if mode != "rigid" and not _grip_active:
                for li, w_cached in [(_cached_l_idx, _cached_w_left),
                                      (_cached_r_idx, _cached_w_right)]:
                    if li in arm_solver._f_ext:
                        arm_solver._f_ext[li] = arm_solver._f_ext[li] + w_cached
                    else:
                        arm_solver.set_external_force(li, w_cached)

            arm_solver.step()
            robot.enforce_mimic()

            fk = robot.forward_kinematics()
            palm_pos = fk[palm_idx].translation.copy()

            _step_box_physics(fk, palm_pos)

            _prev_palm_pos = palm_pos
            sim_time += dt

        # Phase transition debug print
        if phase_idx != _prev_phase_idx:
            name = _phase_names[min(_prev_phase_idx + 1, len(_phase_names) - 1)]
            _debug_state(f"PHASE START: {name}")
            _prev_phase_idx = phase_idx

        # Periodic LIFT debug print
        is_lift_phase = (phase_idx == 5 or
                         (phase_idx >= len(_PHASES) and sim_time >= _LIFT_START_T))
        if is_lift_phase and sim_time >= _lift_debug_t:
            _debug_state(
                f"LIFT t={sim_time:.2f}s grip={'ON' if _grip_active else 'OFF'}")
            _lift_debug_t = sim_time + _LIFT_DEBUG_INTERVAL

        # Update visualisation
        if not headless:
            arm_renderer.update()
            if mode == "rigid":
                box_renderer.update()
            else:
                # FEM / CB: update deformed vertex positions AND paint the
                # surface with per-node von Mises stress (jet colormap,
                # same as examples/drop_test.py).
                if mode == "fem":
                    body, mesh_name = fem_body, "fem_box"
                else:
                    body, mesh_name = cb_body, "cb_box"

                viewer.update_mesh_vertices(mesh_name, body.x)

                vm = batch_von_mises(
                    body.mesh, body.x, body.material,
                    body._dN_list, body._volumes,
                )
                vm_max = max(float(vm.max()), 1.0)
                viewer.update_mesh_color(
                    mesh_name, _stress_color(vm, vm_max))

            p_palm = fk[palm_idx].translation
            p_name = _phase_names[min(phase_idx, len(_phase_names) - 1)]
            com    = _box_com()
            z_lo   = _box_bottom_z()
            tag    = mode.upper() + (f"  n_modes={n_modes}" if mode == "cb" else "")
            stress_line = (f"\nσ_vm peak : {vm_max:.0f} Pa"
                           if mode != "rigid" else "")

            viewer.add_text(
                f"t = {sim_time:6.3f} s    Phase: {p_name}    [{tag}]\n"
                f"Palm  : ({p_palm[0]:+.3f}, {p_palm[1]:+.3f}, {p_palm[2]:+.3f})\n"
                f"Box CoM   : ({com[0]:+.3f}, {com[1]:+.3f}, {com[2]:+.3f})\n"
                f"Box bot Z : {z_lo:+.4f} m  (lifted {max(0.0, z_lo):.4f} m)"
                f"{stress_line}\n"
                f"q_fing: {robot.q[4]:.3f} m   grip: {'ON' if _grip_active else 'off'}\n"
                f"[LDrag=orbit  Scroll=zoom  ESC=quit]"
            )

    # ── Run ─────────────────────────────────────────────────────
    if headless:
        print("=" * 60)
        print(f"  RoboSim — Grasp Demo  mode={mode}"
              + (f"  n_modes={n_modes}" if mode == "cb" else "")
              + "  (headless)")
        print("=" * 60)
        total_dur   = sum(d for _, d in _PHASES)
        total_steps = int(total_dur / dt) + 200
        t_wall0     = time.time()
        for fr in range(total_steps // substeps + 1):
            step(fr)
        t_wall = time.time() - t_wall0

        com_f  = _box_com()
        z_lo_f = _box_bottom_z()
        lift_h = max(0.0, z_lo_f)

        _debug_state("FINAL STATE")
        print(f"\n{'='*60}")
        print(f"Mode       : {mode}"
              + (f"  (n_modes={n_modes})" if mode == "cb" else ""))
        print(f"Simulated  : {sim_time:.2f} s")
        print(f"Wall time  : {t_wall:.2f} s  ({sim_time/t_wall:.2f}× real-time)")
        print(f"Box CoM    : ({com_f[0]:.3f}, {com_f[1]:.3f}, {com_f[2]:.3f})")
        print(f"Box bot Z  : {z_lo_f:.4f} m   (lifted {lift_h:.4f} m)")
        print(f"Lift result: {'SUCCESS' if lift_h > 0.05 else 'FAIL — box not lifted'}")
        if _grip_active:
            print("[NOTE] Constraint grip active: box kinematically coupled to palm.")
    else:
        viewer.add_callback(step)
        viewer.show()


# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grasp Demo: arm6_gripper picks up a box "
                    "(rigid / FEM / CB modes)"
    )
    parser.add_argument(
        "--mode", choices=("rigid", "fem", "cb"), default="rigid",
        help="Box physics mode (default: rigid)",
    )
    parser.add_argument(
        "--n-modes", type=int, default=10,
        help="Craig-Bampton normal modes (cb mode only, default: 10)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI and print final result",
    )
    args = parser.parse_args()

    if not args.headless:
        import taichi as ti
        ti.init(arch=ti.metal)

    run(mode=args.mode, headless=args.headless, n_modes=args.n_modes)


if __name__ == "__main__":
    main()
