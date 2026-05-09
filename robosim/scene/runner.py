"""SimRunner — encapsulates the full simulation step loop.

Handles:
  • Robot RBD stepping with PD control + gravity compensation
  • FEM substeps (configurable fem_dt via FEM.dt_scale)
  • CB stepping (same dt as arm)
  • Ground contact via position projection (FEM/CB) or ContactSolver (rigid)
  • Robot ↔ deformable-body contact:
      FEM  →  position projection  (post-step, avoids large force injection)
      CB   →  penalty forces       (pre-step, stable with modal basis)
  • Kinematic grip — body follows palm FK once finger joints are closed
  • Trajectory advancement (Phase-based)
  • Viewer updates (headless or GUI)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from robosim.scene.scene import Scene
    from robosim.scene.handles import (
        RobotHandle, FEMBodyHandle, CBBodyHandle, RigidBodyHandle,
    )
    from robosim.control.phases import Trajectory


# ── Contact pair ──────────────────────────────────────────────────────────────

@dataclass
class _ContactPair:
    """Internal registration of a robot ↔ deformable body contact pair."""
    robot:         "RobotHandle"
    body:          "FEMBodyHandle | CBBodyHandle"
    contact_links: list[str]

    # Per-link geometry: (geom_offset_in_link_frame, half_extents)
    _geom: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False)

    # Cached reaction wrenches (1-step lag, standard practice)
    _cached_wrenches: dict[int, np.ndarray] = field(
        default_factory=dict, init=False, repr=False)

    # Kinematic grip state
    _grip_active: bool  = field(default=False, init=False)
    _grip_body_x0: np.ndarray | None = field(default=None, init=False)
    _grip_palm_pos0: np.ndarray | None = field(default=None, init=False)

    # Contact params for CB penalty mode
    k_contact: float = 1e4
    c_contact: float = 50.0

    def _build_geom(self) -> None:
        """Read collision box geometry from URDF to derive slab volumes."""
        robot_model = self.robot._model
        for link_name in self.contact_links:
            try:
                link = robot_model.links[robot_model.link_index(link_name)]
                col  = link.collisions[0]
                # size is full extents; half_extents = size/2
                half_ext = col.geometry.size / 2.0
                offset   = col.origin.translation.copy()
                self._geom[link_name] = (offset, half_ext)
            except (IndexError, AttributeError):
                # Fallback: tiny slab
                self._geom[link_name] = (np.zeros(3), np.array([0.01, 0.01, 0.01]))


# ══════════════════════════════════════════════════════════════════════════════
# SimRunner
# ══════════════════════════════════════════════════════════════════════════════

class SimRunner:
    """Runs the simulation loop for a :class:`Scene`.

    Instantiated and called by ``Scene.run()``; users typically do not
    need to interact with this class directly.

    Parameters
    ----------
    scene       : the parent Scene
    dt          : master timestep for the arm/rigid-body solver [s]
    substeps    : physics substeps per render/callback frame
    """

    def __init__(self, scene: "Scene", dt: float = 0.001, substeps: int = 10):
        from robosim.util.fps import FPSCounter
        self._scene     = scene
        self._dt        = dt
        self._substeps  = substeps
        self._time      = 0.0
        self._viewer_ctx: dict | None = None
        # FPS = one tick per outer loop iteration (after substeps + viewer
        # update). Accessible from user on_step callbacks via
        # scene._runner._fps for custom display.
        self._fps       = FPSCounter(smoothing=20)

    # ── public entry point ────────────────────────────────────────────────────

    def run(
        self,
        trajectories: list["Trajectory"] | None = None,
        duration: float = 12.0,
        viewer: bool = False,
        headless: bool = False,
        on_step: Callable | None = None,
    ) -> dict:
        """Run the simulation.

        Parameters
        ----------
        trajectories : phase-based motion plans (may be None for pure dynamics)
        duration     : total simulated time [s]
        viewer       : open GUI window
        headless     : skip GUI even if viewer=True (for CI / testing)
        on_step      : optional ``(scene, t) -> None`` callback, called once
                       per *frame* (not per substep)

        Returns
        -------
        dict with 'sim_time', 'wall_time', 'bodies' (handle → final state)
        """
        trajs = trajectories or []
        dt    = self._dt

        # --- initialise physics ---
        self._init_physics(dt)

        # --- initialise viewer ---
        _viewer_ctx = None
        if viewer and not headless:
            _viewer_ctx = self._init_viewer()

        # --- headless banner ---
        if headless:
            self._print_banner()

        t_wall0     = time.time()
        total_steps = int(duration / dt) + self._substeps

        for fr in range(total_steps // self._substeps + 1):
            if self._time > duration:
                break

            for _ in range(self._substeps):
                if self._time > duration:
                    break
                # ── advance trajectories ──
                for traj in trajs:
                    traj._step(dt)

                # ── re-pin finger PD targets for any active grip ──
                # (must run AFTER trajectory updates target_q, BEFORE PD reads
                #  it in _step_robot, otherwise the trajectory wins and the
                #  finger drives through the kinematically-locked body).
                self._apply_grip_locks()

                # ── step all robots (applies CACHED contact reactions before stepping) ──
                for rh in self._scene._robot_handles.values():
                    self._step_robot(rh, dt)

                # ── step all bodies (computes NEW contact reactions, caches for next tick) ──
                for bh in self._scene._body_handles.values():
                    self._step_body(bh, dt)

                # NOTE: ConstraintContactSolver also exposes
                # project_free_body_penetrations() for non-grasp scenarios
                # (e.g. a free box bouncing off an arm link), but it is
                # NOT called from the runner: for a symmetric two-finger
                # grasp the per-finger pushes cancel and the projection
                # cannot reduce the residual pen without a proper coupled
                # LCP/PGS solver.

                self._time += dt

            # ── frame callbacks ──
            if on_step is not None:
                on_step(self._scene, self._time)

            # ── viewer update ──
            if _viewer_ctx is not None:
                alive = self._update_viewer(_viewer_ctx, trajs)
                if not alive:
                    break   # user closed the window

            # ── frame timing (after the frame's heavy work) ──
            self._fps.tick()

        t_wall = time.time() - t_wall0

        if headless:
            self._print_summary(t_wall)

        # ── Keep viewer open until user closes the window ─────────────────────
        if _viewer_ctx is not None:
            viewer = _viewer_ctx["viewer"]
            while viewer._window.running:
                self._update_viewer(_viewer_ctx, trajs)

        return {
            "sim_time":  self._time,
            "wall_time": t_wall,
        }

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_physics(self, dt: float) -> None:
        """Initialise all registered solvers."""
        for rh in self._scene._robot_handles.values():
            if rh._controller is not None:
                rh._controller._robot = rh._model   # pass model for gravity comp
            rh._solver.initialize(dt=dt)

        # Rigid bodies
        for bh in self._scene._body_handles.values():
            if hasattr(bh, '_solver') and hasattr(bh._solver, 'initialize'):
                phys = self._scene._body_physics.get(bh.name)
                from robosim.scene.objects import FEM, CB
                if isinstance(phys, FEM):
                    fem_dt = dt * phys.dt_scale
                    bh._solver.initialize(dt=fem_dt)
                    bh._fem_dt    = fem_dt
                    bh._fem_every = phys.dt_scale
                    bh._arm_step_count = 0
                else:
                    bh._solver.initialize(dt=dt)
                    bh._fem_dt    = dt
                    bh._fem_every = 1
                    bh._arm_step_count = 0

        # Register arm with ContactSolver
        contact = self._scene._contact
        for rh in self._scene._robot_handles.values():
            contact.register_rbd(rh._solver, robot_id=rh.name)

        # Register rigid bodies with ContactSolver + cross-filters
        from robosim.scene.handles import RigidBodyHandle
        for bh in self._scene._body_handles.values():
            if isinstance(bh, RigidBodyHandle):
                # Free-floating RBD body — flag it so constraint-mode
                # contact can position-project it out of arm penetration.
                contact.register_rbd(bh._solver, robot_id=bh.name,
                                     is_free_body=True)
                # links[-1] is the terminal body link (carries collision geometry);
                # links[0] is the massless base anchor.
                body_link_name = bh._model.links[-1].name
                for rh in self._scene._robot_handles.values():
                    for arm_link in self._scene._arm_contact_links:
                        contact.add_cross_filter(
                            rh.name, arm_link, bh.name, body_link_name,
                        )

        # Build contact pair geometry from URDF collision meshes
        for cp in self._scene._contact_pairs:
            cp._build_geom()
            for li_name in cp.contact_links:
                idx = cp.robot._model.link_index(li_name)
                cp._cached_wrenches[idx] = np.zeros(6)

    # ── robot step ────────────────────────────────────────────────────────────

    def _apply_grip_locks(self) -> None:
        """Re-pin every active grip's trigger joint (and its mimic followers)
        to the q value captured when the grip snapped on.

        Called once per substep AFTER the trajectory advances ``target_q``
        and BEFORE PD reads it.  Without this, the trajectory's LIFT-phase
        target keeps driving the finger inward; since contact is bypassed
        once the body is kinematically locked, the finger then accelerates
        through the body (the "fork through tofu" failure mode).
        """
        scene = self._scene
        for body_name, q_lock in scene._grip_q_lock.items():
            if not scene._grip_active.get(body_name, False):
                continue
            state = scene._grip_state.get(body_name)
            if state is None:
                continue
            rh = state["robot"]
            ctrl = rh._controller
            if ctrl is None or getattr(ctrl, "target_q", None) is None:
                continue
            trigger_q = state["trigger_q_idx"]
            ctrl.target_q = ctrl.target_q.copy()
            ctrl.target_q[trigger_q] = q_lock
            model = rh._model
            mimic_map = getattr(model, "_mimic_map", None)
            if mimic_map:
                for f_idx, (l_idx, mult, off) in mimic_map.items():
                    if model._dof_index[l_idx] == trigger_q:
                        f_dof = model._dof_index[f_idx]
                        ctrl.target_q[f_dof] = mult * q_lock + off

    def _step_robot(self, rh: "RobotHandle", dt: float) -> None:
        rh._solver.clear_external_forces()

        # PD + gravity comp torque
        if rh._controller is not None:
            tau = rh._controller.compute(self._time, rh._model.q, rh._model.qd)
            rh._solver.tau = tau

        # Ground contact wrenches from ContactSolver
        contact  = self._scene._contact
        wrenches = contact.compute_rbd_contact_forces(rh._solver)
        for li, w in wrenches.items():
            rh._solver.set_external_force(li, w)

        # Cached FEM/CB contact reactions (1-step lag).
        # Computed by _step_body() in the PREVIOUS substep; applied here BEFORE step().
        for cp in self._scene._contact_pairs:
            if cp.robot is not rh:
                continue
            if self._scene._grip_active.get(cp.body.name, False):
                continue
            for li_idx, w in cp._cached_wrenches.items():
                if np.linalg.norm(w) < 1e-12:
                    continue
                if li_idx in rh._solver._f_ext:
                    rh._solver._f_ext[li_idx] = rh._solver._f_ext[li_idx] + w
                else:
                    rh._solver.set_external_force(li_idx, w)

        rh._solver.step()

        # Enforce mimic joints if any
        if hasattr(rh._model, 'enforce_mimic'):
            rh._model.enforce_mimic()

    # ── body step ─────────────────────────────────────────────────────────────

    def _step_body(self, bh, dt: float) -> None:
        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle, RigidBodyHandle
        from robosim.scene.objects import FEM, CB

        phys = self._scene._body_physics.get(bh.name)

        if isinstance(bh, RigidBodyHandle):
            # Rigid: ContactSolver handles everything
            contact  = self._scene._contact
            bh._solver.clear_external_forces()
            wrenches = contact.compute_rbd_contact_forces(bh._solver)
            for li, w in wrenches.items():
                bh._solver.set_external_force(li, w)
            bh._solver.step()

            # Ground-hold (suppress XY drift while box is on the ground).
            # links[-1] is the terminal link with collision geometry.
            if not self._scene._grip_active.get(bh.name, False):
                half_z = bh._model.links[-1].collisions[0].geometry.size[2] / 2.0
                if bh._model.q[2] < half_z * 1.2:   # CoM near ground level
                    bh._model.q[:2]  = self._scene._body_ground_xy[bh.name]
                    bh._model.qd[:2] = 0.0
                    bh._model.q[3:6] = 0.0
                    bh._model.qd[3:6] = 0.0

            # Kinematic grip
            self._update_grip_rigid(bh, dt)
            return

        # Deformable bodies: FEM or CB
        bh._arm_step_count += 1
        run_this_step = (bh._arm_step_count % bh._fem_every == 0)
        if not run_this_step:
            return

        grip_active = self._scene._grip_active.get(bh.name, False)

        if isinstance(bh, FEMBodyHandle):
            fem_dt = bh._fem_dt
            if not grip_active:
                # Find contact pair for this body
                cp = self._find_contact_pair(bh)
                if cp is not None:
                    # Position-projection AFTER step
                    bh._solver.step(dt=fem_dt)
                    fk = cp.robot._model.forward_kinematics()
                    self._apply_projection_contact(cp, fk, bh._body, fem_dt)
                else:
                    bh._solver.step(dt=fem_dt)
            else:
                bh._solver.step(dt=fem_dt)
                if cp := self._find_contact_pair(bh):
                    for idx in cp._cached_wrenches:
                        cp._cached_wrenches[idx][:] = 0.0

            self._apply_ground_projection(bh._body)
            self._update_grip_fem(bh, dt)

        elif isinstance(bh, CBBodyHandle):
            if not grip_active:
                cp = self._find_contact_pair(bh)
                if cp is not None:
                    # Penalty forces BEFORE step
                    fk = cp.robot._model.forward_kinematics()
                    f_ext, wrenches = self._compute_penalty_contact(cp, fk, bh._body)
                    bh._solver.step(dt=dt, extra_forces={0: f_ext})
                    # Cache wrenches
                    for li_name, w in wrenches.items():
                        idx = cp.robot._model.link_index(li_name)
                        cp._cached_wrenches[idx] = w
                else:
                    bh._solver.step(dt=dt)
            else:
                bh._solver.step(dt=dt)
                if cp := self._find_contact_pair(bh):
                    for idx in cp._cached_wrenches:
                        cp._cached_wrenches[idx][:] = 0.0

            self._apply_ground_projection(bh._body)
            self._update_grip_fem(bh, dt)

        # Ground-hold for deformable (XY drift suppression)
        if not grip_active:
            ground_xy = self._scene._body_ground_xy.get(bh.name)
            if ground_xy is not None:
                com_xy   = bh._body.x[:, :2].mean(axis=0)
                delta_xy = ground_xy - com_xy
                if np.linalg.norm(delta_xy) > 1e-6:
                    bh._body.x[:, :2] += delta_xy
                    bh._body.v[:, :2]  = 0.0

    # ── grip ──────────────────────────────────────────────────────────────────

    def _update_grip_fem(self, bh, dt: float) -> None:
        """Kinematic grip for FEM/CB bodies — body translates with palm."""
        state = self._scene._grip_state.get(bh.name)
        if state is None:
            return

        robot_handle = state["robot"]
        lift_start_t = state["lift_start_t"]
        trigger_q    = state["trigger_q_idx"]  # joint index to check
        trigger_val  = state["trigger_q_val"]  # threshold

        fk       = robot_handle._model.forward_kinematics()
        palm_idx = robot_handle._model.link_index("palm_link")
        palm_pos = fk[palm_idx].translation.copy()
        palm_rot = fk[palm_idx].rotation.copy()

        grip_active = self._scene._grip_active.get(bh.name, False)
        q_fing = robot_handle._model.q[trigger_q]

        if not grip_active and self._time >= lift_start_t and q_fing < trigger_val:
            self._scene._grip_active[bh.name]    = True
            self._scene._grip_palm_pos0[bh.name] = palm_pos.copy()
            self._scene._grip_palm_rot0[bh.name] = palm_rot.copy()
            self._scene._grip_body_x0[bh.name]   = bh._body.x.copy()
            # Freeze the finger PD target at the q reached when the grip
            # snapped on.  Without this, the LIFT-phase target keeps driving
            # the finger inward — but contact is now bypassed, so the finger
            # accelerates THROUGH the kinematically-locked body, producing the
            # "fork through tofu" appearance.
            self._scene._grip_q_lock[bh.name] = float(q_fing)
            grip_active = True

        if grip_active:
            palm_pos0 = self._scene._grip_palm_pos0[bh.name]
            palm_rot0 = self._scene._grip_palm_rot0[bh.name]
            body_x0   = self._scene._grip_body_x0[bh.name]

            # Rigid transform from grip-time palm pose to current palm pose:
            #   x_new = palm_pos + (palm_rot @ palm_rot0.T) @ (body_x0 - palm_pos0)
            # Body translates AND rotates with the palm so the fingers do not
            # skewer further into the body during a curved lift trajectory.
            R_delta = palm_rot @ palm_rot0.T
            offsets = body_x0 - palm_pos0           # (N, 3)
            bh._body.x = palm_pos + offsets @ R_delta.T

            # Velocity = palm linear velocity + omega × (x - palm_pos)
            # Use the cached spatial velocity built during the contact step.
            try:
                omega, v_origin = robot_handle._model.link_world_velocities()[palm_idx]
            except Exception:
                omega = np.zeros(3); v_origin = np.zeros(3)
            r = bh._body.x - palm_pos                # (N, 3)
            cross = np.column_stack([
                omega[1]*r[:, 2] - omega[2]*r[:, 1],
                omega[2]*r[:, 0] - omega[0]*r[:, 2],
                omega[0]*r[:, 1] - omega[1]*r[:, 0],
            ])
            bh._body.v = v_origin + cross

        self._scene._grip_prev_palm[bh.name] = palm_pos.copy()

    def _update_grip_rigid(self, bh: "RigidBodyHandle", dt: float) -> None:
        """Kinematic grip for rigid bodies."""
        state = self._scene._grip_state.get(bh.name)
        if state is None:
            return

        robot_handle = state["robot"]
        lift_start_t = state["lift_start_t"]
        trigger_q    = state["trigger_q_idx"]
        trigger_val  = state["trigger_q_val"]

        fk       = robot_handle._model.forward_kinematics()
        palm_idx = robot_handle._model.link_index("palm_link")
        palm_pos = fk[palm_idx].translation.copy()

        grip_active = self._scene._grip_active.get(bh.name, False)

        if (not grip_active and self._time >= lift_start_t
                and robot_handle._model.q[trigger_q] < trigger_val):
            self._scene._grip_active[bh.name]    = True
            self._scene._grip_palm_pos0[bh.name] = palm_pos.copy()
            self._scene._grip_box_pos0[bh.name]  = bh._model.q[:3].copy()
            self._scene._grip_offset[bh.name]    = (bh._model.q[:3] - palm_pos).copy()
            self._scene._grip_q_lock[bh.name]    = float(robot_handle._model.q[trigger_q])
            grip_active = True

        if grip_active:
            grip_off  = self._scene._grip_offset[bh.name]
            target    = palm_pos + grip_off
            box_pos0  = self._scene._grip_box_pos0[bh.name]
            if target[2] > box_pos0[2]:
                prev = self._scene._grip_prev_palm.get(bh.name)
                palm_vel = (palm_pos - prev) / dt if prev is not None else np.zeros(3)
                bh._model.q[:3]   = target
                bh._model.qd[:3]  = palm_vel
                bh._model.q[3:6]  = 0.0
                bh._model.qd[3:6] = 0.0

        self._scene._grip_prev_palm[bh.name] = palm_pos.copy()

    # ── FEM position-projection contact ───────────────────────────────────────

    def _apply_projection_contact(
        self,
        cp: "_ContactPair",
        fk: list,
        body,
        dt: float,
    ) -> None:
        """Post-step position projection: push penetrating nodes to finger surface.

        Operates in the finger's LINK frame so the projection direction (the
        finger's inner-face normal) follows the gripper as the arm rotates.
        """
        SLAB_PAD = 0.012   # tangent-axis padding so nodes between mesh rows
                           # can still register contact (see CB contact for context).
        M_diag  = body._M.diagonal()
        m_nodes = M_diag[0::3]

        for link_name in cp.contact_links:
            li_idx = cp.robot._model.link_index(link_name)
            geom_off, half_ext = cp._geom[link_name]

            T_link     = fk[li_idx]
            T_link_inv = T_link.inverse()
            R_link     = T_link.rotation
            hx, hy, hz = half_ext

            is_left = "left" in link_name.lower()
            push_sign = -1.0 if is_left else +1.0
            push_dir  = R_link[:, 1] * push_sign

            p_center   = T_link.translation + R_link @ geom_off
            inner_face = p_center + push_dir * hy

            # Vectorised slab + penetration test (matches CB contact).
            rel = (body.x - T_link.translation) @ R_link - geom_off
            slab = (np.abs(rel[:, 0]) <= hx + SLAB_PAD) & \
                   (np.abs(rel[:, 2]) <= hz + SLAB_PAD)
            pen = (inner_face - body.x) @ push_dir
            active = slab & (pen > 0.0) & (pen < 2.0 * hy)
            if not active.any():
                cp._cached_wrenches[li_idx] = np.zeros(6)
                continue
            idx = np.where(active)[0]
            pen_a = pen[idx]
            mi_a = m_nodes[idx]

            # Project nodes onto inner face (push along +push_dir by pen).
            body.x[idx] = body.x[idx] + pen_a[:, None] * push_dir[None, :]
            # Kill the inward component of velocity at projected nodes.
            v_old = body.v[idx].copy()
            v_into = -(v_old @ push_dir)
            kill = np.maximum(v_into, 0.0)
            body.v[idx] = v_old + kill[:, None] * push_dir[None, :]
            dv = body.v[idx] - v_old

            f_world = -mi_a[:, None] * dv / dt          # (k, 3)
            f_link  = f_world @ R_link
            p_link  = (body.x[idx] - T_link.translation) @ R_link
            tau_link = np.cross(p_link, f_link)
            w = np.zeros(6)
            w[:3] = tau_link.sum(axis=0)
            w[3:] = f_link.sum(axis=0)
            cp._cached_wrenches[li_idx] = w

    # ── CB penalty contact ────────────────────────────────────────────────────

    def _compute_penalty_contact(
        self,
        cp: "_ContactPair",
        fk: list,
        body,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Pre-step penalty forces for CB mode.

        Operates entirely in the finger's LINK frame so the finger geometry
        (slab + inner-face direction) is interpreted correctly regardless of
        how the arm is rotated.  An axial slab inflation (`SLAB_PAD`) lets
        nodes between mesh rows still register a contact, which prevents the
        "two-pinhole" visual where most box-face nodes ignore the finger.
        """
        # How much extra reach to allow along the finger's tangent axes when
        # deciding whether a body node is "under" the finger.  Without this
        # padding, mesh nodes that fall just outside the finger half-extents
        # in X / Z register no contact — the finger appears to skewer the
        # body between mesh rows.  Half a typical mesh spacing is enough.
        SLAB_PAD = 0.012

        n     = body.x.shape[0]
        f_ext = np.zeros(n * 3)
        w_map: dict[str, np.ndarray] = {}

        for link_name in cp.contact_links:
            li_idx = cp.robot._model.link_index(link_name)
            geom_off, half_ext = cp._geom[link_name]

            T_link     = fk[li_idx]
            T_link_inv = T_link.inverse()
            R_link     = T_link.rotation
            hx, hy, hz = half_ext

            # ``push_dir``: world-frame direction the finger pushes the body
            # (away from the finger's interior, toward the body).
            is_left = "left" in link_name.lower()
            push_sign = -1.0 if is_left else +1.0
            push_dir  = R_link[:, 1] * push_sign

            p_center   = T_link.translation + R_link @ geom_off
            inner_face = p_center + push_dir * hy

            # Vectorised slab + penetration test.
            # rel = R^T @ (x - link_origin) - geom_off
            rel = (body.x - T_link.translation) @ R_link - geom_off  # (N,3)
            slab = (np.abs(rel[:, 0]) <= hx + SLAB_PAD) & \
                   (np.abs(rel[:, 2]) <= hz + SLAB_PAD)
            if not slab.any():
                w_map[link_name] = np.zeros(6)
                continue

            # Penetration: dot(inner_face - x, push_dir).
            pen = (inner_face - body.x) @ push_dir   # (N,)
            active = slab & (pen > 0.0) & (pen < 2.0 * hy)
            if not active.any():
                w_map[link_name] = np.zeros(6)
                continue

            idx = np.where(active)[0]
            pen_a = pen[idx]
            v_into_a = -(body.v[idx] @ push_dir)
            fn_a = cp.k_contact * pen_a + cp.c_contact * np.maximum(v_into_a, 0.0)
            keep = fn_a > 0.0
            if not keep.any():
                w_map[link_name] = np.zeros(6)
                continue
            idx = idx[keep]
            fn_a = fn_a[keep]

            f_world = fn_a[:, None] * push_dir[None, :]   # (k, 3)
            # Scatter-add into f_ext (flattened (N*3,)).
            f_view = f_ext.reshape(-1, 3)
            np.add.at(f_view, idx, f_world)

            # Newton-3 wrench on the finger link, in LINK frame.  Aggregate.
            # ``p_link`` is the contact point position in the link frame
            # (relative to the link origin) so the moment arm is correct.
            f_link   = -f_world @ R_link        # link = R^T @ world (rows)
            p_link   = (body.x[idx] - T_link.translation) @ R_link
            tau_link = np.cross(p_link, f_link)
            w = np.zeros(6)
            w[:3] = tau_link.sum(axis=0)
            w[3:] = f_link.sum(axis=0)
            w_map[link_name] = w

        return f_ext, w_map

    # ── ground projection ─────────────────────────────────────────────────────

    @staticmethod
    def _apply_ground_projection(body, ground_z: float = 0.0) -> None:
        below = body.x[:, 2] < ground_z
        if not below.any():
            return
        body.x[below, 2] = ground_z
        body.v[below, 2] = np.maximum(body.v[below, 2], 0.0)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _find_contact_pair(self, bh) -> "_ContactPair | None":
        for cp in self._scene._contact_pairs:
            if cp.body is bh:
                return cp
        return None

    # ── viewer ────────────────────────────────────────────────────────────────

    def _init_viewer(self):
        """Initialise Taichi viewer and renderers. Returns context dict."""
        import taichi as ti
        # ti.init() must be called before any Taichi field/window creation.
        try:
            ti.init(arch=ti.metal)   # macOS Metal (M-series / AMD)
        except Exception:
            try:
                ti.init(arch=ti.vulkan)
            except Exception:
                ti.init()            # CPU fallback
        from robosim.viz.scene_renderer import RobotRenderer
        from robosim.viz.viewer import SimViewer

        scene = self._scene
        title = "RoboSim"
        viewer = SimViewer(title=title, window_size=(1280, 800))
        viewer.initialize()

        ctx = {"viewer": viewer, "renderers": {}, "meshes": {}}

        for rh in scene._robot_handles.values():
            rr = RobotRenderer(rh._model, viewer)
            rr.setup()
            ctx["renderers"][rh.name] = rr

        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle, RigidBodyHandle
        for bh in scene._body_handles.values():
            if isinstance(bh, (FEMBodyHandle, CBBodyHandle)):
                surf = bh._body.mesh.extract_surface()
                color = scene._body_colors.get(bh.name, np.array([0.2, 0.6, 0.9]))
                viewer.add_mesh(bh.name, bh._body.x, surf, color=color, opacity=1.0)
                ctx["meshes"][bh.name] = surf
            elif isinstance(bh, RigidBodyHandle):
                rr = RobotRenderer(bh._model, viewer)
                rr.setup()
                ctx["renderers"][bh.name] = rr

        return ctx

    def _update_viewer(self, ctx: dict, trajs: list) -> bool:
        """Render one viewer frame.  Returns False when the window is closed."""
        viewer = ctx["viewer"]
        scene  = self._scene

        # Check if window was closed by the user
        if not viewer._window.running:
            return False

        for name, rr in ctx["renderers"].items():
            rr.update()

        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle
        for name, surf in ctx["meshes"].items():
            bh = scene._body_handles[name]
            viewer.update_mesh_vertices(name, bh._body.x)

            # Von Mises stress coloring
            try:
                from robosim.physics.fem.assembly import batch_von_mises
                vm = batch_von_mises(
                    bh._body.mesh, bh._body.x, bh._body.material,
                    bh._body._dN_list, bh._body._volumes,
                )
                vm_max = max(float(vm.max()), 1.0)
                t = np.clip(vm / vm_max, 0, 1)
                r = np.clip(1.5 - np.abs(t - 0.75) * 4, 0, 1)
                g = np.clip(1.5 - np.abs(t - 0.50) * 4, 0, 1)
                b = np.clip(1.5 - np.abs(t - 0.25) * 4, 0, 1)
                colors = np.column_stack([r, g, b]).astype(np.float32)
                viewer.update_mesh_color(name, colors)
            except Exception:
                pass

        # HUD text
        phase_strs = [f"{traj.current_phase or 'DONE'}" for traj in trajs]
        phase_str  = " | ".join(phase_strs) if phase_strs else "–"
        lines = [
            f"t = {self._time:.3f} s   phase: {phase_str}",
            self._fps.format(),
        ]
        for bh in scene._body_handles.values():
            c = bh.com
            lines.append(
                f"{bh.name}: CoM=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})"
                f"  bot_z={bh.bottom_z:+.4f}"
            )
        viewer.add_text("\n".join(lines) + "\n[LDrag=orbit  Scroll=zoom  ESC=quit]")

        # ── Actually render the frame ──────────────────────────────────────────
        viewer._render_frame()
        viewer._window.show()
        return True

    # ── headless output ───────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        scene = self._scene
        mode_strs = []
        from robosim.scene.objects import FEM, CB, Rigid
        for bh in scene._body_handles.values():
            phys = scene._body_physics.get(bh.name)
            mode_strs.append(f"{bh.name}={type(phys).__name__.lower()}")
        print("=" * 60)
        print(f"  RoboSim  dt={self._dt}s  " + "  ".join(mode_strs) + "  (headless)")
        print("=" * 60)

    def _print_summary(self, wall_time: float) -> None:
        print(f"\n{'='*60}")
        print(f"Simulated  : {self._time:.2f} s")
        print(f"Wall time  : {wall_time:.2f} s  ({self._time/max(wall_time,1e-9):.2f}× real-time)")
        print(f"Frames     : {self._fps.n_frames}  "
              f"(avg {self._fps.average:.1f} FPS, "
              f"last-window {self._fps.current:.1f} FPS)")
        for bh in self._scene._body_handles.values():
            c    = bh.com
            z_lo = bh.bottom_z
            lift = max(0.0, z_lo)
            print(f"{bh.name}  CoM=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
                  f"  bot_z={z_lo:.4f}  lifted={lift:.4f} m")
            ok = "SUCCESS" if lift > 0.05 else "FAIL — not lifted"
            print(f"  Lift result: {ok}")
        print("=" * 60)
