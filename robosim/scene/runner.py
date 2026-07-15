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
        # Cached rigid-body reaction from deformable-gripper contact (1-step lag).
        self._gripper_box_wrenches: dict[str, np.ndarray] = {}
        # Cached nodal forces on deformable grasp targets (1-step lag).
        self._gripper_box_node_forces: dict[str, np.ndarray] = {}
        # Per-finger normal force from last deformable-gripper contact substep [N].
        self._finger_fn_cache: dict[str, float] = {}
        self._finger_contact_s_cache: dict[str, float] = {}

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
        cap = getattr(self._scene, "_finger_capture", None)
        use_viewer = (viewer and not headless) or (cap is not None)
        if use_viewer:
            _viewer_ctx = self._init_viewer()
            if cap is not None:
                cap.lock_viewer_camera(_viewer_ctx["viewer"])

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

                # ── step deformable grippers (kinematically anchored to robot links) ──
                self._step_deformable_grippers(dt)

                # NOTE: ConstraintContactSolver also exposes
                # project_free_body_penetrations() for non-grasp scenarios
                # (e.g. a free box bouncing off an arm link), but it is
                # NOT called from the runner: for a symmetric two-finger
                # grasp the per-finger pushes cancel and the projection
                # cannot reduce the residual pen without a proper coupled
                # LCP/PGS solver.

                self._time += dt

            # ── viewer update (before on_step so CV matches logged frame) ──
            if _viewer_ctx is not None:
                alive = self._update_viewer(_viewer_ctx, trajs)
                if not alive:
                    break   # user closed the window

            # ── frame callbacks ──
            if on_step is not None:
                on_step(self._scene, self._time)

            # ── frame timing (after the frame's heavy work) ──
            self._fps.tick()

        t_wall = time.time() - t_wall0

        if headless:
            self._print_summary(t_wall)

        # ── Keep viewer open until user closes (skip when batch-capturing) ───
        if _viewer_ctx is not None and cap is None:
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
                from robosim.scene.objects import FEM, FEMPlastic, HybridPlastic, CB
                # Substep-gated bodies (FEM-side): scale fem_dt by
                # phys.dt_scale and run every dt_scale arm steps.
                if isinstance(phys, (FEM, FEMPlastic, HybridPlastic)):
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

        # Deformable gripper CB bodies — initialise the implicit-Euler
        # system, then lift mesh.nodes (link-frame reference) into world
        # via the current robot FK so the first viewer frame and the
        # first physics step both see the body at its anchored pose.
        for grip in self._scene._deformable_grippers:
            rh = grip["robot"]
            fk = rh._model.forward_kinematics()
            for spec in grip["specs"]:
                body     = spec["body"]
                link_idx = spec["link_idx"]
                T_link   = fk[link_idx]
                body.initialize(dt=dt)
                body.set_anchor_pose(T_link.rotation, T_link.translation)
                body.x = body.mesh.nodes @ T_link.rotation.T + T_link.translation
                body.v = np.zeros_like(body.v)

        # Deformable grippers replace rigid finger ↔ object contact.
        contact = self._scene._contact
        for grip in self._scene._deformable_grippers:
            target = grip.get("target_body")
            if target is None:
                continue
            from robosim.scene.handles import RigidBodyHandle
            if not isinstance(target, RigidBodyHandle):
                continue
            body_link = target._model.links[-1].name
            for spec in grip["specs"]:
                contact.add_cross_filter(
                    grip["robot"].name, spec["link_name"],
                    target.name, body_link,
                )

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
        # iterate over a snapshot — we may clear entries below.
        for body_name, q_lock in list(scene._grip_q_lock.items()):
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

            # Lock-as-floor semantics: never let the PD target go *below*
            # the snapshot taken at grip activation. Two effects:
            #   (a) During LIFT, the trajectory's negative finger target
            #       can't drive the fingers further into the body — no
            #       "fork through tofu" creep.
            #   (b) During RELEASE, the trajectory's positive target
            #       (opening the fingers) is *not* clamped — it overrides
            #       the lock, the floor self-clears, and the fingers
            #       actually open.
            traj_target = float(ctrl.target_q[trigger_q])
            if traj_target >= q_lock:
                # Trajectory already commands a more-open finger ⇒ the
                # floor is no longer doing anything. Drop it so the
                # mimic-side override below also doesn't intercept.
                scene._grip_q_lock.pop(body_name, None)
                continue

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
        from robosim.scene.handles import (
            FEMBodyHandle, CBBodyHandle, RigidBodyHandle,
            HybridPlasticBodyHandle,
        )
        from robosim.scene.objects import FEM, CB

        phys = self._scene._body_physics.get(bh.name)

        if isinstance(bh, RigidBodyHandle):
            # Rigid: ContactSolver handles everything
            contact  = self._scene._contact
            bh._solver.clear_external_forces()
            wrenches = contact.compute_rbd_contact_forces(bh._solver)
            for li, w in wrenches.items():
                bh._solver.set_external_force(li, w)
            # Deformable-gripper penalty reaction (1-step lag).
            gw = self._gripper_box_wrenches.get(bh.name)
            if gw is not None and np.linalg.norm(gw) > 1e-12:
                box_li = bh._model.link_index(bh._model.links[-1].name)
                if box_li in bh._solver._f_ext:
                    bh._solver._f_ext[box_li] = bh._solver._f_ext[box_li] + gw
                else:
                    bh._solver.set_external_force(box_li, gw)
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
            grip_nf = self._gripper_box_node_forces.get(bh.name)
            extra = {0: grip_nf} if grip_nf is not None and np.linalg.norm(grip_nf) > 1e-12 else None
            if not grip_active:
                # Find contact pair for this body
                cp = self._find_contact_pair(bh)
                if cp is not None:
                    # Position-projection AFTER step
                    bh._solver.step(dt=fem_dt, extra_forces=extra)
                    extra = None
                    fk = cp.robot._model.forward_kinematics()
                    self._apply_projection_contact(cp, fk, bh._body, fem_dt)
                else:
                    if extra is not None:
                        bh._solver.step(dt=fem_dt, extra_forces=extra)
                    else:
                        bh._solver.step(dt=fem_dt)
            else:
                if extra is not None:
                    bh._solver.step(dt=fem_dt, extra_forces=extra)
                else:
                    bh._solver.step(dt=fem_dt)
                if cp := self._find_contact_pair(bh):
                    for idx in cp._cached_wrenches:
                        cp._cached_wrenches[idx][:] = 0.0

            self._apply_ground_projection(bh._body)
            self._update_grip_fem(bh, dt)

        elif isinstance(bh, CBBodyHandle):
            grip_nf = self._gripper_box_node_forces.get(bh.name)
            extra = {0: grip_nf} if grip_nf is not None and np.linalg.norm(grip_nf) > 1e-12 else None
            if not grip_active:
                cp = self._find_contact_pair(bh)
                if cp is not None:
                    # Penalty forces BEFORE step (drives nodes outward early)
                    fk = cp.robot._model.forward_kinematics()
                    f_ext, _ = self._compute_penalty_contact(cp, fk, bh._body)
                    if extra is not None:
                        f_ext = f_ext + extra
                        extra = None
                    bh._solver.step(dt=dt, extra_forces={0: f_ext})
                    # Position projection AFTER step — hard non-penetration
                    # guarantee.  Penalty alone lets the CB modal deformation
                    # absorb the force and the finger sinks in like tofu.
                    fk = cp.robot._model.forward_kinematics()
                    self._apply_projection_contact(cp, fk, bh._body, dt)
                else:
                    if extra is not None:
                        bh._solver.step(dt=dt, extra_forces=extra)
                    else:
                        bh._solver.step(dt=dt)
            else:
                if extra is not None:
                    bh._solver.step(dt=dt, extra_forces=extra)
                else:
                    bh._solver.step(dt=dt)
                if cp := self._find_contact_pair(bh):
                    for idx in cp._cached_wrenches:
                        cp._cached_wrenches[idx][:] = 0.0

            self._apply_ground_projection(bh._body)
            self._update_grip_fem(bh, dt)

        elif isinstance(bh, HybridPlasticBodyHandle):
            # Same dispatch shape as CB — penalty contact + extra_forces
            # to bh._solver.step. The hybrid body internally routes
            # ELASTIC regions through CB and ACTIVE regions through
            # full-FEM J2 each substep. Use ``fem_dt`` (= arm_dt ·
            # phys.dt_scale) so the dt the body was initialised with
            # stays cached and ``initialize`` doesn't refire each tick.
            fem_dt = bh._fem_dt
            if not grip_active:
                cp = self._find_contact_pair(bh)
                if cp is not None:
                    fk = cp.robot._model.forward_kinematics()
                    f_ext, wrenches = self._compute_penalty_contact(cp, fk, bh._body)
                    bh._solver.step(dt=fem_dt, extra_forces={0: f_ext})
                    for li_name, w in wrenches.items():
                        idx = cp.robot._model.link_index(li_name)
                        cp._cached_wrenches[idx] = w
                else:
                    bh._solver.step(dt=fem_dt)
            else:
                bh._solver.step(dt=fem_dt)
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

    # ── deformable grippers ───────────────────────────────────────────────────

    def _step_deformable_grippers(self, dt: float) -> None:
        """Per-step update of attached deformable gripper CB bodies.

        For each (link, CB body) registered via
        :meth:`Scene.attach_deformable_gripper`: read the current link
        FK, set the body's anchor pose via
        :meth:`CraigBamptonBody.set_anchor_pose`, and advance one
        implicit-Euler step. When :meth:`Scene.contact_deformable_gripper`
        has wired a grasp target, penalty contact on penetrating finger
        nodes drives cantilever-style bending and caches a reaction
        wrench / nodal forces on the target for the next substep.
        """
        self._gripper_box_wrenches.clear()
        self._gripper_box_node_forces.clear()
        self._finger_fn_cache.clear()
        self._finger_contact_s_cache.clear()
        for grip in self._scene._deformable_grippers:
            rh = grip["robot"]
            fk = rh._model.forward_kinematics()
            target = grip.get("target_body")
            k_pen = grip.get("k_contact", 8e3)
            c_pen = grip.get("c_contact", 40.0)
            box_wrench = np.zeros(6)

            for spec in grip["specs"]:
                body     = spec["body"]
                link_idx = spec["link_idx"]
                T_link   = fk[link_idx]
                R_new    = T_link.rotation
                t_new    = T_link.translation

                R_old = (body._anchor_R if body._anchor_R is not None
                         else np.eye(3))
                t_old = (body._anchor_t if body._anchor_t is not None
                         else np.zeros(3))
                relR  = R_new @ R_old.T
                body.x = (body.x - t_old) @ relR.T + t_new
                body.v = body.v @ relR.T

                extra_forces = None
                f_ext_cached = None
                if target is not None:
                    from robosim.scene.handles import RigidBodyHandle, CBBodyHandle, FEMBodyHandle
                    if isinstance(target, RigidBodyHandle):
                        f_ext, w_box = self._compute_gripper_rigid_contact(
                            body, target, spec, fk, k_pen, c_pen,
                        )
                        if np.linalg.norm(f_ext) > 1e-12:
                            extra_forces = {0: f_ext}
                            f_ext_cached = f_ext
                        box_wrench += w_box
                    elif isinstance(target, (CBBodyHandle, FEMBodyHandle)):
                        f_ext, f_box = self._compute_gripper_deformable_contact(
                            body, target._body, k_pen, c_pen,
                        )
                        if np.linalg.norm(f_ext) > 1e-12:
                            extra_forces = {0: f_ext}
                            f_ext_cached = f_ext
                        if np.linalg.norm(f_box) > 1e-12:
                            prev = self._gripper_box_node_forces.get(target.name)
                            if prev is None:
                                self._gripper_box_node_forces[target.name] = f_box.copy()
                            else:
                                self._gripper_box_node_forces[target.name] = prev + f_box

                body.set_anchor_pose(R_new, t_new)
                body.step(dt=dt, extra_forces=extra_forces)

                # Elastic grasp load: U from CB field, δ at contact station.
                # Pass held s_c so brief zero-penalty frames keep F consistent.
                from robosim.scene.finger_probe import elastic_normal_force_n
                link_name = spec["link_name"]
                fn_n, s_c = elastic_normal_force_n(
                    body, R_new, t_new, link_name,
                    f_ext=f_ext_cached,
                    contact_s=self._finger_contact_s_cache.get(link_name),
                )
                self._finger_fn_cache[link_name] = fn_n
                if s_c is not None and fn_n >= 1e-6:
                    self._finger_contact_s_cache[link_name] = float(s_c)
                elif fn_n < 1e-6:
                    self._finger_contact_s_cache.pop(link_name, None)

                if target is not None:
                    from robosim.scene.handles import RigidBodyHandle
                    if isinstance(target, RigidBodyHandle):
                        self._project_gripper_rigid_contact(body, target, spec)
                    else:
                        self._project_gripper_deformable_contact(
                            body, target._body, spec,
                        )

            from robosim.scene.handles import RigidBodyHandle
            if isinstance(target, RigidBodyHandle):
                self._gripper_box_wrenches[target.name] = box_wrench

    def sample_finger_probe(self) -> dict[str, dict[str, float]]:
        """Per-finger deflection and normal force for logging / CV validation.

        Returns a dict keyed by link name (``left_finger``, ``right_finger``)
        with ``deflection_m`` / ``deflection_y_m`` (signed closing bend, m)
        and ``fn_n`` (Newtons).
        """
        from robosim.scene.finger_probe import (
            empty_probe,
            tip_deflection_y_m,
            deflection_profile_closing_mm,
            normal_force_from_wrench,
        )

        out = empty_probe()

        for grip in self._scene._deformable_grippers:
            rh = grip["robot"]
            fk = rh._model.forward_kinematics()
            for spec in grip["specs"]:
                link_name = spec["link_name"]
                body = spec["body"]
                T = fk[spec["link_idx"]]
                R, t = T.rotation, T.translation
                # Signed closing-direction bend (same convention as CV).
                dy = tip_deflection_y_m(body, R, t, link_name)
                out[link_name]["deflection_m"] = dy
                out[link_name]["deflection_y_m"] = dy
                out[link_name]["fn_n"] = self._finger_fn_cache.get(link_name, 0.0)
                out[link_name]["contact_s"] = self._finger_contact_s_cache.get(
                    link_name, float("nan"),
                )
                x_m, u_mm = deflection_profile_closing_mm(body, R, t, link_name)
                out[link_name]["profile_x_m"] = x_m
                out[link_name]["profile_u_mm"] = u_mm

        for cp in self._scene._contact_pairs:
            if self._scene._grip_active.get(cp.body.name, False):
                continue
            fk = cp.robot._model.forward_kinematics()
            for link_name in cp.contact_links:
                if out[link_name]["fn_n"] > 1e-9:
                    continue
                li_idx = cp.robot._model.link_index(link_name)
                w = cp._cached_wrenches.get(li_idx, np.zeros(6))
                out[link_name]["fn_n"] = normal_force_from_wrench(w, link_name)

        return out

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
        release_val  = state.get("release_val")

        fk       = robot_handle._model.forward_kinematics()
        palm_idx = robot_handle._model.link_index("palm_link")
        palm_pos = fk[palm_idx].translation.copy()
        palm_rot = fk[palm_idx].rotation.copy()

        grip_active = self._scene._grip_active.get(bh.name, False)
        q_fing = robot_handle._model.q[trigger_q]

        # Release path: fingers commanded back open ⇒ drop the kinematic
        # lock and let the body inherit the palm's current velocity. The
        # last per-step v assignment in the still-active branch below
        # already mirrors palm linear+angular velocity, so v carries
        # through correctly into the freely-falling next step.
        if grip_active and release_val is not None and q_fing > release_val:
            self._scene._grip_active[bh.name] = False
            self._scene._grip_q_lock.pop(bh.name, None)
            grip_active = False

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
        release_val  = state.get("release_val")

        fk       = robot_handle._model.forward_kinematics()
        palm_idx = robot_handle._model.link_index("palm_link")
        palm_pos = fk[palm_idx].translation.copy()

        grip_active = self._scene._grip_active.get(bh.name, False)
        q_fing = robot_handle._model.q[trigger_q]

        # Release path: fingers re-open past release_val ⇒ release lock.
        # The last per-step qd assignment (palm_vel) sticks on the box,
        # so the box starts falling with the palm's linear velocity at
        # the moment of release.
        if grip_active and release_val is not None and q_fing > release_val:
            self._scene._grip_active[bh.name] = False
            self._scene._grip_q_lock.pop(bh.name, None)
            grip_active = False

        if (not grip_active and self._time >= lift_start_t
                and q_fing < trigger_val):
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

    # ── deformable gripper ↔ rigid body contact ───────────────────────────────

    def _compute_gripper_rigid_contact(
        self,
        finger_body,
        rigid_bh: "RigidBodyHandle",
        spec: dict,
        fk: list,
        k: float,
        c: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Penalty contact: deformable finger tip nodes vs a rigid box OBB.

        Only free (non-anchor) finger nodes participate so contact can sit
        anywhere along the prong as grasp depth changes. Used to *drive*
        the CB mesh; logged grasp load uses strain-energy / contact-s.
        """
        MAX_PEN = 0.002
        MAX_FN  = 0.12

        from robosim.scene.finger_probe import free_contact_node_indices

        model = rigid_bh._model
        box_link = model.links[-1]
        fk_box = model.forward_kinematics()
        T_box = fk_box[model.link_index(box_link.name)]
        R_box = T_box.rotation
        p_box = T_box.translation

        col = box_link.collisions[0]
        half_ext = col.geometry.size / 2.0
        col_off = col.origin.translation
        center = p_box + R_box @ col_off

        contact_nodes = free_contact_node_indices(finger_body)

        n = finger_body.x.shape[0]
        f_ext = np.zeros(n * 3)

        rel = (finger_body.x[contact_nodes] - center) @ R_box
        pen_axes = half_ext - np.abs(rel)
        inside_local = (pen_axes[:, 0] > 0) & (pen_axes[:, 1] > 0) & (pen_axes[:, 2] > 0)
        if not inside_local.any():
            return f_ext, np.zeros(6)

        idx_all = contact_nodes[inside_local]
        pen_a = pen_axes[inside_local]
        rel_a = rel[inside_local]
        axis = np.argmin(pen_a, axis=1)
        depth = np.clip(pen_a[np.arange(len(idx_all)), axis], 0.0, MAX_PEN)
        sign = np.sign(rel_a[np.arange(len(idx_all)), axis])
        sign[sign == 0] = 1.0
        n_local = np.zeros((len(idx_all), 3))
        n_local[np.arange(len(idx_all)), axis] = sign
        n_world = n_local @ R_box.T

        v_out = np.sum(finger_body.v[idx_all] * n_world, axis=1)
        fn = np.clip(k * depth + c * np.maximum(-v_out, 0.0), 0.0, MAX_FN)
        keep = fn > 0.0
        if not keep.any():
            return f_ext, np.zeros(6)
        idx = idx_all[keep]
        fn = fn[keep]
        n_world = n_world[keep]

        f_on_finger = fn[:, None] * n_world
        f_view = f_ext.reshape(-1, 3)
        np.add.at(f_view, idx, f_on_finger)

        f_on_box = -f_on_finger.sum(axis=0)
        r_arm = finger_body.x[idx] - p_box
        tau_on_box = np.cross(r_arm, -f_on_finger).sum(axis=0)

        w_box = np.zeros(6)
        w_box[:3] = R_box.T @ tau_on_box
        w_box[3:] = R_box.T @ f_on_box
        return f_ext, w_box

    def _project_gripper_rigid_contact(
        self,
        finger_body,
        rigid_bh: "RigidBodyHandle",
        spec: dict,
    ) -> None:
        """Post-step: push penetrating free finger nodes to the box surface."""
        from robosim.scene.finger_probe import free_contact_node_indices

        model = rigid_bh._model
        box_link = model.links[-1]
        fk_box = model.forward_kinematics()
        T_box = fk_box[model.link_index(box_link.name)]
        R_box = T_box.rotation
        p_box = T_box.translation
        col = box_link.collisions[0]
        half_ext = col.geometry.size / 2.0
        col_off = col.origin.translation
        center = p_box + R_box @ col_off

        contact_nodes = free_contact_node_indices(finger_body)

        rel = (finger_body.x[contact_nodes] - center) @ R_box
        pen_axes = half_ext - np.abs(rel)
        inside = (pen_axes[:, 0] > 0) & (pen_axes[:, 1] > 0) & (pen_axes[:, 2] > 0)
        if not inside.any():
            return

        idx = contact_nodes[inside]
        rel_a = rel[inside]
        pen_a = pen_axes[inside]
        axis = np.argmin(pen_a, axis=1)
        depth = pen_a[np.arange(len(idx)), axis]
        sign = np.sign(rel_a[np.arange(len(idx)), axis])
        sign[sign == 0] = 1.0
        n_local = np.zeros((len(idx), 3))
        n_local[np.arange(len(idx)), axis] = sign
        n_world = n_local @ R_box.T

        finger_body.x[idx] += depth[:, None] * n_world
        v_into = -(finger_body.v[idx] * n_world).sum(axis=1)
        finger_body.v[idx] += np.maximum(v_into, 0.0)[:, None] * n_world

    # ── deformable gripper ↔ deformable body contact ─────────────────────────

    @staticmethod
    def _box_aabb(box_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World-frame centre and half-extents from current node positions."""
        lo = box_x.min(axis=0)
        hi = box_x.max(axis=0)
        return 0.5 * (lo + hi), 0.5 * (hi - lo)

    @staticmethod
    def _distribute_reaction_to_box(
        box_x: np.ndarray,
        contact_pos: np.ndarray,
        contact_forces: np.ndarray,
    ) -> np.ndarray:
        """Newton-3: spread finger contact forces onto nearest box nodes."""
        n_box = box_x.shape[0]
        f_box = np.zeros(n_box * 3)
        if len(contact_pos) == 0:
            return f_box
        f_view = f_box.reshape(-1, 3)
        # Nearest-node scatter (one target per contact point).
        d2 = ((box_x[:, None, :] - contact_pos[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argmin(d2, axis=0)
        np.add.at(f_view, nearest, -contact_forces)
        return f_box

    def _compute_gripper_deformable_contact(
        self,
        finger_body,
        target_body,
        k: float,
        c: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Penalty contact: deformable finger free nodes vs deformable box AABB."""
        MAX_PEN = 0.002
        MAX_FN  = 0.12

        from robosim.scene.finger_probe import free_contact_node_indices

        center, half_ext = self._box_aabb(target_body.x)
        contact_nodes = free_contact_node_indices(finger_body)

        n = finger_body.x.shape[0]
        f_ext = np.zeros(n * 3)

        rel = finger_body.x[contact_nodes] - center
        pen_axes = half_ext - np.abs(rel)
        inside = (pen_axes[:, 0] > 0) & (pen_axes[:, 1] > 0) & (pen_axes[:, 2] > 0)
        if not inside.any():
            return f_ext, np.zeros(target_body.x.shape[0] * 3)

        idx_all = contact_nodes[inside]
        pen_a = pen_axes[inside]
        rel_a = rel[inside]
        axis = np.argmin(pen_a, axis=1)
        depth = np.clip(pen_a[np.arange(len(idx_all)), axis], 0.0, MAX_PEN)
        sign = np.sign(rel_a[np.arange(len(idx_all)), axis])
        sign[sign == 0] = 1.0
        n_local = np.zeros((len(idx_all), 3))
        n_local[np.arange(len(idx_all)), axis] = sign
        n_world = n_local  # AABB normals are world-axis aligned

        v_out = np.sum(finger_body.v[idx_all] * n_world, axis=1)
        fn = np.clip(k * depth + c * np.maximum(-v_out, 0.0), 0.0, MAX_FN)
        keep = fn > 0.0
        if not keep.any():
            return f_ext, np.zeros(target_body.x.shape[0] * 3)
        idx = idx_all[keep]
        fn = fn[keep]
        n_world = n_world[keep]

        f_on_finger = fn[:, None] * n_world
        f_view = f_ext.reshape(-1, 3)
        np.add.at(f_view, idx, f_on_finger)

        f_box = self._distribute_reaction_to_box(
            target_body.x, finger_body.x[idx], f_on_finger,
        )
        return f_ext, f_box

    def _project_gripper_deformable_contact(
        self,
        finger_body,
        target_body,
        spec: dict,
    ) -> None:
        """Post-step: push penetrating fingertip nodes to the box AABB surface."""
        TIP_LEN = 0.050
        center, half_ext = self._box_aabb(target_body.x)

        ref_link = finger_body.mesh.nodes
        x_tip = ref_link[:, 0].max() - TIP_LEN
        tip_nodes = np.where(ref_link[:, 0] >= x_tip)[0]

        rel = finger_body.x[tip_nodes] - center
        pen_axes = half_ext - np.abs(rel)
        inside = (pen_axes[:, 0] > 0) & (pen_axes[:, 1] > 0) & (pen_axes[:, 2] > 0)
        if not inside.any():
            return

        idx = tip_nodes[inside]
        rel_a = rel[inside]
        pen_a = pen_axes[inside]
        axis = np.argmin(pen_a, axis=1)
        depth = pen_a[np.arange(len(idx)), axis]
        sign = np.sign(rel_a[np.arange(len(idx)), axis])
        sign[sign == 0] = 1.0
        n_local = np.zeros((len(idx), 3))
        n_local[np.arange(len(idx)), axis] = sign
        n_world = n_local

        finger_body.x[idx] += depth[:, None] * n_world
        v_into = -(finger_body.v[idx] * n_world).sum(axis=1)
        finger_body.v[idx] += np.maximum(v_into, 0.0)[:, None] * n_world

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

        from robosim.scene.handles import (
            FEMBodyHandle, CBBodyHandle, RigidBodyHandle,
            HybridPlasticBodyHandle,
        )
        for bh in scene._body_handles.values():
            if isinstance(bh, (FEMBodyHandle, CBBodyHandle, HybridPlasticBodyHandle)):
                surf = bh._body.mesh.extract_surface()
                color = scene._body_colors.get(bh.name, np.array([0.2, 0.6, 0.9]))
                viewer.add_mesh(bh.name, bh._body.x, surf, color=color, opacity=1.0)
                ctx["meshes"][bh.name] = surf
            elif isinstance(bh, RigidBodyHandle):
                rr = RobotRenderer(bh._model, viewer)
                rr.setup()
                ctx["renderers"][bh.name] = rr

        # ── Deformable grippers ──
        # For each (robot, link) registered via Scene.attach_deformable_gripper:
        #   • hide the URDF visual for that link (prevents Z-fighting),
        #   • add the CB body's surface mesh; positions come straight
        #     from body.x, which _init_physics already lifted into world
        #     coordinates via the link's current FK.
        # The hide key MUST use the URDF <robot name="..."> tag
        # (== rh._model.name), not the user-facing handle name —
        # viewer.set_mesh_visible is a silent no-op when the key is
        # absent, so a wrong prefix is invisible to tests and only
        # catches the eye in the headed viewer.
        ctx["gripper_meshes"] = {}
        from robosim.viz.colormap import gripper_solid_red
        for grip in scene._deformable_grippers:
            rh         = grip["robot"]
            color      = grip["color"]
            use_grad   = grip.get("gradient", False)
            vis_prefix = f"{rh._model.name}/"
            for spec in grip["specs"]:
                link_name = spec["link_name"]
                body      = spec["body"]

                viewer.set_mesh_visible(f"{vis_prefix}{link_name}", False)

                faces     = body.mesh.extract_surface()
                mesh_name = f"defgrip/{rh.name}/{link_name}"
                vtx_color = None
                if use_grad:
                    vtx_color = gripper_solid_red(body.mesh.nodes)
                viewer.add_mesh(mesh_name, body.x, faces,
                                color=color, opacity=1.0,
                                per_vertex_color=vtx_color)
                ctx["gripper_meshes"][mesh_name] = {"body": body}

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

        for mesh_name, info in ctx.get("gripper_meshes", {}).items():
            # body.x is updated each substep by _step_deformable_grippers;
            # just push the latest world-frame node positions.
            viewer.update_mesh_vertices(mesh_name, info["body"].x)

        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle
        for name, surf in ctx["meshes"].items():
            bh = scene._body_handles[name]
            viewer.update_mesh_vertices(name, bh._body.x)

            # Von Mises stress coloring
            if bh._body._dN_list is not None:
                from robosim.physics.fem.assembly import batch_von_mises
                vm = batch_von_mises(
                    bh._body.mesh, bh._body.x, bh._body.material,
                    bh._body._dN_list, bh._body._volumes,
                )
                vm_min = float(vm.min())
                vm_max = float(vm.max())
                if vm_max > vm_min:
                    t = np.clip((vm - vm_min) / (vm_max - vm_min), 0, 1)
                else:
                    t = np.zeros_like(vm)
                # blue → cyan → green → yellow → red
                r = np.clip(t * 2.0 - 0.5, 0, 1).astype(np.float32)
                g = np.clip(1.0 - np.abs(t - 0.5) * 2.0, 0, 1).astype(np.float32)
                b = np.clip(1.0 - t * 2.0 + 0.0, 0, 1).astype(np.float32)
                colors = np.column_stack([r, g, b]).astype(np.float32)
                viewer.update_mesh_color(name, colors)

        # HUD text (skip on clean capture frames — HUD occludes gradient CV)
        cap = scene._finger_capture
        capture_this = (
            cap is not None and (cap._frame_idx % cap.every == 0)
        )
        show_hud = not (cap is not None and cap.clean_hud and capture_this)
        if show_hud:
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
        else:
            viewer.add_text("")

        # Palm-mounted capture camera must be set before the frame is drawn.
        if cap is not None:
            cap.update_palm_camera(viewer, scene)

        # ── Actually render the frame ──────────────────────────────────────────
        viewer._render_frame()

        if cap is not None:
            phase = trajs[0].current_phase if trajs else None
            if not getattr(cap, "_finger_bodies", None):
                bodies = {}
                for grip in scene._deformable_grippers:
                    for spec in grip["specs"]:
                        bodies[spec["link_name"]] = spec["body"]
                cap.set_finger_bodies(bodies)
            cap.process_after_render(
                viewer, self._time, phase, self.sample_finger_probe(),
            )

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
