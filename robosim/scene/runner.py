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
        self._scene     = scene
        self._dt        = dt
        self._substeps  = substeps
        self._time      = 0.0

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

                # ── step all robots (applies CACHED contact reactions before stepping) ──
                for rh in self._scene._robot_handles.values():
                    self._step_robot(rh, dt)

                # ── step all bodies (computes NEW contact reactions, caches for next tick) ──
                for bh in self._scene._body_handles.values():
                    self._step_body(bh, dt)

                self._time += dt

            # ── frame callbacks ──
            if on_step is not None:
                on_step(self._scene, self._time)

            # ── viewer update ──
            if _viewer_ctx is not None:
                self._update_viewer(_viewer_ctx, trajs)

        t_wall = time.time() - t_wall0

        if headless:
            self._print_summary(t_wall)

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
                contact.register_rbd(bh._solver, robot_id=bh.name)
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

        # Cached FEM/CB contact reactions (1-step lag, same pattern as grasp_demo.py).
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

        grip_active = self._scene._grip_active.get(bh.name, False)
        q_fing = robot_handle._model.q[trigger_q]

        if not grip_active and self._time >= lift_start_t and q_fing < trigger_val:
            self._scene._grip_active[bh.name]    = True
            self._scene._grip_palm_pos0[bh.name] = palm_pos.copy()
            self._scene._grip_body_x0[bh.name]   = bh._body.x.copy()
            grip_active = True

        if grip_active:
            palm_pos0 = self._scene._grip_palm_pos0[bh.name]
            body_x0   = self._scene._grip_body_x0[bh.name]
            delta      = palm_pos - palm_pos0

            prev = self._scene._grip_prev_palm.get(bh.name)
            palm_vel = (palm_pos - prev) / dt if prev is not None else np.zeros(3)

            bh._body.x = body_x0 + delta
            if palm_vel[2] > 0.0:
                bh._body.v[:] = palm_vel

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
        """Post-step position projection: push penetrating nodes to finger surface."""
        M_diag  = body._M.diagonal()
        m_nodes = M_diag[0::3]

        for link_name in cp.contact_links:
            li_idx = cp.robot._model.link_index(link_name)
            geom_off, half_ext = cp._geom[link_name]

            T_link     = fk[li_idx]
            T_link_inv = T_link.inverse()
            p_center   = T_link.translation + geom_off  # center of collision box in world

            hx, hy, hz = half_ext

            # Contact axis: perpendicular to the "inner face" of the finger
            # For +Y-axis fingers, the inner face normal is ±Y in world frame.
            # We determine it from the joint axis of the finger link's parent joint.
            # Heuristic: left finger → inner face in -Y direction; right → +Y.
            is_left = "left" in link_name.lower()
            sign    = +1.0 if is_left else -1.0  # +1: inner face is -Y of left finger

            # inner face Y position (world frame)
            inner_y = p_center[1] - sign * hy

            w = np.zeros(6)
            for i in range(body.x.shape[0]):
                xi = body.x[i]
                # Node inside the XZ slab?
                if not (abs(xi[0] - p_center[0]) < hx
                        and abs(xi[2] - p_center[2]) < hz):
                    continue

                mi    = m_nodes[i]
                pen_y = sign * (xi[1] - inner_y)   # >0 means node is inside finger
                if pen_y <= 0.0:
                    continue

                # Project node to finger surface
                body.x[i, 1]  = inner_y
                v_old         = body.v[i, 1]
                body.v[i, 1]  = min(v_old, 0.0) if is_left else max(v_old, 0.0)
                dv            = body.v[i, 1] - v_old

                # Newton 3rd-law reaction wrench on arm link
                f_world  = np.array([0.0, -mi * dv / dt, 0.0])
                f_link   = T_link_inv.apply_vector(f_world)
                p_link   = T_link_inv.apply_point(xi)
                tau_link = np.cross(p_link, f_link)
                w[:3]   += tau_link
                w[3:]   += f_link

            cp._cached_wrenches[li_idx] = w

    # ── CB penalty contact ────────────────────────────────────────────────────

    def _compute_penalty_contact(
        self,
        cp: "_ContactPair",
        fk: list,
        body,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Pre-step penalty forces for CB mode."""
        n     = body.x.shape[0]
        f_ext = np.zeros(n * 3)
        w_map: dict[str, np.ndarray] = {}

        for link_name in cp.contact_links:
            li_idx = cp.robot._model.link_index(link_name)
            geom_off, half_ext = cp._geom[link_name]

            T_link     = fk[li_idx]
            T_link_inv = T_link.inverse()
            p_center   = T_link.translation + geom_off
            hx, hy, hz = half_ext

            is_left = "left" in link_name.lower()
            sign    = +1.0 if is_left else -1.0
            inner_y = p_center[1] - sign * hy

            w = np.zeros(6)
            for i in range(n):
                xi = body.x[i]
                vi = body.v[i]
                if not (abs(xi[0] - p_center[0]) < hx
                        and abs(xi[2] - p_center[2]) < hz):
                    continue
                pen = sign * (xi[1] - inner_y)
                if pen <= 0.0:
                    continue

                v_normal = sign * vi[1]
                fn = cp.k_contact * pen + cp.c_contact * max(0.0, v_normal)

                f_ext[i * 3 + 1] -= sign * fn

                f_world  = np.array([0.0, sign * fn, 0.0])
                f_link   = T_link_inv.apply_vector(f_world)
                p_link   = T_link_inv.apply_point(xi)
                tau_link = np.cross(p_link, f_link)
                w[:3]   += tau_link
                w[3:]   += f_link

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
        import taichi as ti  # noqa: F401
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

    def _update_viewer(self, ctx: dict, trajs: list) -> None:
        viewer = ctx["viewer"]
        scene  = self._scene

        for name, rr in ctx["renderers"].items():
            rr.update()

        from robosim.scene.handles import FEMBodyHandle, CBBodyHandle
        for name, surf in ctx["meshes"].items():
            bh = scene._body_handles[name]
            viewer.update_mesh_vertices(name, bh._body.x)

            # Von Mises stress coloring
            try:
                from robosim.physics.fem.assembly import batch_von_mises
                from robosim.viz.colormap import jet
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
        lines = [f"t = {self._time:.3f} s   phase: {phase_str}"]
        for bh in scene._body_handles.values():
            c = bh.com
            lines.append(
                f"{bh.name}: CoM=({c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f})"
                f"  bot_z={bh.bottom_z:+.4f}"
            )
        viewer.add_text("\n".join(lines) + "\n[LDrag=orbit  Scroll=zoom  ESC=quit]")

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
        for bh in self._scene._body_handles.values():
            c    = bh.com
            z_lo = bh.bottom_z
            lift = max(0.0, z_lo)
            print(f"{bh.name}  CoM=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})"
                  f"  bot_z={z_lo:.4f}  lifted={lift:.4f} m")
            ok = "SUCCESS" if lift > 0.05 else "FAIL — not lifted"
            print(f"  Lift result: {ok}")
        print("=" * 60)
