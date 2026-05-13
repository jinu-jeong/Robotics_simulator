"""ContactSolver: orchestrates detection + response for RBD and FEM bodies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.math.transforms import Transform, cross3
from robosim.model.geometry import GeometryType
from robosim.physics.contact.sdf import ContactPoint, points_ground
from robosim.physics.contact.detection import (
    ContactDetector,
    GroundPlane,
)
from robosim.physics.contact.response import (
    ContactForce,
    ContactParams,
    compute_contact_force,
)
from robosim.physics.contact.fem_contact import (
    FEMSurfaceCollider,
    FEMContactPoint,
    FEMEdgeContactPoint,
    RBDFEMContactPoint,
    detect_fem_fem,
    detect_fem_edge_edge,
    detect_rbd_fem,
    resolve_fem_fem_contacts,
    resolve_fem_edge_contacts,
    resolve_rbd_fem_contacts,
)


def _link_point_velocity(
    robot, link_idx: int, point_world: np.ndarray,
    fk: list | None = None,
) -> np.ndarray:
    """Compute world-frame velocity of a point attached to a link.

    Fast path: looks up the link's spatial velocity from the cache built
    by :meth:`Robot.link_world_velocities` (which propagates joint
    velocities once per ``(q, qd)`` snapshot, instead of per contact).
    """
    if fk is None:
        fk = robot.forward_kinematics()
    omega, v_origin = robot.link_world_velocities()[link_idx]
    link_origin = fk[link_idx].translation
    r = point_world - link_origin
    return v_origin + np.array([
        omega[1]*r[2] - omega[2]*r[1],
        omega[2]*r[0] - omega[0]*r[2],
        omega[0]*r[1] - omega[1]*r[0],
    ])


@dataclass
class PenaltyContactSolver:
    """Explicit penalty contact + regularized kinetic Coulomb friction.

    Normal force: f_n = k * pen - c * v_n  (penalty spring with damping).
    Friction:     |f_t| ≤ μ |f_n|, regularized via v_t / max(|v_t|, eps).

    Limitations vs constraint-based contact:
    * No static (stick) friction — at v_rel → 0 the regularizer scales
      friction to zero, so a held object slides under gravity unless
      held by a kinematic grip + finger-PD lock.
    * Contacts are resolved independently per body, so multi-finger
      grasping does not enforce a joint friction-cone constraint.

    A future ConstraintContactSolver will solve all contacts jointly with
    a full Coulomb cone (PGS / convex), which removes the need for the
    kinematic grip workaround.
    """

    detector: ContactDetector = field(default_factory=lambda: ContactDetector())
    params: ContactParams = field(default_factory=ContactParams)
    _last_forces: list[ContactForce] = field(default_factory=list, repr=False)

    # Mapping: body_id -> (body_type, owner_index)
    _body_map: dict[int, tuple[str, int]] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        ground: GroundPlane | None = None,
        params: ContactParams | None = None,
    ):
        self.detector = ContactDetector(ground=ground)
        self.params = params if params is not None else ContactParams()
        self._last_forces = []
        self._body_map = {}          # bid -> (robot_id, link_idx)
        self._robot_solvers = {}     # robot_id -> rbd_solver
        self._robot_body_ids = {}    # robot_id -> list[bid]
        self._solver_to_rid = {}     # id(solver) -> robot_id
        self._free_body_ids = set()  # robot_ids registered as free-floating
        self._fem_colliders = []

    def _get_robot_id(self, rbd_solver) -> str:
        """Look up registered robot_id for a solver."""
        return self._solver_to_rid.get(id(rbd_solver), rbd_solver.robot.name)

    # ── Friction strategy hook ──
    # Penalty solver: regularised kinetic Coulomb (the long-standing default).
    # ConstraintContactSolver overrides this hook to feed stick_slip + dt.
    def _contact_force(self, contact, v_a, v_b, effective_mass=None):
        return compute_contact_force(
            contact, v_a, v_b, self.params, effective_mass=effective_mass,
        )

    def register_rbd(
        self,
        rbd_solver,
        robot_id: str | None = None,
        is_free_body: bool = False,
    ) -> None:
        """Register collision geometries from robot links.

        Parameters
        ----------
        rbd_solver : RBDSolver to register
        robot_id : unique identifier for this robot (defaults to robot.name)
        is_free_body : True for a free-floating single-body RBD (e.g. a box
                       created by ``create_free_box``). Used by the
                       constraint-mode solver to decide which side of a
                       contact to position-project; the penalty solver
                       ignores this flag.
        """
        robot = rbd_solver.robot
        if robot_id is None:
            robot_id = robot.name
        if is_free_body:
            self._free_body_ids.add(robot_id)
        fk = robot.forward_kinematics()

        self._robot_solvers[robot_id] = rbd_solver
        self._solver_to_rid[id(rbd_solver)] = robot_id
        body_ids: list[int] = []

        for i, link in enumerate(robot.links):
            if not link.collisions:
                continue
            col = link.collisions[0]
            T_world = fk[i].compose(col.origin)
            bid = self.detector.add_body(
                geometry=col.geometry,
                transform=T_world,
                body_type="rbd_link",
                owner_index=i,
            )
            self._body_map[bid] = (robot_id, i)
            body_ids.append(bid)

        self._robot_body_ids[robot_id] = body_ids

        # Auto-filter parent-child link pairs (self-collision prevention)
        self._add_adjacency_filters(robot, body_ids)

    def add_cross_filter(
        self,
        robot_id_a: str, link_name_a: str,
        robot_id_b: str, link_name_b: str,
    ) -> None:
        """Disable collision between specific links on different robots.

        Useful for filtering hand-object contacts during grasping
        (only fingers should contact the grasped object).
        """
        solver_a = self._robot_solvers.get(robot_id_a)
        solver_b = self._robot_solvers.get(robot_id_b)
        if solver_a is None or solver_b is None:
            return

        # Find link indices
        link_idx_a = solver_a.robot.link_index(link_name_a)
        link_idx_b = solver_b.robot.link_index(link_name_b)

        # Find body IDs for those links
        bid_a = bid_b = None
        for bid in self._robot_body_ids.get(robot_id_a, []):
            if self._body_map[bid][1] == link_idx_a:
                bid_a = bid
                break
        for bid in self._robot_body_ids.get(robot_id_b, []):
            if self._body_map[bid][1] == link_idx_b:
                bid_b = bid
                break

        if bid_a is not None and bid_b is not None:
            self.detector.add_filter(bid_a, bid_b)

    def _add_adjacency_filters(self, robot, body_ids: list[int]):
        """Add collision filters for parent-child and sibling link pairs."""
        # Build bid -> link_idx lookup for this robot's bodies
        bid_to_link = {}
        for bid in body_ids:
            _, link_idx = self._body_map[bid]
            bid_to_link[bid] = link_idx

        link_to_bid = {v: k for k, v in bid_to_link.items()}

        for bid in body_ids:
            link_idx = bid_to_link[bid]
            parent_idx = robot.parent_index(link_idx)

            # Filter with direct parent
            if parent_idx >= 0 and parent_idx in link_to_bid:
                self.detector.add_filter(bid, link_to_bid[parent_idx])

            # Filter with grandparent (common for multi-link chains)
            if parent_idx >= 0:
                gp = robot.parent_index(parent_idx)
                if gp >= 0 and gp in link_to_bid:
                    self.detector.add_filter(bid, link_to_bid[gp])

    def update_rbd_transforms(self, rbd_solver=None) -> None:
        """Update collision body transforms from current robot FK.

        If rbd_solver is None, updates all registered robots.
        """
        if rbd_solver is not None:
            self._update_single_rbd(rbd_solver)
        else:
            for solver in self._robot_solvers.values():
                self._update_single_rbd(solver)

    def _update_single_rbd(self, rbd_solver) -> None:
        robot = rbd_solver.robot
        fk = robot.forward_kinematics()
        robot_id = self._get_robot_id(rbd_solver)

        for bid in self._robot_body_ids.get(robot_id, []):
            body = self.detector.bodies[bid]
            link_idx = body.owner_index
            link = robot.links[link_idx]
            if link.collisions:
                col = link.collisions[0]
                body.transform = fk[link_idx].compose(col.origin)

    def compute_rbd_contact_forces(self, rbd_solver) -> dict[int, np.ndarray]:
        """Detect contacts and return per-link spatial wrenches for one robot.

        Returns dict: link_idx -> (6,) wrench [torque(3); force(3)] in link frame.

        Uses a two-pass approach:
          1. Count contacts per link to distribute effective mass.
          2. Compute forces with effective_mass = link_mass / n_contacts,
             preventing over-damped energy injection on light bodies.
        """
        # Update ALL registered robots (so cross-robot contacts are accurate)
        self.update_rbd_transforms()
        contacts = self.detector.detect_all()

        robot = rbd_solver.robot
        robot_id = self._get_robot_id(rbd_solver)
        fk = robot.forward_kinematics()
        wrenches: dict[int, np.ndarray] = {}
        self._last_forces = []

        # Cache FK per robot to avoid redundant recomputation
        fk_cache: dict[str, list] = {robot_id: fk}

        def _get_fk(rid):
            if rid not in fk_cache:
                s = self._robot_solvers.get(rid)
                fk_cache[rid] = s.robot.forward_kinematics() if s else []
            return fk_cache[rid]

        # ── Pass 1: classify contacts and count per-link ──
        # Fast batched path when the C++ ``link_world_velocities`` filled
        # ``_cpp_omega_arr`` / ``_cpp_v_origin_arr`` on the robot. We
        # collect (link_idx, point) pairs and resolve all v_a in one
        # ``batch_point_velocities`` call. v_b (other body) stays
        # per-contact since cross-body work is rare and irregular.
        classified: list[tuple[ContactPoint, int, np.ndarray, np.ndarray]] = []
        contacts_per_link: dict[int, int] = {}

        records = []   # (cp_or_swapped, link_idx, bid_other, side_b)
        # First scan: identify which contacts belong to this robot.
        for cp, bid_a, bid_b in contacts:
            rid_a, link_a = self._body_map.get(bid_a, (None, None))
            if rid_a == robot_id:
                records.append((cp, link_a, bid_b, False))
                continue
            rid_b, link_b = self._body_map.get(bid_b, (None, None))
            if rid_b == robot_id:
                cp_swapped = ContactPoint(
                    point_a=cp.point_b, point_b=cp.point_a,
                    normal=-cp.normal, penetration=cp.penetration,
                )
                records.append((cp_swapped, link_b, bid_a, True))

        # Batched v_a via C++ when available.
        v_a_arr = None
        if records:
            try:
                from robosim.model._cpp_bridge import HAVE_CPP_RBD, _cpp
            except ImportError:
                HAVE_CPP_RBD = False
            # Force the velocity cache to be fresh for the current
            # (q, qd) — this also populates ``_cpp_omega_arr`` /
            # ``_cpp_v_origin_arr`` if the C++ path is active.
            robot.link_world_velocities()
            if (HAVE_CPP_RBD
                    and getattr(robot, "_cpp_omega_arr", None) is not None
                    and getattr(robot, "_cpp_t_arr", None) is not None):
                N = len(records)
                link_idx_arr = np.empty(N, dtype=np.int32)
                pts_arr      = np.empty((N, 3), dtype=np.float64)
                for i, (cp_i, li, _, _) in enumerate(records):
                    link_idx_arr[i] = li
                    pts_arr[i]      = cp_i.point_a
                v_a_arr = _cpp.kin.batch_point_velocities(
                    link_idx_arr, pts_arr,
                    robot._cpp_omega_arr, robot._cpp_v_origin_arr,
                    robot._cpp_t_arr,
                )

        for i, (cp_i, link_idx, bid_other, _) in enumerate(records):
            if v_a_arr is not None:
                v_a = v_a_arr[i]
            else:
                v_a = _link_point_velocity(robot, link_idx, cp_i.point_a, fk=fk)
            v_b = np.zeros(3)
            if bid_other >= 0:
                rid_o, link_o = self._body_map.get(bid_other, (None, None))
                if rid_o is not None:
                    other_solver = self._robot_solvers.get(rid_o)
                    if other_solver:
                        v_b = _link_point_velocity(
                            other_solver.robot, link_o, cp_i.point_b,
                            fk=_get_fk(rid_o))
            classified.append((cp_i, link_idx, v_a, v_b))
            contacts_per_link[link_idx] = contacts_per_link.get(link_idx, 0) + 1

        # ── Pass 2: compute forces with per-contact effective mass ──
        for contact_pt, link_idx, v_a, v_b in classified:
            link_mass = robot.links[link_idx].mass
            n_contacts = contacts_per_link.get(link_idx, 1)

            # Effective mass per contact: total link mass shared among contacts
            eff_mass = None
            if link_mass > 0:
                eff_mass = link_mass / n_contacts

            cf = self._contact_force(
                contact_pt, v_a, v_b, effective_mass=eff_mass,
            )
            if cf is None:
                continue

            self._last_forces.append(cf)

            # Convert world-frame force to link-frame spatial wrench
            T_inv = fk[link_idx].inverse()
            f_link = T_inv.apply_vector(cf.force)
            p_link = T_inv.apply_point(cf.point)
            tau_link = cross3(p_link, f_link)
            wrench = np.concatenate([tau_link, f_link])

            if link_idx in wrenches:
                wrenches[link_idx] += wrench
            else:
                wrenches[link_idx] = wrench

        return wrenches

    # ------------------------------------------------------------------
    # FEM contact — impulse-based (preferred)
    # ------------------------------------------------------------------

    def resolve_fem_contact(
        self,
        fem_body,
        restitution: float = 0.3,
        friction_mu: float = 0.5,
    ) -> int:
        """Impulse-based ground contact for FEM — call AFTER fem_solver.step().

        For each node below the ground plane:
          1. Project position back to ground surface.
          2. Reflect normal velocity with *restitution*.
          3. Apply Coulomb friction to tangential velocity.

        No spring constants to tune. Energy behaviour is controlled
        directly by *restitution* (0 = perfectly inelastic, 1 = elastic).

        Returns the number of active contact nodes.
        """
        if self.detector.ground is None:
            return 0

        gh = self.detector.ground.height
        gn = self.detector.ground.normal
        gn = gn / np.linalg.norm(gn)

        dists = fem_body.x @ gn - gh  # signed distance per node
        below = np.where(dists < 0)[0]

        self._last_forces = []

        for idx in below:
            # --- position: project onto ground ---
            fem_body.x[idx] -= dists[idx] * gn

            # --- velocity decomposition ---
            v = fem_body.v[idx]
            v_n = float(np.dot(v, gn))
            v_t = v - v_n * gn

            # normal: reflect with restitution (only if approaching)
            if v_n < 0:
                v_n_new = -restitution * v_n
            else:
                v_n_new = v_n

            # tangential: Coulomb friction
            v_t_mag = np.linalg.norm(v_t)
            if v_t_mag > 1e-12 and v_n < 0:
                # max friction impulse ∝ normal impulse
                max_dv_t = friction_mu * (1.0 + restitution) * abs(v_n)
                scale = min(1.0, max_dv_t / v_t_mag)
                v_t *= (1.0 - scale)

            fem_body.v[idx] = v_n_new * gn + v_t

            # Record for visualisation
            self._last_forces.append(ContactForce(
                point=fem_body.x[idx].copy(),
                force=(v_n_new - v_n) * gn,  # impulse direction (informational)
                normal_force=abs(v_n_new - v_n),
            ))

        return len(below)

    # ------------------------------------------------------------------
    # FEM contact — penalty-based (legacy, kept for RBD compatibility)
    # ------------------------------------------------------------------

    def compute_fem_contact_forces(self, fem_body) -> np.ndarray:
        """Penalty-based ground contact for FEM surface nodes.

        Prefer :meth:`resolve_fem_contact` for stability.
        """
        n_dof = fem_body.mesh.n_nodes * 3
        f_contact = np.zeros(n_dof)

        if self.detector.ground is None:
            return f_contact

        gh = self.detector.ground.height
        gn = self.detector.ground.normal

        surface_nodes = np.unique(fem_body.mesh.extract_surface())
        positions = fem_body.x[surface_nodes]

        node_contacts = points_ground(positions, gh, gn)
        M_diag = fem_body._M.diagonal()

        self._last_forces = []
        for local_idx, cp in node_contacts:
            node_idx = surface_nodes[local_idx]
            v_node = fem_body.v[node_idx]
            node_mass = float(M_diag[node_idx * 3])

            cf = self._contact_force(
                cp, v_node, np.zeros(3), effective_mass=node_mass,
            )
            if cf is None:
                continue

            self._last_forces.append(cf)
            f_contact[node_idx * 3: node_idx * 3 + 3] += cf.force

        return f_contact

    # ------------------------------------------------------------------
    # FEM-FEM contact
    # ------------------------------------------------------------------

    _fem_colliders: list[FEMSurfaceCollider] = field(default_factory=list, repr=False)

    def register_fem(self, fem_solver) -> None:
        """Build surface colliders for all FEM bodies."""
        self._fem_colliders = []
        for i, body in enumerate(fem_solver.bodies):
            collider = FEMSurfaceCollider.from_body(body, body_idx=i)
            self._fem_colliders.append(collider)

    def resolve_fem_fem_all(
        self,
        fem_solver,
        restitution: float = 0.1,
        friction_mu: float = 0.5,
        d_hat: float = 0.005,
    ) -> int:
        """Detect and resolve all FEM-FEM contacts.

        Call AFTER fem_solver.step(). Tests every pair of FEM bodies.

        Returns total number of contacts resolved.
        """
        if len(self._fem_colliders) < 2:
            return 0

        bodies = fem_solver.bodies
        n = len(bodies)
        total = 0

        # Update AABBs
        for collider in self._fem_colliders:
            collider.update_aabb(bodies[collider.body_idx].x)

        # All pairs
        for i in range(n):
            for j in range(i + 1, n):
                ci, cj = self._fem_colliders[i], self._fem_colliders[j]
                xi, xj = bodies[i].x, bodies[j].x

                # ── Vertex-face (both directions) ──
                contacts_vf = detect_fem_fem(ci, cj, xi, xj, d_hat)
                contacts_vf += detect_fem_fem(cj, ci, xj, xi, d_hat)
                total += resolve_fem_fem_contacts(
                    contacts_vf, bodies, restitution, friction_mu,
                )

                # ── Edge-edge ──
                contacts_ee = detect_fem_edge_edge(ci, cj, xi, xj, d_hat)
                total += resolve_fem_edge_contacts(
                    contacts_ee, bodies, restitution, friction_mu,
                )

        return total

    # ------------------------------------------------------------------
    # RBD-FEM contact
    # ------------------------------------------------------------------

    def compute_rbd_fem_forces(
        self,
        rbd_solver,
        fem_solver,
        d_hat: float = 0.005,
        stiffness: float = 1e4,
        friction_mu: float = 0.5,
        damping_ratio: float = 0.5,
    ) -> dict[int, np.ndarray]:
        """Compute penalty-based contact forces from RBD onto FEM bodies.

        Returns forces suitable for passing as ``extra_forces`` to
        :meth:`FEMSolver.step`.  Call **before** the FEM step so the
        implicit integrator incorporates the contact.

        For each FEM surface node within *d_hat* of an RBD collision
        geometry, a spring-damper penalty force (normal) and a friction
        drag force (tangential) are computed.

        Parameters
        ----------
        stiffness : penalty spring stiffness (N/m)
        friction_mu : Coulomb friction coefficient (tangential drag)
        damping_ratio : fraction of critical damping for penalty

        Returns ``{body_idx: (n_dof,) force_array}``.
        """
        if not self._fem_colliders:
            return {}

        robot = rbd_solver.robot
        fk = robot.forward_kinematics()
        bodies = fem_solver.bodies

        # Update FEM AABBs
        for collider in self._fem_colliders:
            collider.update_aabb(bodies[collider.body_idx].x)

        result: dict[int, np.ndarray] = {}

        for collider in self._fem_colliders:
            contacts = detect_rbd_fem(
                robot, fk, collider,
                bodies[collider.body_idx].x, d_hat,
            )
            if not contacts:
                continue

            body = bodies[collider.body_idx]
            n_dof = body.mesh.n_nodes * 3
            f = result.get(collider.body_idx)
            if f is None:
                f = np.zeros(n_dof)
                result[collider.body_idx] = f

            M_diag = body._M.diagonal()

            for c in contacts:
                node = c.fem_node_idx
                normal = c.normal

                node_mass = float(M_diag[node * 3])
                if node_mass < 1e-12:
                    continue

                # ── Normal penalty force ──
                pen = c.penetration           # d_hat-gap for proximity
                c_damp = damping_ratio * 2.0 * np.sqrt(stiffness * node_mass)

                v_fem = body.v[node]
                v_rbd = _link_point_velocity(
                    robot, c.link_idx, c.rbd_contact_point, fk=fk)
                v_rel = v_fem - v_rbd
                v_n = float(np.dot(v_rel, normal))

                f_n_mag = stiffness * pen - c_damp * v_n
                if f_n_mag < 0:
                    f_n_mag = 0.0

                f_node = f_n_mag * normal

                # ── Tangential friction (drag toward RBD velocity) ──
                v_t = v_rel - v_n * normal
                v_t_mag = np.linalg.norm(v_t)
                if v_t_mag > 1e-10:
                    f_t_max = friction_mu * f_n_mag
                    f_t_drag = c_damp * v_t_mag  # viscous drag
                    f_t_mag = min(f_t_max, f_t_drag)
                    f_node -= f_t_mag * (v_t / v_t_mag)

                f[node * 3: node * 3 + 3] += f_node

        return result

    def resolve_rbd_fem_all(
        self,
        rbd_solver,
        fem_solver,
        restitution: float = 0.1,
        friction_mu: float = 0.5,
        d_hat: float = 0.005,
    ) -> int:
        """Detect and resolve all RBD-FEM contacts.

        Call AFTER both rbd_solver.step() and fem_solver.step().
        The RBD is treated as infinite mass (already committed).

        Returns total number of contacts resolved.
        """
        if not self._fem_colliders:
            return 0

        robot = rbd_solver.robot
        fk = robot.forward_kinematics()
        bodies = fem_solver.bodies
        total = 0

        # Update FEM AABBs
        for collider in self._fem_colliders:
            collider.update_aabb(bodies[collider.body_idx].x)

        for collider in self._fem_colliders:
            contacts = detect_rbd_fem(robot, fk, collider, bodies[collider.body_idx].x, d_hat)
            total += resolve_rbd_fem_contacts(
                contacts, robot, bodies, restitution, friction_mu,
            )

        return total

    def compute_all_rbd_contact_forces(
        self,
    ) -> dict[str, dict[int, np.ndarray]]:
        """Detect contacts once and compute forces for ALL registered robots.

        Ensures consistent contact detection across all bodies. Returns::

            { robot_id: { link_idx: wrench(6,) } }

        Call this once per timestep, then apply forces and step all robots.
        """
        self.update_rbd_transforms()
        contacts = self.detector.detect_all()

        # Prepare result per robot
        all_wrenches: dict[str, dict[int, np.ndarray]] = {
            rid: {} for rid in self._robot_solvers
        }
        self._last_forces = []

        # Cache FK per robot to avoid redundant recomputation
        fk_cache: dict[str, list] = {}

        def _get_fk(rid):
            if rid not in fk_cache:
                s = self._robot_solvers.get(rid)
                fk_cache[rid] = s.robot.forward_kinematics() if s else []
            return fk_cache[rid]

        # ── Pass 1: classify contacts per robot, count per-link ──
        # Key: (robot_id, link_idx) → list of (ContactPoint, v_a, v_b)
        classified: dict[tuple[str, int], list] = {}
        contacts_per_link: dict[tuple[str, int], int] = {}

        for cp, bid_a, bid_b in contacts:
            rid_a, link_a = self._body_map.get(bid_a, (None, None))
            rid_b, link_b = self._body_map.get(bid_b, (None, None))

            # For each body in the contact that belongs to a robot,
            # record the contact from that body's perspective.
            entries = []

            if rid_a is not None and rid_a in self._robot_solvers:
                robot_a = self._robot_solvers[rid_a].robot
                fk_a = _get_fk(rid_a)
                v_a = _link_point_velocity(robot_a, link_a, cp.point_a, fk=fk_a)
                v_b = np.zeros(3)
                if rid_b is not None:
                    robot_b_solver = self._robot_solvers.get(rid_b)
                    if robot_b_solver:
                        fk_b_tmp = _get_fk(rid_b)
                        v_b = _link_point_velocity(
                            robot_b_solver.robot, link_b, cp.point_b,
                            fk=fk_b_tmp)
                entries.append((rid_a, link_a, cp, v_a, v_b, fk_a))

            if rid_b is not None and rid_b in self._robot_solvers and bid_b >= 0:
                robot_b = self._robot_solvers[rid_b].robot
                fk_b = _get_fk(rid_b)
                cp_swapped = ContactPoint(
                    point_a=cp.point_b, point_b=cp.point_a,
                    normal=-cp.normal, penetration=cp.penetration,
                )
                v_a_b = _link_point_velocity(robot_b, link_b, cp_swapped.point_a,
                                             fk=fk_b)
                v_b_b = np.zeros(3)
                if rid_a is not None:
                    robot_a_solver = self._robot_solvers.get(rid_a)
                    if robot_a_solver:
                        fk_a_tmp = _get_fk(rid_a)
                        v_b_b = _link_point_velocity(
                            robot_a_solver.robot, link_a, cp_swapped.point_b,
                            fk=fk_a_tmp)
                entries.append((rid_b, link_b, cp_swapped, v_a_b, v_b_b, fk_b))

            for rid, lidx, cpt, va, vb, fk in entries:
                key = (rid, lidx)
                if key not in classified:
                    classified[key] = []
                classified[key].append((cpt, va, vb, fk))
                contacts_per_link[key] = contacts_per_link.get(key, 0) + 1

        # ── Pass 2: compute forces ──
        for (rid, lidx), contact_list in classified.items():
            robot = self._robot_solvers[rid].robot
            link_mass = robot.links[lidx].mass
            n_contacts = contacts_per_link[(rid, lidx)]

            eff_mass = None
            if link_mass > 0:
                eff_mass = link_mass / n_contacts

            for cpt, va, vb, fk in contact_list:
                cf = self._contact_force(
                    cpt, va, vb, effective_mass=eff_mass,
                )
                if cf is None:
                    continue
                self._last_forces.append(cf)

                T_inv = fk[lidx].inverse()
                f_link = T_inv.apply_vector(cf.force)
                p_link = T_inv.apply_point(cf.point)
                tau_link = cross3(p_link, f_link)
                wrench = np.concatenate([tau_link, f_link])

                w_dict = all_wrenches[rid]
                if lidx in w_dict:
                    w_dict[lidx] += wrench
                else:
                    w_dict[lidx] = wrench

        return all_wrenches

    @property
    def last_forces(self) -> list[ContactForce]:
        return self._last_forces


# Backward-compat alias. Existing code (Scene, demos, tests) imports
# ``ContactSolver`` directly; the rename to PenaltyContactSolver is internal
# until the constraint-based solver lands and an ABC is extracted.
ContactSolver = PenaltyContactSolver
