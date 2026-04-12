"""ContactSolver: orchestrates detection + response for RBD and FEM bodies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.math.transforms import Transform
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
    RBDFEMContactPoint,
    detect_fem_fem,
    detect_rbd_fem,
    resolve_fem_fem_contacts,
    resolve_rbd_fem_contacts,
)


def _link_point_velocity(
    robot, link_idx: int, point_world: np.ndarray,
) -> np.ndarray:
    """Compute world-frame velocity of a point attached to a link.

    Uses joint velocity propagation through the kinematic chain.
    """
    from robosim.math.spatial import (
        joint_transform,
        motion_subspace_revolute,
        motion_subspace_prismatic,
    )
    from robosim.model.joint import JointType

    fk = robot.forward_kinematics()
    n_links = robot.n_links

    # Propagate spatial velocities from root to link_idx
    # Spatial velocity: [omega(3); v(3)] in world frame
    omega = np.zeros(3)  # angular velocity in world frame
    v_origin = np.zeros(3)  # linear velocity of link origin in world frame

    # Build path from root to link_idx
    path = []
    idx = link_idx
    while idx >= 0:
        path.append(idx)
        idx = robot.parent_index(idx)
    path.reverse()

    for i in path:
        parent_idx = robot.parent_index(i)
        if parent_idx < 0:
            # Root link: no parent velocity
            pass

        # Add joint velocity contribution
        j_idx = robot.joint_index_for_link(i)
        if j_idx is not None:
            joint = robot.joints[j_idx]
            q_j = robot.q[j_idx]
            qd_j = robot.qd[j_idx]

            # Joint axis in world frame
            T_world_link = fk[i]
            if joint.joint_type in (JointType.REVOLUTE, JointType.CONTINUOUS):
                axis_world = T_world_link.rotation @ joint.axis
                omega += qd_j * axis_world
            elif joint.joint_type == JointType.PRISMATIC:
                axis_world = T_world_link.rotation @ joint.axis
                v_origin += qd_j * axis_world

    # Velocity at point: v = v_origin + omega x (point - origin)
    link_origin = fk[link_idx].translation
    r = point_world - link_origin
    v_point = v_origin + np.cross(omega, r)
    return v_point


@dataclass
class ContactSolver:
    """Manages contact detection and response."""

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
        self._fem_colliders = []

    def register_rbd(self, rbd_solver, robot_id: str | None = None) -> None:
        """Register collision geometries from robot links.

        Parameters
        ----------
        rbd_solver : RBDSolver to register
        robot_id : unique identifier for this robot (defaults to robot.name)
        """
        robot = rbd_solver.robot
        if robot_id is None:
            robot_id = robot.name
        fk = robot.forward_kinematics()

        self._robot_solvers[robot_id] = rbd_solver
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
        robot_id = robot.name

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
        """
        # Update ALL registered robots (so cross-robot contacts are accurate)
        self.update_rbd_transforms()
        contacts = self.detector.detect_all()

        robot = rbd_solver.robot
        robot_id = robot.name
        fk = robot.forward_kinematics()
        wrenches: dict[int, np.ndarray] = {}
        self._last_forces = []

        for cp, bid_a, bid_b in contacts:
            # Apply force to body_a if it belongs to this robot
            rid_a, link_a = self._body_map.get(bid_a, (None, None))
            if rid_a != robot_id:
                # Also check body_b (swap direction)
                rid_b, link_b = self._body_map.get(bid_b, (None, None))
                if rid_b != robot_id:
                    continue
                # Swap: this contact has our robot on the B side
                cp_swapped = ContactPoint(
                    point_a=cp.point_b, point_b=cp.point_a,
                    normal=-cp.normal, penetration=cp.penetration,
                )
                link_idx = link_b
                v_a = _link_point_velocity(robot, link_idx, cp_swapped.point_a)

                # Other body velocity
                v_b = np.zeros(3)
                if rid_a is not None:
                    other_solver = self._robot_solvers.get(rid_a)
                    if other_solver:
                        v_b = _link_point_velocity(other_solver.robot, link_a, cp_swapped.point_b)

                cf = compute_contact_force(cp_swapped, v_a, v_b, self.params)
                if cf is None:
                    continue
                self._last_forces.append(cf)

                T_inv = fk[link_idx].inverse()
                f_link = T_inv.apply_vector(cf.force)
                p_link = T_inv.apply_point(cf.point)
                tau_link = np.cross(p_link, f_link)
                wrench = np.concatenate([tau_link, f_link])
                if link_idx in wrenches:
                    wrenches[link_idx] += wrench
                else:
                    wrenches[link_idx] = wrench
                continue

            link_idx = link_a

            # Velocity at contact point
            v_a = _link_point_velocity(robot, link_idx, cp.point_a)
            v_b = np.zeros(3)  # ground or other body

            if bid_b >= 0:
                rid_b, link_b = self._body_map.get(bid_b, (None, None))
                if rid_b is not None:
                    other_solver = self._robot_solvers.get(rid_b)
                    if other_solver:
                        v_b = _link_point_velocity(other_solver.robot, link_b, cp.point_b)

            cf = compute_contact_force(cp, v_a, v_b, self.params)
            if cf is None:
                continue

            self._last_forces.append(cf)

            # Convert world-frame force to link-frame spatial wrench
            T_inv = fk[link_idx].inverse()
            f_link = T_inv.apply_vector(cf.force)
            p_link = T_inv.apply_point(cf.point)
            tau_link = np.cross(p_link, f_link)
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

            cf = compute_contact_force(
                cp, v_node, np.zeros(3), self.params,
                effective_mass=node_mass,
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
                # A nodes vs B faces
                contacts_ab = detect_fem_fem(
                    self._fem_colliders[i], self._fem_colliders[j],
                    bodies[i].x, bodies[j].x, d_hat,
                )
                # B nodes vs A faces
                contacts_ba = detect_fem_fem(
                    self._fem_colliders[j], self._fem_colliders[i],
                    bodies[j].x, bodies[i].x, d_hat,
                )
                total += resolve_fem_fem_contacts(
                    contacts_ab + contacts_ba, bodies, restitution, friction_mu,
                )

        return total

    # ------------------------------------------------------------------
    # RBD-FEM contact
    # ------------------------------------------------------------------

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

    @property
    def last_forces(self) -> list[ContactForce]:
        return self._last_forces
