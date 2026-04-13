"""FEM surface collision detection and response.

Provides:
  - FEMSurfaceCollider : wraps a DeformableBody for collision queries
  - detect_fem_fem     : vertex-face proximity between two FEM surfaces
  - detect_rbd_fem     : RBD primitive vs FEM surface triangles
  - resolve_fem_fem_contact  : impulse-based FEM-FEM contact
  - resolve_rbd_fem_contact  : impulse-based RBD-FEM contact
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.physics.contact.sdf import (
    ContactPoint,
    point_triangle_distance,
    triangle_normal,
    sphere_triangle,
)


# ═══════════════════════════════════════════════════════════════
# FEM Surface Collider
# ═══════════════════════════════════════════════════════════════

@dataclass
class FEMSurfaceCollider:
    """Collision query wrapper for a DeformableBody's surface.

    Caches surface faces, unique surface node indices, and AABBs for
    broad-phase culling.
    """

    body_idx: int                       # index in FEMSolver.bodies
    surface_faces: np.ndarray           # (n_faces, 3) node indices
    surface_nodes: np.ndarray           # unique surface node indices
    _aabb_min: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)
    _aabb_max: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False)

    @classmethod
    def from_body(cls, body, body_idx: int) -> "FEMSurfaceCollider":
        """Build from a DeformableBody."""
        faces = body.mesh.extract_surface()
        nodes = np.unique(faces.ravel())
        return cls(body_idx=body_idx, surface_faces=faces, surface_nodes=nodes)

    def update_aabb(self, positions: np.ndarray, margin: float = 0.01):
        """Recompute AABB from current node positions."""
        surf_pos = positions[self.surface_nodes]
        self._aabb_min = surf_pos.min(axis=0) - margin
        self._aabb_max = surf_pos.max(axis=0) + margin

    def aabb_overlaps(self, other: "FEMSurfaceCollider") -> bool:
        """Test AABB overlap with another FEM surface."""
        return bool(
            np.all(self._aabb_min <= other._aabb_max) and
            np.all(other._aabb_min <= self._aabb_max)
        )

    def aabb_overlaps_box(self, box_min: np.ndarray, box_max: np.ndarray) -> bool:
        """Test AABB overlap with an axis-aligned box."""
        return bool(
            np.all(self._aabb_min <= box_max) and
            np.all(box_min <= self._aabb_max)
        )

    @property
    def n_faces(self) -> int:
        return self.surface_faces.shape[0]


# ═══════════════════════════════════════════════════════════════
# FEM-FEM Contact Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class FEMContactPoint:
    """Contact between two FEM bodies."""
    node_idx: int           # node index on body A (the penetrating vertex)
    face_idx: int           # face index on body B
    bary: np.ndarray        # (3,) barycentric coords on face B
    normal: np.ndarray      # (3,) contact normal (B face → A node)
    penetration: float      # positive = overlap
    body_a_idx: int = 0     # index in FEMSolver.bodies
    body_b_idx: int = 0


def detect_fem_fem(
    collider_a: FEMSurfaceCollider,
    collider_b: FEMSurfaceCollider,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    d_hat: float = 0.005,
) -> list[FEMContactPoint]:
    """Vertex-face proximity detection between two FEM surfaces.

    For each surface node of body A, finds the closest triangle on body B.
    If distance < d_hat, creates a contact.

    Parameters
    ----------
    collider_a, collider_b : surface colliders
    pos_a, pos_b : (n_nodes, 3) current positions
    d_hat : contact distance threshold (m)

    Returns
    -------
    contacts : list of FEMContactPoint
    """
    # Broad phase: AABB check
    if not collider_a.aabb_overlaps(collider_b):
        return []

    contacts = []
    faces_b = collider_b.surface_faces
    n_faces_b = faces_b.shape[0]

    # Precompute face B triangle vertices
    tri_b = pos_b[faces_b]  # (n_faces_b, 3, 3)

    # Per-face AABB for body B (for per-vertex culling)
    face_min_b = tri_b.min(axis=1) - d_hat  # (n_faces_b, 3)
    face_max_b = tri_b.max(axis=1) + d_hat

    # Test each surface node of A against faces of B
    for node_a in collider_a.surface_nodes:
        p = pos_a[node_a]

        # Cull: check node against each face AABB
        in_range = np.all(p >= face_min_b, axis=1) & np.all(p <= face_max_b, axis=1)
        candidate_faces = np.where(in_range)[0]

        if len(candidate_faces) == 0:
            continue

        best_dist = d_hat
        best_contact = None

        for fi in candidate_faces:
            v0, v1, v2 = tri_b[fi]
            closest, dist, bary = point_triangle_distance(p, v0, v1, v2)

            if dist < best_dist:
                best_dist = dist
                # Normal: from face toward node
                n = triangle_normal(v0, v1, v2)
                # Ensure normal points from B toward A (toward the node)
                if np.dot(n, p - closest) < 0:
                    n = -n

                best_contact = FEMContactPoint(
                    node_idx=int(node_a),
                    face_idx=int(fi),
                    bary=bary,
                    normal=n,
                    penetration=d_hat - dist,  # positive when within d_hat
                    body_a_idx=collider_a.body_idx,
                    body_b_idx=collider_b.body_idx,
                )

        if best_contact is not None:
            contacts.append(best_contact)

    return contacts


# ═══════════════════════════════════════════════════════════════
# RBD-FEM Contact Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class RBDFEMContactPoint:
    """Contact between an RBD link and an FEM surface node."""
    fem_node_idx: int       # FEM node that penetrates
    link_idx: int           # RBD link index
    normal: np.ndarray      # (3,) contact normal (RBD surface → FEM node)
    penetration: float
    rbd_contact_point: np.ndarray  # (3,) contact point on RBD surface
    fem_body_idx: int = 0
    gap: float = 0.0        # 0 = true overlap; >0 = proximity distance


def detect_rbd_fem(
    robot,
    fk: list,
    fem_collider: FEMSurfaceCollider,
    fem_pos: np.ndarray,
    d_hat: float = 0.005,
) -> list[RBDFEMContactPoint]:
    """Detect contacts between RBD link geometries and FEM surface nodes.

    For each collision geometry on the robot, tests proximity against
    each FEM surface node.

    Parameters
    ----------
    robot : Robot model
    fk : forward kinematics transforms
    fem_collider : FEM surface collider
    fem_pos : (n_nodes, 3) current FEM positions
    d_hat : contact distance threshold

    Returns
    -------
    contacts : list of RBDFEMContactPoint
    """
    from robosim.model.geometry import GeometryType

    contacts = []

    for link_idx, link in enumerate(robot.links):
        if not link.collisions:
            continue

        col = link.collisions[0]
        geom = col.geometry
        T_world = fk[link_idx].compose(col.origin)
        center = T_world.translation
        R = T_world.rotation

        # Quick AABB check
        if geom.geometry_type == GeometryType.SPHERE:
            r = geom.radius + d_hat
            geom_min = center - r
            geom_max = center + r
        elif geom.geometry_type == GeometryType.BOX:
            he = geom.size / 2.0 + d_hat
            extent = np.abs(R) @ he
            geom_min = center - extent
            geom_max = center + extent
        elif geom.geometry_type == GeometryType.CYLINDER:
            bound = np.sqrt(geom.radius**2 + (geom.length / 2)**2) + d_hat
            geom_min = center - bound
            geom_max = center + bound
        else:
            continue

        if not fem_collider.aabb_overlaps_box(geom_min, geom_max):
            continue

        # Test each FEM surface node against this geometry
        for node_idx in fem_collider.surface_nodes:
            p = fem_pos[node_idx]

            # Point-in-AABB check
            if not (np.all(p >= geom_min) and np.all(p <= geom_max)):
                continue

            cp = _point_vs_geometry(p, geom, center, R, d_hat=d_hat)
            if cp is not None and cp.penetration > 0:
                # Determine gap: 0 for actual overlap, >0 for proximity
                # _point_vs_geometry sets pen=d_hat-gap for outside points
                # and pen=face_depth for inside points (which can be > d_hat).
                # When pen <= d_hat the node is outside (proximity).
                gap = max(0.0, d_hat - cp.penetration)
                contacts.append(RBDFEMContactPoint(
                    fem_node_idx=int(node_idx),
                    link_idx=link_idx,
                    normal=cp.normal,
                    penetration=cp.penetration,
                    rbd_contact_point=cp.point_b.copy(),
                    fem_body_idx=fem_collider.body_idx,
                    gap=gap,
                ))

    return contacts


def _point_vs_geometry(point, geom, center, R, d_hat: float = 0.0) -> ContactPoint | None:
    """Test a point against a primitive geometry (sphere, box, cylinder).

    Returns ContactPoint if point is inside the geometry, **or** within
    *d_hat* of its surface (proximity detection).

    When *d_hat* == 0, only penetration (point-inside) contacts are
    reported — backward-compatible with the original behaviour.

    Normal always points from the geometry surface toward the point.
    Penetration is positive both for true overlaps and for proximity
    contacts (for proximity: ``penetration = d_hat - gap``).
    """
    from robosim.model.geometry import GeometryType

    if geom.geometry_type == GeometryType.SPHERE:
        diff = point - center
        dist = np.linalg.norm(diff)
        gap = dist - geom.radius          # >0 outside, <0 inside

        if gap > d_hat:
            return None

        if dist < 1e-12:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = diff / dist

        surface_pt = center + geom.radius * n
        pen = -gap if gap <= 0 else d_hat - gap
        return ContactPoint(
            point_a=point.copy(), point_b=surface_pt,
            normal=n, penetration=pen,
        )

    elif geom.geometry_type == GeometryType.BOX:
        p_local = R.T @ (point - center)
        he = geom.size / 2.0

        inside = np.all(np.abs(p_local) <= he)

        if inside:
            # Point inside box — standard penetration
            face_dists = he - np.abs(p_local)
            axis = int(np.argmin(face_dists))
            pen = face_dists[axis]
            n_local = np.zeros(3)
            n_local[axis] = np.sign(p_local[axis]) if abs(p_local[axis]) > 1e-12 else 1.0
            surface_local = p_local.copy()
            surface_local[axis] = np.sign(p_local[axis]) * he[axis]
        else:
            # Point outside — compute closest point on box surface
            clamped = np.clip(p_local, -he, he)
            diff = p_local - clamped
            gap = np.linalg.norm(diff)

            if gap > d_hat:
                return None

            pen = d_hat - gap
            if gap < 1e-12:
                # Degenerate (on a face/edge/corner): pick closest face
                face_dists = he - np.abs(p_local)
                axis = int(np.argmin(np.abs(face_dists)))
                n_local = np.zeros(3)
                n_local[axis] = np.sign(p_local[axis]) if abs(p_local[axis]) > 1e-12 else 1.0
            else:
                n_local = diff / gap
            surface_local = clamped

        n_world = R @ n_local
        surface_pt = R @ surface_local + center
        return ContactPoint(
            point_a=point.copy(), point_b=surface_pt,
            normal=n_world, penetration=pen,
        )

    elif geom.geometry_type == GeometryType.CYLINDER:
        p_local = R.T @ (point - center)
        hl = geom.length / 2.0
        r = geom.radius
        r_xy = np.sqrt(p_local[0]**2 + p_local[1]**2)

        inside_radial = r_xy <= r
        inside_axial = abs(p_local[2]) <= hl

        if inside_radial and inside_axial:
            # Inside cylinder — standard penetration
            pen_radial = r - r_xy
            pen_axial = hl - abs(p_local[2])

            if pen_radial < pen_axial:
                if r_xy < 1e-12:
                    n_local = np.array([1.0, 0.0, 0.0])
                else:
                    n_local = np.array([p_local[0], p_local[1], 0.0]) / r_xy
                pen = pen_radial
                surface_local = np.array([r * n_local[0], r * n_local[1], p_local[2]])
            else:
                n_local = np.array([0.0, 0.0, np.sign(p_local[2])])
                pen = pen_axial
                surface_local = np.array([p_local[0], p_local[1], np.sign(p_local[2]) * hl])
        else:
            # Outside cylinder — closest point on surface
            clamped_z = np.clip(p_local[2], -hl, hl)
            if r_xy < 1e-12:
                clamped_xy = np.array([0.0, 0.0])
            else:
                clamped_xy = np.array([p_local[0], p_local[1]]) * min(r / r_xy, 1.0)

            closest_local = np.array([clamped_xy[0], clamped_xy[1], clamped_z])
            diff = p_local - closest_local
            gap = np.linalg.norm(diff)

            if gap > d_hat:
                return None

            pen = d_hat - gap
            if gap < 1e-12:
                n_local = np.array([1.0, 0.0, 0.0])
            else:
                n_local = diff / gap
            surface_local = closest_local

        n_world = R @ n_local
        surface_pt = R @ surface_local + center
        return ContactPoint(
            point_a=point.copy(), point_b=surface_pt,
            normal=n_world, penetration=pen,
        )

    return None


# ═══════════════════════════════════════════════════════════════
# Contact Response: Impulse-based
# ═══════════════════════════════════════════════════════════════

def resolve_fem_fem_contacts(
    contacts: list[FEMContactPoint],
    bodies: list,
    restitution: float = 0.1,
    friction_mu: float = 0.5,
) -> int:
    """Apply impulse-based contact response for FEM-FEM collisions.

    For each contact:
      1. Compute relative velocity at contact point
      2. Split position correction and velocity impulse by mass ratio
      3. Apply Coulomb friction

    Parameters
    ----------
    contacts : detected FEM-FEM contacts
    bodies : list of DeformableBody (from FEMSolver.bodies)
    restitution : velocity restitution coefficient
    friction_mu : Coulomb friction coefficient

    Returns number of contacts resolved.
    """
    n_resolved = 0
    for c in contacts:
        body_a = bodies[c.body_a_idx]
        body_b = bodies[c.body_b_idx]

        node_a = c.node_idx
        face_nodes_b = body_b.mesh.extract_surface()[c.face_idx]
        bary = c.bary
        normal = c.normal

        if c.penetration <= 0:
            continue

        # Mass of node A
        m_a = float(body_a._M.diagonal()[node_a * 3])

        # Effective mass of face B (barycentric weighted)
        M_diag_b = body_b._M.diagonal()
        m_b_nodes = np.array([float(M_diag_b[ni * 3]) for ni in face_nodes_b])
        m_b_eff = 1.0 / (np.sum(bary**2 / np.maximum(m_b_nodes, 1e-10)) + 1e-10)

        m_total = m_a + m_b_eff
        if m_total < 1e-10:
            continue

        w_a = m_b_eff / m_total   # weight for A (heavier B → A moves more)
        w_b = m_a / m_total

        # --- Position correction ---
        correction = c.penetration * normal
        body_a.x[node_a] += w_a * correction
        for i, ni in enumerate(face_nodes_b):
            body_b.x[ni] -= w_b * bary[i] * correction

        # --- Velocity impulse ---
        v_a = body_a.v[node_a]
        v_b_contact = sum(bary[i] * body_b.v[face_nodes_b[i]] for i in range(3))

        v_rel = v_a - v_b_contact
        v_n = float(np.dot(v_rel, normal))

        # Only resolve if approaching
        if v_n >= 0:
            n_resolved += 1
            continue

        # Normal impulse magnitude
        j_n = -(1.0 + restitution) * v_n / (1.0 / m_a + 1.0 / m_b_eff)

        # Apply normal impulse
        body_a.v[node_a] += (j_n / m_a) * normal
        for i, ni in enumerate(face_nodes_b):
            body_b.v[ni] -= (j_n * bary[i] / max(m_b_nodes[i], 1e-10)) * normal

        # --- Coulomb friction ---
        v_t = v_rel - v_n * normal
        v_t_mag = np.linalg.norm(v_t)
        if v_t_mag > 1e-10:
            j_t_max = friction_mu * abs(j_n)
            # Friction impulse (clamped)
            j_t = min(j_t_max, m_a * m_b_eff / m_total * v_t_mag)
            t_dir = v_t / v_t_mag
            body_a.v[node_a] -= (j_t / m_a) * t_dir
            for i, ni in enumerate(face_nodes_b):
                body_b.v[ni] += (j_t * bary[i] / max(m_b_nodes[i], 1e-10)) * t_dir

        n_resolved += 1

    return n_resolved


def resolve_rbd_fem_contacts(
    contacts: list[RBDFEMContactPoint],
    robot,
    fem_bodies: list,
    restitution: float = 0.1,
    friction_mu: float = 0.5,
) -> int:
    """Apply impulse-based contact response for RBD-FEM collisions.

    The RBD is treated as infinite mass (already stepped). The FEM node
    absorbs the full correction.

    For **true penetrations** (gap == 0): full position correction +
    velocity impulse + Coulomb friction.
    For **proximity contacts** (gap > 0): drag-based friction that
    matches the node's tangential velocity to the RBD surface.
    Closer proximity → stronger coupling.  This enables grasping
    and lifting of FEM bodies by RBD grippers.

    Parameters
    ----------
    contacts : detected RBD-FEM contacts
    robot : Robot model (for velocity computation)
    fem_bodies : list of DeformableBody
    restitution, friction_mu : contact parameters

    Returns number of contacts resolved.
    """
    from robosim.physics.contact.solver import _link_point_velocity

    n_resolved = 0
    fk = robot.forward_kinematics()

    for c in contacts:
        body = fem_bodies[c.fem_body_idx]
        node_idx = c.fem_node_idx
        normal = c.normal

        if c.penetration <= 0:
            continue

        v_fem = body.v[node_idx]
        v_rbd = _link_point_velocity(robot, c.link_idx, c.rbd_contact_point,
                                      fk=fk)
        v_rel = v_fem - v_rbd
        v_n = float(np.dot(v_rel, normal))

        if c.gap <= 0:
            # ── True penetration: standard impulse response ──
            body.x[node_idx] += c.penetration * normal

            if v_n >= 0:
                n_resolved += 1
                continue

            # Normal impulse: reflect
            dv_n = -(1.0 + restitution) * v_n
            body.v[node_idx] += dv_n * normal

            # Coulomb friction
            v_t = v_rel - v_n * normal
            v_t_mag = np.linalg.norm(v_t)
            if v_t_mag > 1e-10:
                max_dv_t = friction_mu * (1.0 + restitution) * abs(v_n)
                scale = min(1.0, max_dv_t / v_t_mag)
                body.v[node_idx] -= scale * v_t
        else:
            # ── Proximity contact: kinematic coupling ──
            # Strength ramps from 0 at d_hat distance to 1 at surface.
            alpha = c.penetration / (c.penetration + c.gap)

            # Prevent approach (normal direction)
            if v_n < 0:
                body.v[node_idx] -= v_n * normal

            # Kinematic coupling: directly blend node velocity toward
            # the RBD surface velocity.  This acts as a "grab"
            # constraint that enables grasping and lifting.
            body.v[node_idx] = (1.0 - alpha) * body.v[node_idx] + alpha * v_rbd

        n_resolved += 1

    return n_resolved
