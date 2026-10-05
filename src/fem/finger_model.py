"""High-level "compliant finger" FEM model assembled from ``configs/fem.yaml``.

Bundles geometry, mesh, material, clamped-root boundary conditions, the
factorised :class:`LinearFEM` and the point-contact operator so that demos,
dataset generation and inverse solvers share one construction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contact.contact_mapping import PointContactMapping, SurfaceContact
from ..geometry.finger import (
    FingerGeometry,
    TetMesh,
    default_element_size,
    make_rectangular_finger_mesh,
    resolution_from_element_size,
    root_fixed_nodes,
)
from .boundary import DofPartition, clamp_nodes
from .material import LinearElasticMaterial
from .solver import FEMResult, LinearFEM


@dataclass
class FingerFEMModel:
    geometry: FingerGeometry
    mesh: TetMesh
    material: LinearElasticMaterial
    partition: DofPartition
    fem: LinearFEM
    contact: PointContactMapping
    config: dict[str, Any]

    # ------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], nx: int | None = None, ny: int | None = None, nz: int | None = None) -> "FingerFEMModel":
        """Build the model. Mesh resolution: ``mesh.element_size`` [m] (default
        smallest dimension / 4) sets all three axes; ``mesh.nx/ny/nz`` in the
        config or the keyword arguments override individual axes."""
        geom = FingerGeometry.from_config(cfg)
        m = cfg.get("mesh", {}) or {}
        dx, dy, dz = resolution_from_element_size(geom, float(m.get("element_size", default_element_size(geom))))
        mesh = make_rectangular_finger_mesh(
            geom, nx or int(m.get("nx", dx)), ny or int(m.get("ny", dy)), nz or int(m.get("nz", dz)),
            tet_fraction=float(m.get("tet_fraction", 0.2)),
        )
        mat = LinearElasticMaterial.from_config(cfg)
        bc = cfg.get("boundary_conditions", {})
        if bc.get("fixed_surface", "root") != "root":
            raise NotImplementedError("only the clamped root surface is supported so far")
        part = clamp_nodes(mesh.n_nodes, root_fixed_nodes(mesh, float(bc.get("tolerance", 1e-9))))
        fem = LinearFEM(mesh, mat, part)
        fem.factorize()
        # Contact candidates: the whole boundary except the clamped root face.
        faces = mesh.surface_faces
        not_root = ~np.all(np.abs(mesh.nodes[faces][:, :, 0]) < 1e-9, axis=1)
        contact = PointContactMapping(mesh, np.nonzero(not_root)[0])
        model = cls(geom, mesh, mat, part, fem, contact, dict(cfg))
        gcfg = cfg.get("gravity") or {}
        if bool(gcfg.get("enabled", False)):
            model.gravity_world = np.asarray(gcfg.get("vector", [0.0, 0.0, -9.80665]), float).reshape(3)
        return model

    # ------------------------------------------------------------ self-weight
    @property
    def gravity_world(self) -> np.ndarray | None:
        """World-frame gravitational acceleration [m/s²] or ``None`` (self-weight off)."""
        return getattr(self, "_gravity_world", None)

    @gravity_world.setter
    def gravity_world(self, g) -> None:
        self._gravity_world = None if g is None else np.asarray(g, float).reshape(3)
        self._gravity_cache: dict[tuple, np.ndarray] = {}

    @property
    def has_gravity(self) -> bool:
        return self.gravity_world is not None and float(self.material.density) > 0.0

    def gravity_local(self, R_world_from_local=None) -> np.ndarray:
        """Gravity expressed in the finger frame for a finger rotated by ``R`` (local → world)."""
        g = np.zeros(3) if self.gravity_world is None else self.gravity_world
        if R_world_from_local is None:
            return g.copy()
        return np.asarray(R_world_from_local, float).reshape(3, 3).T @ g

    def gravity_displacement(self, R_world_from_local=None) -> np.ndarray:
        """(N, 3) self-weight sag ``K⁻¹ f_g`` for the finger orientation ``R`` (cached per R).

        Zero when self-weight is disabled. Linear superposition: the total field
        under a contact load is ``u_contact + gravity_displacement(R)``.
        """
        if not self.has_gravity:
            return np.zeros((self.mesh.n_nodes, 3))
        g_loc = self.gravity_local(R_world_from_local)
        key = tuple(np.round(g_loc, 9))
        cache = getattr(self, "_gravity_cache", None)
        if cache is None:
            self._gravity_cache = cache = {}
        if key not in cache:
            cache[key] = self.fem.solve(self.fem.body_force_vector(g_loc), meta={"load": "self-weight"}).u.copy()
        return cache[key]

    def solve_gravity(self, R_world_from_local=None) -> FEMResult:
        """Full FEM result for self-weight alone (for plots / verification)."""
        return self.fem.solve(self.fem.body_force_vector(self.gravity_local(R_world_from_local)), meta={"load": "self-weight"})

    def surface_face_ids(self, surface: str) -> np.ndarray:
        """Indices (into ``mesh.surface_faces``) of a named finger face.

        ``top`` (z=H), ``bottom`` (z=0), ``side_pos_y`` (y=W), ``side_neg_y`` (y=0), ``tip`` (x=L).
        """
        g, f = self.geometry, self.mesh.surface_faces
        tri = self.mesh.nodes[f]
        planes = {
            "top": (2, g.height), "bottom": (2, 0.0),
            "side_pos_y": (1, g.width), "side_neg_y": (1, 0.0),
            "tip": (0, g.length),
        }
        if surface not in planes:
            raise KeyError(f"unknown surface {surface!r}; choose from {sorted(planes)}")
        axis, val = planes[surface]
        return np.nonzero(np.all(np.abs(tri[:, :, axis] - val) < 1e-9, axis=1))[0]

    def restrict_contact_surface(self, surface: str | None) -> None:
        """Limit contact candidates to one named face (``None`` = all but the root)."""
        if surface is None:
            faces = self.mesh.surface_faces
            ids = np.nonzero(~np.all(np.abs(self.mesh.nodes[faces][:, :, 0]) < 1e-9, axis=1))[0]
        else:
            ids = self.surface_face_ids(surface)
        self.contact = PointContactMapping(self.mesh, ids)

    # ------------------------------------------------------------ solve
    def solve_point_force(self, point, force_vector, meta: dict | None = None) -> tuple[FEMResult, SurfaceContact]:
        """Apply ``force_vector`` [N] at the surface point closest to ``point``."""
        f, c = self.contact.load_vector(point, force_vector)
        res = self.fem.solve(f, meta={"contact_position": c.position.tolist(), "force_vector": list(map(float, force_vector)), **(meta or {})})
        return res, c

    def solve_normal_contact(self, point, magnitude: float, meta: dict | None = None) -> tuple[FEMResult, SurfaceContact]:
        """Pressing normal contact (force along -n) of ``magnitude`` [N]."""
        f, c = self.contact.normal_load_vector(point, magnitude)
        res = self.fem.solve(f, meta={"contact_position": c.position.tolist(), "force_magnitude": float(magnitude), **(meta or {})})
        return res, c

    def default_contact_point(self) -> np.ndarray:
        rel = self.config.get("loading", {}).get("contact_position_rel", [0.9, 0.5, 1.0])
        return self.geometry.point_from_relative(rel)

    # ------------------------------------------------------------ viz
    def visualization_state(
        self,
        result: FEMResult,
        contact: SurfaceContact,
        force_vector,
        estimated_force=None,
        with_object: bool = True,
        with_stress: bool = True,
        observation_camera=None,
        info: dict | None = None,
    ):
        from ..visualization.state import ContactPoint, ForceVector, ObjectGeometry

        force_vector = np.asarray(force_vector, float)
        objects = []
        if with_object:
            r = 0.4 * self.geometry.height
            objects.append(ObjectGeometry.sphere(contact.position + r * contact.normal, r, name="probe", attach_to=contact.position))
        scalar = self.fem.stresses(result)["von_mises_nodal"] if with_stress else None
        est = None
        if estimated_force is not None:
            est = ForceVector(contact.position, np.asarray(estimated_force, float), label="F_est")
        return result.to_visualization_state(
            contact_points=[ContactPoint(contact.position, node_index=contact.nearest_node, normal=contact.normal)],
            ground_truth_force=ForceVector(contact.position, force_vector, label="F_true"),
            estimated_force=est,
            objects=objects,
            observation_camera=observation_camera,
            node_scalar=scalar,
            node_scalar_name="von Mises [Pa]",
            info=info,
        )
