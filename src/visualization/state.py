"""``VisualizationState`` – the single interface between physics and the viewer.

Every producer in the project (artificial demo, FEM solver, ROM, inverse force
solver, vision pipeline, dataset inspector) fills a :class:`VisualizationState`
and hands it to :class:`~src.visualization.taichi_viewer.TaichiViewer`. The
viewer never imports solver code, and solvers never import Taichi.

Conventions
-----------
* All quantities are *physical* and in SI units (m, N). Visualization-only
  parameters (deformation amplification, arrow scale, colors) live in the
  viewer's :class:`~src.visualization.taichi_viewer.ViewOptions`, never here.
* ``displacement`` is the nodal displacement ``u`` with shape (N, 3) and
  DOF ordering ``[u_x, u_y, u_z]`` per node (the flat FEM vector is
  ``u.reshape(-1)`` -> ``3*i + comp``).
* ``nodes_deformed = nodes_original + displacement`` (alpha = 1).
* Forces are the forces *acting on the finger* at their ``origin``
  (a contact pushing down on the top surface has a ``-z`` vector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..geometry import primitives
from .camera import ObservationCamera


@dataclass
class ForceVector:
    """A single force arrow. ``vector`` is in Newtons, ``origin`` in meters."""

    origin: np.ndarray
    vector: np.ndarray
    label: str = "force"
    color: tuple[float, float, float] | None = None  # None -> viewer default for its role

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.vector = np.asarray(self.vector, dtype=float).reshape(3)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def direction(self) -> np.ndarray:
        m = self.magnitude
        return self.vector / m if m > 0 else np.zeros(3)


@dataclass
class ContactPoint:
    """Point contact on the finger surface (extended to patches later)."""

    position: np.ndarray
    node_index: int | None = None
    normal: np.ndarray | None = None  # outward surface normal at the contact
    label: str = "contact"

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float).reshape(3)
        if self.normal is not None:
            self.normal = np.asarray(self.normal, dtype=float).reshape(3)


@dataclass
class ObjectGeometry:
    """Display-only triangle mesh of an object the finger interacts with.

    ``attach_to`` (optional, world point in meters): when set, the viewer
    translates the object by the *amplified* displacement of the finger node
    nearest to that point, so an object resting on the contact stays on the
    visually deformed surface. Purely cosmetic; physics never sees it.
    """

    vertices: np.ndarray
    faces: np.ndarray
    name: str = "object"
    color: tuple[float, float, float] | None = None
    attach_to: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.attach_to is not None:
            self.attach_to = np.asarray(self.attach_to, dtype=float).reshape(3)

    @classmethod
    def sphere(cls, center, radius: float, name: str = "sphere", color=None, attach_to=None, n_lat: int = 16, n_lon: int = 24):
        v, f = primitives.sphere_mesh(center, radius, n_lat, n_lon)
        return cls(v, f, name=name, color=color, attach_to=attach_to)

    @classmethod
    def box(cls, center, size, name: str = "box", color=None, attach_to=None):
        v, f = primitives.box_mesh(center, size)
        return cls(v, f, name=name, color=color, attach_to=attach_to)


@dataclass
class VisualizationState:
    """Everything the Taichi viewer needs to draw one mechanical configuration.

    Required
    --------
    nodes_original : (N, 3) undeformed node coordinates [m]
    surface_faces : (F, 3) outward-oriented boundary triangles

    Optional
    --------
    element_edges : (E, 2) element corner edges (internal wireframe)
    surface_edges : (Es, 2) wireframe edges of the boundary; None -> derived
        from ``surface_faces``
    tetrahedra : (M, k) element connectivity – legacy input; converted to
        ``element_edges`` when the latter is not given
    displacement : (N, 3) nodal displacement u [m]; None -> zero
    fixed_nodes : (K,) indices of fully clamped nodes
    contact_points : point contacts (markers)
    ground_truth_force / estimated_force : force arrows anchored at contacts
    extra_forces : any additional arrows (e.g. reaction forces)
    objects : display meshes (sphere, box, arbitrary)
    observation_camera : research camera drawn as marker + frustum
    node_scalar : (N,) optional scalar (stress, pressure, ...) used for
        coloring when the viewer's color mode is ``"scalar"``
    node_scalar_name : label for the colorbar / GUI
    displacement_alt : (N, 3) optional second displacement field (e.g. ROM
        reconstruction) that the viewer can toggle against ``displacement``
    displacement_alt_name : label
    info : ordered text lines shown in the GUI panel (sample id, F_true, ...)
    """

    nodes_original: np.ndarray
    surface_faces: np.ndarray
    element_edges: np.ndarray | None = None
    surface_edges: np.ndarray | None = None
    tetrahedra: np.ndarray | None = None
    displacement: np.ndarray | None = None
    fixed_nodes: np.ndarray | None = None
    contact_points: list[ContactPoint] = field(default_factory=list)
    ground_truth_force: ForceVector | None = None
    estimated_force: ForceVector | None = None
    extra_forces: list[ForceVector] = field(default_factory=list)
    objects: list[ObjectGeometry] = field(default_factory=list)
    observation_camera: ObservationCamera | None = None
    node_scalar: np.ndarray | None = None
    node_scalar_name: str = "scalar"
    displacement_alt: np.ndarray | None = None
    displacement_alt_name: str = "alt"
    keypoints: np.ndarray | None = None  # (M, 3) surface markers in the *undeformed* configuration [m]
    keypoints_visible: np.ndarray | None = None  # (M,) bool – seen by the observation camera
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.keypoints is not None:
            self.keypoints = np.ascontiguousarray(self.keypoints, dtype=np.float64).reshape(-1, 3)
            if self.keypoints_visible is None:
                self.keypoints_visible = np.ones(len(self.keypoints), dtype=bool)
            self.keypoints_visible = np.ascontiguousarray(self.keypoints_visible, dtype=bool).reshape(-1)
            if len(self.keypoints_visible) != len(self.keypoints):
                raise ValueError("keypoints_visible must match keypoints")
        self.nodes_original = np.ascontiguousarray(self.nodes_original, dtype=np.float64).reshape(-1, 3)
        self.surface_faces = np.ascontiguousarray(self.surface_faces, dtype=np.int64).reshape(-1, 3)
        if self.tetrahedra is not None:
            self.tetrahedra = np.ascontiguousarray(self.tetrahedra, dtype=np.int64)
            if self.tetrahedra.ndim != 2:
                raise ValueError("tetrahedra must be (M, k)")
            if self.element_edges is None and self.tetrahedra.size:
                from ..geometry.mesh_utils import unique_edges

                self.element_edges = unique_edges(self.tetrahedra)
        if self.element_edges is not None:
            self.element_edges = np.ascontiguousarray(self.element_edges, dtype=np.int64).reshape(-1, 2)
        if self.surface_edges is not None:
            self.surface_edges = np.ascontiguousarray(self.surface_edges, dtype=np.int64).reshape(-1, 2)
        if self.displacement is not None:
            self.displacement = np.ascontiguousarray(self.displacement, dtype=np.float64).reshape(-1, 3)
        if self.displacement_alt is not None:
            self.displacement_alt = np.ascontiguousarray(self.displacement_alt, dtype=np.float64).reshape(-1, 3)
        if self.fixed_nodes is not None:
            self.fixed_nodes = np.ascontiguousarray(self.fixed_nodes, dtype=np.int64).reshape(-1)
        if self.node_scalar is not None:
            self.node_scalar = np.ascontiguousarray(self.node_scalar, dtype=np.float64).reshape(-1)
        self.validate()

    # ------------------------------------------------------------ checks
    def validate(self) -> None:
        n = self.n_nodes
        if self.surface_faces.size and self.surface_faces.max() >= n:
            raise ValueError("surface_faces reference non-existent nodes")
        for name in ("tetrahedra", "element_edges", "surface_edges"):
            arr = getattr(self, name)
            if arr is not None and arr.size and arr.max() >= n:
                raise ValueError(f"{name} reference non-existent nodes")
        for name in ("displacement", "displacement_alt"):
            arr = getattr(self, name)
            if arr is not None and arr.shape != (n, 3):
                raise ValueError(f"{name} must have shape ({n}, 3), got {arr.shape}")
        if self.fixed_nodes is not None and self.fixed_nodes.size and self.fixed_nodes.max() >= n:
            raise ValueError("fixed_nodes reference non-existent nodes")
        if self.node_scalar is not None and self.node_scalar.shape != (n,):
            raise ValueError(f"node_scalar must have shape ({n},)")

    # ------------------------------------------------------------ derived
    @property
    def n_nodes(self) -> int:
        return int(self.nodes_original.shape[0])

    @property
    def nodes_deformed(self) -> np.ndarray:
        """Physical deformed coordinates (alpha = 1)."""
        return self.displaced(1.0)

    def displaced(self, amplification: float = 1.0, use_alt: bool = False) -> np.ndarray:
        """``x_display = x_original + alpha * u``. Visualization only."""
        u = self.displacement_alt if use_alt else self.displacement
        if u is None:
            return self.nodes_original.copy()
        return self.nodes_original + float(amplification) * u

    def displacement_magnitude(self, use_alt: bool = False) -> np.ndarray:
        u = self.displacement_alt if use_alt else self.displacement
        if u is None:
            return np.zeros(self.n_nodes)
        return np.linalg.norm(u, axis=1)

    def max_displacement(self) -> float:
        return float(self.displacement_magnitude().max()) if self.displacement is not None else 0.0

    def all_forces(self) -> list[tuple[str, ForceVector]]:
        """``[(role, force), ...]`` with role in {"gt", "est", "extra"}."""
        out: list[tuple[str, ForceVector]] = []
        if self.ground_truth_force is not None:
            out.append(("gt", self.ground_truth_force))
        if self.estimated_force is not None:
            out.append(("est", self.estimated_force))
        out.extend(("extra", f) for f in self.extra_forces)
        return out

    def scene_bounds(self, include_objects: bool = True, include_camera: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Bounding box of the finger (undeformed + deformed) and optionally objects."""
        pts = [self.nodes_original, self.nodes_deformed]
        if include_objects:
            pts += [o.vertices for o in self.objects if len(o.vertices)]
        if include_camera and self.observation_camera is not None:
            pts.append(self.observation_camera.position[None, :])
        allp = np.concatenate(pts, axis=0)
        return allp.min(axis=0), allp.max(axis=0)

    def finger_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.nodes_original.min(axis=0), self.nodes_original.max(axis=0)

    def topology_key(self) -> tuple:
        """Hashable summary used by the viewer to decide whether GPU buffers
        must be re-allocated (topology changed) or merely updated."""
        n_ee = 0 if self.element_edges is None else int(self.element_edges.shape[0])
        n_se = -1 if self.surface_edges is None else int(self.surface_edges.shape[0])
        return (self.n_nodes, int(self.surface_faces.shape[0]), n_ee, n_se)
