"""Finite element mesh data structures.

Supports:
  - TetMesh  : Tet4 (backward compatible)
  - FEMesh   : General mesh (Tet4, Tet10, Hex8)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robosim.physics.fem.elements import ElementType, NODES_PER_ELEMENT


# ═══════════════════════════════════════════════════════════════
# General FE Mesh
# ═══════════════════════════════════════════════════════════════

@dataclass
class FEMesh:
    """General finite element mesh supporting multiple element types.

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions
    elements : (n_elements, nodes_per_elem) connectivity
    element_type : ElementType enum
    """

    nodes: np.ndarray
    elements: np.ndarray
    element_type: ElementType = ElementType.TET4

    _surface_faces: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def nodes_per_element(self) -> int:
        return NODES_PER_ELEMENT[self.element_type]

    def extract_surface(self) -> np.ndarray:
        """Extract boundary triangle faces for visualization."""
        if self._surface_faces is not None:
            return self._surface_faces

        if self.element_type in (ElementType.TET4, ElementType.TET10):
            self._surface_faces = _extract_surface_tet(self.elements, self.element_type)
        elif self.element_type == ElementType.HEX8:
            self._surface_faces = _extract_surface_hex(self.elements)
        return self._surface_faces

    # ── Factory methods ──

    @staticmethod
    def create_hex_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (3, 3, 3),
    ) -> FEMesh:
        """Create a Hex8 box mesh.

        Parameters
        ----------
        origin : lower corner
        size : extent in x, y, z
        divisions : cells per direction
        """
        nx, ny, nz = divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        xs = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        ys = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        zs = np.linspace(origin[2], origin[2] + size[2], nz + 1)

        nodes = []
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nodes.append([xs[ix], ys[iy], zs[iz]])
        nodes = np.array(nodes, dtype=np.float64)

        def idx(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        elements = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # 8 corners in standard Hex8 ordering:
                    #   0-3: bottom face (z), 4-7: top face (z+1)
                    #   counterclockwise when viewed from outside
                    v = [
                        idx(ix,     iy,     iz),
                        idx(ix + 1, iy,     iz),
                        idx(ix + 1, iy + 1, iz),
                        idx(ix,     iy + 1, iz),
                        idx(ix,     iy,     iz + 1),
                        idx(ix + 1, iy,     iz + 1),
                        idx(ix + 1, iy + 1, iz + 1),
                        idx(ix,     iy + 1, iz + 1),
                    ]
                    elements.append(v)

        elements = np.array(elements, dtype=np.int64)
        return FEMesh(nodes=nodes, elements=elements, element_type=ElementType.HEX8)

    @staticmethod
    def create_tet10_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (2, 2, 2),
    ) -> FEMesh:
        """Create a Tet10 box mesh.

        First creates a Tet4 mesh, then adds edge midpoint nodes.
        """
        # Start with Tet4 mesh
        tet4 = TetMesh.create_box(origin=origin, size=size, divisions=divisions)
        nodes = list(tet4.nodes)
        n_orig = len(nodes)

        # Map edge -> midpoint node index
        edge_to_mid: dict[tuple[int, int], int] = {}

        def get_midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in edge_to_mid:
                mid = 0.5 * (np.array(nodes[a]) + np.array(nodes[b]))
                edge_to_mid[key] = len(nodes)
                nodes.append(mid)
            return edge_to_mid[key]

        # Tet4 edges: (0,1), (1,2), (0,2), (0,3), (1,3), (2,3)
        elements10 = []
        for e in range(tet4.n_elements):
            n0, n1, n2, n3 = tet4.elements[e]
            m01 = get_midpoint(n0, n1)  # node 4
            m12 = get_midpoint(n1, n2)  # node 5
            m02 = get_midpoint(n0, n2)  # node 6
            m03 = get_midpoint(n0, n3)  # node 7
            m13 = get_midpoint(n1, n3)  # node 8
            m23 = get_midpoint(n2, n3)  # node 9
            elements10.append([n0, n1, n2, n3, m01, m12, m02, m03, m13, m23])

        nodes = np.array(nodes, dtype=np.float64)
        elements10 = np.array(elements10, dtype=np.int64)
        return FEMesh(nodes=nodes, elements=elements10, element_type=ElementType.TET10)

    def __repr__(self) -> str:
        return f"FEMesh({self.element_type.value}, nodes={self.n_nodes}, elements={self.n_elements})"


# ═══════════════════════════════════════════════════════════════
# Surface extraction helpers
# ═══════════════════════════════════════════════════════════════

def _extract_surface_tet(elements, etype):
    """Extract surface triangles from tet mesh (Tet4 or Tet10)."""
    face_count: dict[tuple, int] = {}
    face_to_orig: dict[tuple, tuple] = {}

    for e in range(elements.shape[0]):
        n = elements[e]
        # 4 faces, using corner nodes only (first 4)
        faces = [
            (n[0], n[1], n[2]),
            (n[0], n[1], n[3]),
            (n[0], n[2], n[3]),
            (n[1], n[2], n[3]),
        ]
        for f in faces:
            key = tuple(sorted(f))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_to_orig:
                face_to_orig[key] = f

    surface = [face_to_orig[k] for k, c in face_count.items() if c == 1]
    return np.array(surface, dtype=np.int64) if surface else np.zeros((0, 3), dtype=np.int64)


def _extract_surface_hex(elements):
    """Extract surface triangles from Hex8 mesh (each quad → 2 triangles)."""
    # 6 faces of a hex, each defined by 4 corner indices (local)
    hex_faces = [
        (0, 3, 2, 1),  # bottom (-z)
        (4, 5, 6, 7),  # top (+z)
        (0, 1, 5, 4),  # front (-y)
        (2, 3, 7, 6),  # back (+y)
        (0, 4, 7, 3),  # left (-x)
        (1, 2, 6, 5),  # right (+x)
    ]

    face_count: dict[tuple, int] = {}
    face_to_quad: dict[tuple, tuple] = {}

    for e in range(elements.shape[0]):
        n = elements[e]
        for lf in hex_faces:
            quad = tuple(n[i] for i in lf)
            key = tuple(sorted(quad))
            face_count[key] = face_count.get(key, 0) + 1
            if key not in face_to_quad:
                face_to_quad[key] = quad

    triangles = []
    for key, count in face_count.items():
        if count == 1:
            q = face_to_quad[key]
            triangles.append((q[0], q[1], q[2]))
            triangles.append((q[0], q[2], q[3]))

    return np.array(triangles, dtype=np.int64) if triangles else np.zeros((0, 3), dtype=np.int64)


# ═══════════════════════════════════════════════════════════════
# TetMesh (backward compatible)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TetMesh:
    """Linear tetrahedral (Tet4) mesh — backward compatible API.

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions
    elements : (n_elements, 4) node indices per tetrahedron
    """

    nodes: np.ndarray
    elements: np.ndarray

    _surface_faces: np.ndarray | None = field(default=None, repr=False)
    _volumes: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self.nodes = np.asarray(self.nodes, dtype=np.float64)
        self.elements = np.asarray(self.elements, dtype=np.int64)

    @property
    def element_type(self) -> ElementType:
        return ElementType.TET4

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    @property
    def n_dof(self) -> int:
        return self.n_nodes * 3

    @property
    def nodes_per_element(self) -> int:
        return 4

    def compute_volumes(self) -> np.ndarray:
        if self._volumes is not None:
            return self._volumes
        vols = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            n0, n1, n2, n3 = self.elements[e]
            d1 = self.nodes[n1] - self.nodes[n0]
            d2 = self.nodes[n2] - self.nodes[n0]
            d3 = self.nodes[n3] - self.nodes[n0]
            vols[e] = abs(np.dot(d1, np.cross(d2, d3))) / 6.0
        self._volumes = vols
        return vols

    def total_volume(self) -> float:
        return float(self.compute_volumes().sum())

    def extract_surface(self) -> np.ndarray:
        if self._surface_faces is not None:
            return self._surface_faces
        self._surface_faces = _extract_surface_tet(self.elements, ElementType.TET4)
        return self._surface_faces

    @staticmethod
    def create_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (3, 3, 3),
    ) -> TetMesh:
        """Create a box-shaped tet mesh (5 tets per hex cell)."""
        nx, ny, nz = divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        xs = np.linspace(origin[0], origin[0] + size[0], nx + 1)
        ys = np.linspace(origin[1], origin[1] + size[1], ny + 1)
        zs = np.linspace(origin[2], origin[2] + size[2], nz + 1)

        nodes = []
        for iz in range(nz + 1):
            for iy in range(ny + 1):
                for ix in range(nx + 1):
                    nodes.append([xs[ix], ys[iy], zs[iz]])
        nodes = np.array(nodes, dtype=np.float64)

        def node_idx(ix, iy, iz):
            return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

        elements = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    v = [
                        node_idx(ix, iy, iz), node_idx(ix+1, iy, iz),
                        node_idx(ix+1, iy+1, iz), node_idx(ix, iy+1, iz),
                        node_idx(ix, iy, iz+1), node_idx(ix+1, iy, iz+1),
                        node_idx(ix+1, iy+1, iz+1), node_idx(ix, iy+1, iz+1),
                    ]
                    parity = (ix + iy + iz) % 2
                    if parity == 0:
                        elements.append([v[0], v[1], v[3], v[4]])
                        elements.append([v[1], v[2], v[3], v[6]])
                        elements.append([v[1], v[4], v[5], v[6]])
                        elements.append([v[3], v[4], v[6], v[7]])
                        elements.append([v[1], v[3], v[4], v[6]])
                    else:
                        elements.append([v[0], v[1], v[2], v[5]])
                        elements.append([v[0], v[2], v[3], v[7]])
                        elements.append([v[0], v[4], v[5], v[7]])
                        elements.append([v[2], v[5], v[6], v[7]])
                        elements.append([v[0], v[2], v[5], v[7]])

        elements = np.array(elements, dtype=np.int64)
        return TetMesh(nodes=nodes, elements=elements)

    def __repr__(self) -> str:
        return f"TetMesh(nodes={self.n_nodes}, elements={self.n_elements})"
