"""Tetrahedral mesh data structure for FEM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TetMesh:
    """Linear tetrahedral (Tet4) mesh.

    Attributes
    ----------
    nodes : (n_nodes, 3) rest positions
    elements : (n_elements, 4) node indices per tetrahedron
    """

    nodes: np.ndarray
    elements: np.ndarray

    # Computed on build
    _surface_faces: np.ndarray | None = field(default=None, repr=False)
    _volumes: np.ndarray | None = field(default=None, repr=False)

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

    def compute_volumes(self) -> np.ndarray:
        """Compute volume of each tetrahedron. Caches result."""
        if self._volumes is not None:
            return self._volumes

        vols = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            n0, n1, n2, n3 = self.elements[e]
            x0, x1, x2, x3 = self.nodes[n0], self.nodes[n1], self.nodes[n2], self.nodes[n3]
            d1 = x1 - x0
            d2 = x2 - x0
            d3 = x3 - x0
            vols[e] = abs(np.dot(d1, np.cross(d2, d3))) / 6.0
        self._volumes = vols
        return vols

    def total_volume(self) -> float:
        return float(self.compute_volumes().sum())

    def extract_surface(self) -> np.ndarray:
        """Extract boundary triangle faces.

        Returns (n_faces, 3) array of node indices for surface triangles.
        A face is on the surface if it belongs to exactly one tetrahedron.
        """
        if self._surface_faces is not None:
            return self._surface_faces

        face_count: dict[tuple[int, ...], int] = {}
        face_to_sorted: dict[tuple[int, ...], tuple[int, int, int]] = {}

        for e in range(self.n_elements):
            n = self.elements[e]
            # 4 faces of a tet, each defined by 3 nodes
            faces = [
                (n[0], n[1], n[2]),
                (n[0], n[1], n[3]),
                (n[0], n[2], n[3]),
                (n[1], n[2], n[3]),
            ]
            for f in faces:
                key = tuple(sorted(f))
                face_count[key] = face_count.get(key, 0) + 1
                if key not in face_to_sorted:
                    face_to_sorted[key] = f

        surface = [face_to_sorted[k] for k, count in face_count.items() if count == 1]
        self._surface_faces = np.array(surface, dtype=np.int64)
        return self._surface_faces

    @staticmethod
    def create_box(
        origin: np.ndarray = np.zeros(3),
        size: np.ndarray = np.ones(3),
        divisions: tuple[int, int, int] = (3, 3, 3),
    ) -> TetMesh:
        """Create a box-shaped tet mesh by subdividing a hexahedral grid.

        Each hex cell is split into 5 tetrahedra.

        Parameters
        ----------
        origin : (3,) lower corner
        size : (3,) extent in x, y, z
        divisions : number of cells in each direction
        """
        nx, ny, nz = divisions
        origin = np.asarray(origin, dtype=np.float64)
        size = np.asarray(size, dtype=np.float64)

        # Generate nodes on a regular grid
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

        # Split each hex cell into 5 tets
        elements = []
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    # 8 corners of the hex cell
                    v = [
                        node_idx(ix,     iy,     iz),      # 0
                        node_idx(ix + 1, iy,     iz),      # 1
                        node_idx(ix + 1, iy + 1, iz),      # 2
                        node_idx(ix,     iy + 1, iz),      # 3
                        node_idx(ix,     iy,     iz + 1),   # 4
                        node_idx(ix + 1, iy,     iz + 1),   # 5
                        node_idx(ix + 1, iy + 1, iz + 1),   # 6
                        node_idx(ix,     iy + 1, iz + 1),   # 7
                    ]

                    # 5-tet decomposition (consistent diagonal)
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
