"""Quadratic solid elements for 3D linear elasticity: 10-node tetrahedron (T10)
and 20-node serendipity hexahedron (H20), integrated with Gauss quadrature.

Node ordering (VTK / Gmsh convention)
-------------------------------------
T10 : corners 0..3, mid-edge 4:(0,1) 5:(1,2) 6:(0,2) 7:(0,3) 8:(1,3) 9:(2,3)
      natural coordinates (ξ, η, ζ), L0 = 1-ξ-η-ζ, L1 = ξ, L2 = η, L3 = ζ
      N_i = L_i (2 L_i - 1)  (corners),   N_ab = 4 L_a L_b  (edges)
H20 : corners 0..7 at (±1, ±1, ±1) — 0:(-,-,-) 1:(+,-,-) 2:(+,+,-) 3:(-,+,-)
      4:(-,-,+) 5:(+,-,+) 6:(+,+,+) 7:(-,+,+); mid-edge 8..19 for the edges
      (0,1) (1,2) (2,3) (3,0) (4,5) (5,6) (6,7) (7,4) (0,4) (1,5) (2,6) (3,7)
      N_c = 1/8 (1+ξξ_c)(1+ηη_c)(1+ζζ_c)(ξξ_c+ηη_c+ζζ_c-2)
      N_m = 1/4 (1-ξ²)(1+ηη_m)(1+ζζ_m)   (ξ_m = 0; cyclic for the other axes)

Quadrature: T10 4-point degree-2 rule (exact for the stiffness of straight-
sided tets), H20 3×3×3 Gauss (exact for affine hexes).

Element DOF ordering: ``3 * local_node + component``; Voigt strain
[ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz] as in :mod:`src.fem.tetra_element`.

All routines are vectorised over elements. ``element_stiffness`` returns the
stiffness ``K_e (M, 3n, 3n)``, the volume-averaged strain-displacement matrix
``B̄ (M, 6, 3n)`` (for element-mean strain / stress recovery), the volumes
``V (M,)`` and the consistent lumped mass shares ``m (M, n)`` = ∫ N_i dV used
for body forces (Σ_i m_i = V).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ topology
T10_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]], dtype=np.int64)
T10_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int64)  # corner faces, outward when V>0

H20_EDGES = np.array(
    [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]], dtype=np.int64
)
# corner quads, outward (CCW seen from outside) for a right-handed corner ordering
H20_FACES = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], dtype=np.int64)

_H20_CORNER_XI = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
)


# ------------------------------------------------------------------ quadrature
def tet_quadrature() -> tuple[np.ndarray, np.ndarray]:
    """4-point degree-2 rule on the reference tet (volume 1/6)."""
    a, b = 0.5854101966249685, 0.1381966011250105
    pts = np.array([[a, b, b], [b, a, b], [b, b, a], [b, b, b]])
    w = np.full(4, 1.0 / 24.0)
    return pts, w


def hex_quadrature(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Tensor Gauss–Legendre rule with ``n`` points per axis on [-1, 1]³."""
    x, w = np.polynomial.legendre.leggauss(n)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    W = np.einsum("i,j,k->ijk", w, w, w)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), W.ravel()


# ------------------------------------------------------------------ shape functions
def t10_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """T10 shape functions ``N (G, 10)`` and derivatives ``dN (G, 10, 3)`` at points ``xi (G, 3)``."""
    xi = np.asarray(xi, float).reshape(-1, 3)
    L = np.stack([1.0 - xi.sum(axis=1), xi[:, 0], xi[:, 1], xi[:, 2]], axis=1)  # (G, 4)
    dL = np.array([[-1.0, -1.0, -1.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # (4, 3)
    G = xi.shape[0]
    N = np.empty((G, 10))
    dN = np.empty((G, 10, 3))
    for i in range(4):
        N[:, i] = L[:, i] * (2.0 * L[:, i] - 1.0)
        dN[:, i, :] = (4.0 * L[:, i] - 1.0)[:, None] * dL[i][None, :]
    for k, (a, b) in enumerate(T10_EDGES):
        N[:, 4 + k] = 4.0 * L[:, a] * L[:, b]
        dN[:, 4 + k, :] = 4.0 * (L[:, a][:, None] * dL[b][None, :] + L[:, b][:, None] * dL[a][None, :])
    return N, dN


def h20_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H20 serendipity shape functions ``N (G, 20)`` and derivatives ``dN (G, 20, 3)``."""
    xi = np.asarray(xi, float).reshape(-1, 3)
    G = xi.shape[0]
    x, y, z = xi[:, 0], xi[:, 1], xi[:, 2]
    N = np.empty((G, 20))
    dN = np.empty((G, 20, 3))
    for c in range(8):
        xc, yc, zc = _H20_CORNER_XI[c]
        a, b, d = 1.0 + xc * x, 1.0 + yc * y, 1.0 + zc * z
        s = xc * x + yc * y + zc * z - 2.0
        N[:, c] = 0.125 * a * b * d * s
        dN[:, c, 0] = 0.125 * xc * b * d * (s + a)
        dN[:, c, 1] = 0.125 * yc * a * d * (s + b)
        dN[:, c, 2] = 0.125 * zc * a * b * (s + d)
    for k, (p, q) in enumerate(H20_EDGES):
        m = 0.5 * (_H20_CORNER_XI[p] + _H20_CORNER_XI[q])  # one entry is 0
        xm, ym, zm = m
        if xm == 0.0:
            b, d = 1.0 + ym * y, 1.0 + zm * z
            N[:, 8 + k] = 0.25 * (1.0 - x * x) * b * d
            dN[:, 8 + k, 0] = -0.5 * x * b * d
            dN[:, 8 + k, 1] = 0.25 * (1.0 - x * x) * ym * d
            dN[:, 8 + k, 2] = 0.25 * (1.0 - x * x) * b * zm
        elif ym == 0.0:
            a, d = 1.0 + xm * x, 1.0 + zm * z
            N[:, 8 + k] = 0.25 * a * (1.0 - y * y) * d
            dN[:, 8 + k, 0] = 0.25 * xm * (1.0 - y * y) * d
            dN[:, 8 + k, 1] = -0.5 * y * a * d
            dN[:, 8 + k, 2] = 0.25 * a * (1.0 - y * y) * zm
        else:
            a, b = 1.0 + xm * x, 1.0 + ym * y
            N[:, 8 + k] = 0.25 * a * b * (1.0 - z * z)
            dN[:, 8 + k, 0] = 0.25 * xm * b * (1.0 - z * z)
            dN[:, 8 + k, 1] = 0.25 * a * ym * (1.0 - z * z)
            dN[:, 8 + k, 2] = -0.5 * z * a * b
    return N, dN


@dataclass(frozen=True)
class ElementType:
    name: str
    n_nodes: int
    edges: np.ndarray  # local corner pairs whose midpoints are the higher-order nodes (in node order)
    corner_faces: np.ndarray  # local corner faces (tri or quad), outward for positive orientation
    n_corners: int
    quad_points: np.ndarray
    quad_weights: np.ndarray
    shape: callable

    @property
    def n_dof(self) -> int:
        return 3 * self.n_nodes

    def evaluate(self) -> tuple[np.ndarray, np.ndarray]:
        """Shape functions / derivatives at the quadrature points."""
        return self.shape(self.quad_points)


TET10 = ElementType("tet10", 10, T10_EDGES, T10_FACES, 4, *tet_quadrature(), t10_shape)
HEX20 = ElementType("hex20", 20, H20_EDGES, H20_FACES, 8, *hex_quadrature(3), h20_shape)

ELEMENT_TYPES = {TET10.name: TET10, HEX20.name: HEX20}


def element_type_for(conn: np.ndarray) -> ElementType:
    n = int(np.asarray(conn).shape[1])
    if n == 10:
        return TET10
    if n == 20:
        return HEX20
    raise ValueError(f"unsupported element with {n} nodes (expected 10 or 20)")


# ------------------------------------------------------------------ kernels
def _strain_matrix(grads: np.ndarray) -> np.ndarray:
    """B (M, 6, 3n) from shape gradients (M, n, 3)."""
    M, n, _ = grads.shape
    B = np.zeros((M, 6, 3 * n))
    bx, by, bz = grads[:, :, 0], grads[:, :, 1], grads[:, :, 2]
    cols = 3 * np.arange(n)
    B[:, 0, cols + 0] = bx
    B[:, 1, cols + 1] = by
    B[:, 2, cols + 2] = bz
    B[:, 3, cols + 0] = by
    B[:, 3, cols + 1] = bx
    B[:, 4, cols + 1] = bz
    B[:, 4, cols + 2] = by
    B[:, 5, cols + 0] = bz
    B[:, 5, cols + 2] = bx
    return B


def element_stiffness(nodes: np.ndarray, conn: np.ndarray, C: np.ndarray):
    """Stiffness, mean strain matrix, volumes and nodal mass shares for one element block.

    Returns ``(K_e (M,3n,3n), B̄ (M,6,3n), V (M,), m (M,n))``.
    """
    et = element_type_for(conn)
    x = nodes[np.asarray(conn, dtype=np.int64)]  # (M, n, 3)
    M, n = x.shape[0], et.n_nodes
    N, dN = et.evaluate()  # (G, n), (G, n, 3)
    Ke = np.zeros((M, 3 * n, 3 * n))
    Bbar = np.zeros((M, 6, 3 * n))
    V = np.zeros(M)
    mass = np.zeros((M, n))
    for g, w in enumerate(et.quad_weights):
        J = np.einsum("mni,nj->mij", x, dN[g])  # J_ij = Σ_n x_ni dN_nj  -> dx/dξ (M,3,3)
        detJ = np.linalg.det(J)
        if np.any(detJ <= 0.0):
            raise ValueError(f"{et.name}: non-positive Jacobian (inverted or badly ordered element)")
        Jinv = np.linalg.inv(J)
        grads = np.einsum("nj,mji->mni", dN[g], Jinv)  # ∂N/∂x = dN/dξ · J⁻¹  (M, n, 3)
        B = _strain_matrix(grads)
        wdet = w * detJ
        Ke += np.einsum("mji,jk,mkl->mil", B, C, B) * wdet[:, None, None]
        Bbar += B * wdet[:, None, None]
        V += wdet
        mass += N[g][None, :] * wdet[:, None]
    Bbar /= V[:, None, None]
    return Ke, Bbar, V, mass


def element_dof_indices(conn: np.ndarray) -> np.ndarray:
    """(M, 3n) global DOF indices ``3 * node + component``."""
    conn = np.asarray(conn, dtype=np.int64)
    return (3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(len(conn), -1)


def element_volumes(nodes: np.ndarray, conn: np.ndarray) -> np.ndarray:
    """Volumes (M,) by quadrature (exact for straight-sided elements)."""
    et = element_type_for(conn)
    x = nodes[np.asarray(conn, dtype=np.int64)]
    _, dN = et.evaluate()
    V = np.zeros(x.shape[0])
    for g, w in enumerate(et.quad_weights):
        J = np.einsum("mni,nj->mij", x, dN[g])
        V += w * np.linalg.det(J)
    return V


def interpolate(conn_row: np.ndarray, et: ElementType, xi, u: np.ndarray) -> np.ndarray:
    """Field ``u (N,3)`` interpolated at natural coordinates ``xi`` of one element."""
    N, _ = et.shape(np.asarray(xi, float).reshape(1, 3))
    return N[0] @ u[np.asarray(conn_row, dtype=np.int64)]
