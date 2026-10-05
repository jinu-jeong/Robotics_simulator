"""Global stiffness assembly (SciPy sparse) for mixed H20 / T10 meshes.

Global DOF ordering: ``dof = 3 * node + component`` with component 0,1,2 =
x,y,z. The global matrix is ``K = Σ_e A_e^T K_e A_e`` and is assembled via a
vectorised COO scatter followed by conversion to CSR (duplicates summed).
Element blocks are processed in mesh order (tets, then hexes); per-element
data (mean B, volumes, nodal mass shares) is returned in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from ..geometry.finger import FEMesh
from .elements import element_dof_indices, element_stiffness
from .material import LinearElasticMaterial


@dataclass
class ElementData:
    """Per-block element quantities for post-processing / body forces."""

    conn: list[np.ndarray]  # connectivity per block
    B: list[np.ndarray]  # volume-averaged strain matrices per block (M_b, 6, 3n)
    volumes: np.ndarray  # (M,) all blocks concatenated
    mass_shares: list[np.ndarray]  # ∫ N_i dV per block (M_b, n)

    @property
    def n_elements(self) -> int:
        return int(self.volumes.shape[0])


def assemble_blocks(nodes: np.ndarray, blocks: list[np.ndarray], material: LinearElasticMaterial) -> tuple[sp.csr_matrix, ElementData]:
    """Assemble ``K`` (3N × 3N, CSR) from connectivity blocks (each (M_b, 10 or 20))."""
    C = material.stiffness_matrix()
    n_dof = 3 * nodes.shape[0]
    rows, cols, vals = [], [], []
    Bs, vols, mass = [], [], []
    for conn in blocks:
        conn = np.asarray(conn, dtype=np.int64)
        if conn.size == 0:
            continue
        Ke, Bbar, V, m = element_stiffness(nodes, conn, C)
        dofs = element_dof_indices(conn)  # (M, 3n)
        n = dofs.shape[1]
        rows.append(np.repeat(dofs[:, :, None], n, axis=2).reshape(-1))
        cols.append(np.repeat(dofs[:, None, :], n, axis=1).reshape(-1))
        vals.append(Ke.reshape(-1))
        Bs.append(Bbar)
        vols.append(V)
        mass.append(m)
    if rows:
        K = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_dof, n_dof)).tocsr()
        K.sum_duplicates()
    else:
        K = sp.csr_matrix((n_dof, n_dof))
    data = ElementData(
        conn=[np.asarray(c, dtype=np.int64) for c in blocks if np.asarray(c).size],
        B=Bs, volumes=np.concatenate(vols) if vols else np.zeros(0), mass_shares=mass,
    )
    return K, data


def assemble_global_stiffness(
    nodes: np.ndarray,
    elements,
    material: LinearElasticMaterial,
    return_element_data: bool = False,
):
    """Assemble the global stiffness matrix K.

    ``elements`` is a connectivity array (M, 10) / (M, 20) or a list of such
    blocks. With ``return_element_data`` the :class:`ElementData` is returned too.
    """
    blocks = list(elements) if isinstance(elements, (list, tuple)) else [np.asarray(elements)]
    K, data = assemble_blocks(np.asarray(nodes, float), blocks, material)
    if return_element_data:
        return K, data
    return K


def assemble_mesh(mesh: FEMesh, material: LinearElasticMaterial, **kw):
    return assemble_global_stiffness(mesh.nodes, mesh.element_blocks, material, **kw)


def symmetry_error(K: sp.spmatrix) -> float:
    """max |K - K^T| relative to max |K| (should be ~1e-16)."""
    d = (K - K.T)
    return float(abs(d).max() / max(abs(K).max(), 1e-300))
