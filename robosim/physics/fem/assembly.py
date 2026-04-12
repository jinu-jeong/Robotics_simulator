"""Global force and stiffness matrix assembly for FEM.

Vectorized with NumPy batch operations — all elements are processed in parallel.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from robosim.physics.fem.mesh import TetMesh
from robosim.physics.fem.elements import compute_shape_derivatives
from robosim.physics.fem.materials import CorotationalElastic, NeoHookean


def precompute_element_data(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """Precompute shape function gradients and volumes for all elements.

    Returns
    -------
    dN_all : (n_elements, 4, 3) shape function gradients
    volumes : (n_elements,) element volumes
    """
    ne = mesh.n_elements
    dN_all = np.zeros((ne, 4, 3))
    volumes = np.zeros(ne)

    for e in range(ne):
        n0, n1, n2, n3 = mesh.elements[e]
        dN_all[e], volumes[e] = compute_shape_derivatives(
            mesh.nodes[n0], mesh.nodes[n1], mesh.nodes[n2], mesh.nodes[n3]
        )

    return dN_all, volumes


def _batch_deformation_gradients(
    x: np.ndarray, elements: np.ndarray, dN_all: np.ndarray
) -> np.ndarray:
    """Compute deformation gradient for all elements at once.

    F_e = x_def_e^T @ dN_e   for each element e.

    Parameters
    ----------
    x : (n_nodes, 3)
    elements : (n_elements, 4)
    dN_all : (n_elements, 4, 3)

    Returns
    -------
    F_all : (n_elements, 3, 3)
    """
    x_def = x[elements]           # (ne, 4, 3)
    F_all = np.einsum('eai,eaj->eij', x_def, dN_all)  # (ne, 3, 3)
    return F_all


def _batch_polar_decomposition(F_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar decomposition F = R @ S for all elements (batch SVD).

    Returns
    -------
    R_all : (ne, 3, 3) rotation matrices
    S_all : (ne, 3, 3) stretch tensors
    """
    ne = F_all.shape[0]
    R_all = np.zeros_like(F_all)
    S_all = np.zeros_like(F_all)

    # Guard: replace degenerate/NaN deformation gradients with identity
    bad = ~np.all(np.isfinite(F_all.reshape(ne, -1)), axis=1)
    if np.any(bad):
        F_all[bad] = np.eye(3)

    # np.linalg.svd is already batched
    U, sigma, Vt = np.linalg.svd(F_all)

    # Fix reflections (ensure det(R) = +1)
    det_UV = np.linalg.det(U) * np.linalg.det(Vt)
    flip = det_UV < 0
    if np.any(flip):
        U[flip, :, 2] *= -1
        sigma[flip, 2] *= -1

    # Clamp singular values: prevent inverted / degenerate elements
    # from producing extreme stresses that blow up the simulation.
    sigma = np.maximum(sigma, 0.1)

    R_all = U @ Vt                                      # (ne, 3, 3)
    S_all = np.einsum('eji,ej,ejk->eik', Vt, sigma, Vt)  # V @ diag(sigma) @ V^T
    return R_all, S_all


def _batch_corotational_stress(
    F_all: np.ndarray, R_all: np.ndarray, S_all: np.ndarray,
    mu: float, lam: float
) -> np.ndarray:
    """Compute first Piola-Kirchhoff stress P for all elements.

    P = R @ (2*mu*(S-I) + lam*tr(S-I)*I)

    Returns
    -------
    P_all : (ne, 3, 3)
    """
    ne = F_all.shape[0]
    I3 = np.eye(3)[np.newaxis]                         # (1, 3, 3)
    eps = S_all - I3                                    # (ne, 3, 3)
    trace_eps = np.trace(eps, axis1=1, axis2=2)         # (ne,)
    T = 2.0 * mu * eps + lam * trace_eps[:, None, None] * I3
    P_all = R_all @ T
    return P_all


def assemble_forces(
    mesh: TetMesh,
    x: np.ndarray,
    material: CorotationalElastic | NeoHookean,
    dN_all: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Assemble global internal force vector (vectorized).

    f_int = -sum_e vol_e * P_e @ dN_e^T  scattered to global DOFs.
    """
    ne = mesh.n_elements
    n_dof = mesh.n_nodes * 3

    F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)

    if isinstance(material, CorotationalElastic):
        R_all, S_all = _batch_polar_decomposition(F_all)
        P_all = _batch_corotational_stress(F_all, R_all, S_all, material.mu, material.lam)
    else:
        # Neo-Hookean batch
        P_all = _batch_neohookean_stress(F_all, material.mu, material.lam)

    # H_all[e] = vol_e * P_e @ dN_e^T  -> (ne, 3, 4)
    H_all = volumes[:, None, None] * np.einsum('eij,ekj->eik', P_all, dN_all)

    # Scatter: f[node_a*3 : node_a*3+3] -= H_all[e, :, a]
    f = np.zeros(n_dof)
    for a in range(4):
        node_indices = mesh.elements[:, a]  # (ne,)
        # Scatter H_all[:, :, a] into f at node_indices*3
        np.add.at(f.reshape(-1, 3), node_indices, -H_all[:, :, a])

    return f


def assemble_stiffness(
    mesh: TetMesh,
    x: np.ndarray,
    material: CorotationalElastic | NeoHookean,
    dN_all: np.ndarray,
    volumes: np.ndarray,
) -> sp.csr_matrix:
    """Assemble global stiffness matrix K (vectorized COO assembly).

    For corotational: K_e[ab] = vol * R @ (mu*(dNa.dNb)*I + mu*dNb(x)dNa + lam*dNa(x)dNb) @ R^T
    """
    ne = mesh.n_elements
    n_dof = mesh.n_nodes * 3

    F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)

    if isinstance(material, CorotationalElastic):
        R_all, S_all = _batch_polar_decomposition(F_all)
        Ke_all = _batch_corotational_stiffness(R_all, dN_all, volumes, material.mu, material.lam)
    else:
        Ke_all = _batch_numerical_stiffness(mesh, x, material, dN_all, volumes)

    # Build COO triplets from all element stiffness matrices
    # Ke_all: (ne, 12, 12), elements: (ne, 4)
    # Global DOF indices for each element: (ne, 4) -> (ne, 12)
    elem_dofs = np.repeat(mesh.elements * 3, 3, axis=1) + np.tile([0, 1, 2], 4)  # (ne, 12)

    # Row/col indices for all elements: (ne, 12, 12)
    row_idx = np.repeat(elem_dofs[:, :, np.newaxis], 12, axis=2)  # (ne, 12, 12)
    col_idx = np.repeat(elem_dofs[:, np.newaxis, :], 12, axis=1)  # (ne, 12, 12)

    K = sp.coo_matrix(
        (Ke_all.ravel(), (row_idx.ravel(), col_idx.ravel())),
        shape=(n_dof, n_dof),
    )
    return K.tocsr()


def _batch_corotational_stiffness(
    R_all: np.ndarray, dN_all: np.ndarray,
    volumes: np.ndarray, mu: float, lam: float,
) -> np.ndarray:
    """Compute all element stiffness matrices (vectorized).

    Returns
    -------
    Ke_all : (ne, 12, 12)
    """
    ne = R_all.shape[0]
    Ke_all = np.zeros((ne, 12, 12))

    for a in range(4):
        for b in range(4):
            dNa = dN_all[:, a, :]  # (ne, 3)
            dNb = dN_all[:, b, :]  # (ne, 3)

            # dot products: (ne,)
            dot_ab = np.einsum('ei,ei->e', dNa, dNb)

            # Block: vol * (mu * dot * I + mu * outer(dNb, dNa) + lam * outer(dNa, dNb))
            I3 = np.eye(3)[np.newaxis]                          # (1,3,3)
            K_block = (
                mu * dot_ab[:, None, None] * I3
                + mu * np.einsum('ei,ej->eij', dNb, dNa)
                + lam * np.einsum('ei,ej->eij', dNa, dNb)
            )
            K_block *= volumes[:, None, None]

            # Rotate: R @ K_block @ R^T
            K_block = np.einsum('eij,ejk,elk->eil', R_all, K_block, R_all)

            i0, i1 = a * 3, a * 3 + 3
            j0, j1 = b * 3, b * 3 + 3
            Ke_all[:, i0:i1, j0:j1] = K_block

    return Ke_all


def _batch_neohookean_stress(F_all: np.ndarray, mu: float, lam: float) -> np.ndarray:
    """Neo-Hookean stress for all elements."""
    ne = F_all.shape[0]
    P_all = np.zeros_like(F_all)
    for e in range(ne):
        F = F_all[e]
        J = np.linalg.det(F)
        if J < 1e-10:
            J = 1e-10
        F_inv_T = np.linalg.inv(F).T
        P_all[e] = mu * (F - F_inv_T) + lam * np.log(J) * F_inv_T
    return P_all


def _batch_numerical_stiffness(
    mesh: TetMesh, x: np.ndarray, material,
    dN_all: np.ndarray, volumes: np.ndarray, eps: float = 1e-7,
) -> np.ndarray:
    """Numerical stiffness for all elements (fallback for Neo-Hookean)."""
    ne = mesh.n_elements
    Ke_all = np.zeros((ne, 12, 12))
    for e in range(ne):
        nodes_e = mesh.elements[e]
        dN = dN_all[e]
        vol = volumes[e]
        if vol < 1e-20:
            continue
        x_def = x[nodes_e].copy()
        F0 = x_def.T @ dN
        P0, _ = material.compute_stress(F0)
        f0 = (-vol * (P0 @ dN.T)).T.ravel()

        for j in range(12):
            a, d = divmod(j, 3)
            x_pert = x_def.copy()
            x_pert[a, d] += eps
            F_p = x_pert.T @ dN
            P_p, _ = material.compute_stress(F_p)
            f_p = (-vol * (P_p @ dN.T)).T.ravel()
            Ke_all[e, :, j] = -(f_p - f0) / eps

        Ke_all[e] = 0.5 * (Ke_all[e] + Ke_all[e].T)
    return Ke_all


def batch_von_mises(
    mesh: TetMesh,
    x: np.ndarray,
    material: CorotationalElastic,
    dN_all: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Compute per-node von Mises stress (vectorized).

    Returns
    -------
    vm_per_node : (n_nodes,) averaged von Mises stress
    """
    F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)
    R_all, S_all = _batch_polar_decomposition(F_all)

    I3 = np.eye(3)[np.newaxis]
    eps = S_all - I3                                         # (ne, 3, 3)
    trace_eps = np.trace(eps, axis1=1, axis2=2)              # (ne,)
    sigma = 2.0 * material.mu * eps + material.lam * trace_eps[:, None, None] * I3

    # Deviatoric: s = sigma - tr(sigma)/3 * I
    trace_sigma = np.trace(sigma, axis1=1, axis2=2)          # (ne,)
    dev = sigma - (trace_sigma[:, None, None] / 3.0) * I3
    vm = np.sqrt(1.5 * np.sum(dev**2, axis=(1, 2)))         # (ne,)

    # Scatter to nodes (average)
    stress_sum = np.zeros(mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    for a in range(4):
        np.add.at(stress_sum, mesh.elements[:, a], vm)
        np.add.at(count, mesh.elements[:, a], 1.0)
    count[count == 0] = 1.0
    return stress_sum / count


def assemble_mass_matrix(
    mesh: TetMesh,
    density: float,
    volumes: np.ndarray,
) -> sp.csr_matrix:
    """Assemble lumped mass matrix (diagonal, vectorized)."""
    n_dof = mesh.n_nodes * 3
    diag = np.zeros(n_dof)

    # Mass per node = density * volume / 4 for each adjacent element
    elem_node_mass = density * volumes / 4.0  # (ne,)

    for a in range(4):
        node_indices = mesh.elements[:, a]
        np.add.at(diag[::3], node_indices, elem_node_mass)
        np.add.at(diag[1::3], node_indices, elem_node_mass)
        np.add.at(diag[2::3], node_indices, elem_node_mass)

    return sp.diags(diag, format="csr")
