"""Global force and stiffness matrix assembly for FEM.

Supports Tet4 (legacy vectorized path) and general elements (Hex8, Tet10)
via multi-point Gauss quadrature.  CompositeMesh with mixed element blocks
is also supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from robosim.physics.fem.mesh import TetMesh, FEMesh, CompositeMesh, ElementBlock
from robosim.physics.fem.elements import (
    ElementType, NODES_PER_ELEMENT,
    compute_shape_derivatives, shape_function, gauss_rule,
)
from robosim.physics.fem.materials import (
    CorotationalElastic, CorotationalPlastic, NeoHookean,
)


# ═══════════════════════════════════════════════════════════════
# Precomputed element integration data
# ═══════════════════════════════════════════════════════════════

@dataclass
class ElementIntegrationData:
    """Precomputed per-element, per-Gauss-point shape gradients and weights.

    For Tet4 (1 Gauss point): compatible with legacy (ne, 4, 3) shape.
    """
    # (ne, n_gauss, nodes_per_elem, 3) physical-space shape gradients
    dN: np.ndarray
    # (ne, n_gauss) integration weight = det(J) * gauss_weight
    weights: np.ndarray
    n_gauss: int
    nodes_per_elem: int
    element_type: ElementType


@dataclass
class CompositeIntegrationData:
    """Precomputed integration data for a CompositeMesh with multiple element blocks.

    Each block has its own ElementIntegrationData and connectivity.
    Assembly functions iterate over blocks, accumulating into the same global arrays.
    """
    block_data: list[ElementIntegrationData]   # one per block
    block_elements: list[np.ndarray]           # connectivity per block
    n_global_nodes: int                        # total unique nodes


class _BlockAdapter:
    """Lightweight adapter: makes an ElementBlock look like an FEMesh for assembly.

    Existing general-path functions (_assemble_forces_general, etc.) access only
    mesh.elements, mesh.n_elements, and mesh.n_nodes.  This adapter provides
    exactly those, with n_nodes set to the *global* count so DOF vectors/matrices
    are sized correctly for the full composite mesh.
    """
    __slots__ = ("nodes", "elements", "n_elements", "n_nodes")

    def __init__(self, global_nodes: np.ndarray, block_elements: np.ndarray,
                 n_global_nodes: int):
        self.nodes = global_nodes
        self.elements = block_elements
        self.n_elements = block_elements.shape[0]
        self.n_nodes = n_global_nodes


def precompute_element_data(mesh) -> tuple:
    """Precompute shape function gradients and integration weights.

    Parameters
    ----------
    mesh : TetMesh or FEMesh

    Returns
    -------
    For TetMesh (backward compatible): (dN_all, volumes) as (ne,4,3) and (ne,)
    For FEMesh: ElementIntegrationData
    """
    if isinstance(mesh, CompositeMesh):
        return _precompute_composite(mesh)

    if isinstance(mesh, TetMesh):
        return _precompute_tet4(mesh)

    etype = mesh.element_type
    if etype == ElementType.TET4:
        return _precompute_tet4(mesh)

    return _precompute_general(mesh)


def _precompute_tet4(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Legacy Tet4 precompute — returns (dN_all, volumes)."""
    ne = mesh.n_elements
    dN_all = np.zeros((ne, 4, 3))
    volumes = np.zeros(ne)
    for e in range(ne):
        n0, n1, n2, n3 = mesh.elements[e]
        dN_all[e], volumes[e] = compute_shape_derivatives(
            mesh.nodes[n0], mesh.nodes[n1], mesh.nodes[n2], mesh.nodes[n3]
        )
    return dN_all, volumes


def _precompute_general(mesh: FEMesh) -> ElementIntegrationData:
    """General precompute for Hex8, Tet10, etc."""
    etype = mesh.element_type
    npe = NODES_PER_ELEMENT[etype]
    ne = mesh.n_elements
    gp_ref, gp_wts = gauss_rule(etype)
    ng = len(gp_wts)

    dN_all = np.zeros((ne, ng, npe, 3))
    weights = np.zeros((ne, ng))

    for e in range(ne):
        x_ref = mesh.nodes[mesh.elements[e]]  # (npe, 3)

        for g in range(ng):
            _, dNdxi = shape_function(etype, gp_ref[g])  # (npe, 3)

            # Jacobian: J = x_ref^T @ dNdxi  (3x3)
            J = x_ref.T @ dNdxi  # (3, 3)
            det_J = np.linalg.det(J)

            if abs(det_J) < 1e-20:
                continue

            J_inv = np.linalg.inv(J)
            dN_phys = dNdxi @ J_inv  # (npe, 3) in physical coords

            dN_all[e, g] = dN_phys
            weights[e, g] = abs(det_J) * gp_wts[g]

    return ElementIntegrationData(
        dN=dN_all, weights=weights, n_gauss=ng,
        nodes_per_elem=npe, element_type=etype,
    )


def _precompute_composite(mesh: CompositeMesh) -> CompositeIntegrationData:
    """Precompute integration data for every block in a CompositeMesh."""
    block_data = []
    block_elements = []

    for block in mesh.blocks:
        adapter = _BlockAdapter(mesh.nodes, block.elements, mesh.n_nodes)
        etype = block.element_type

        if etype == ElementType.TET4:
            # Use general path (not legacy tuple) so all blocks have same data format
            npe = NODES_PER_ELEMENT[etype]
            ne = block.n_elements
            gp_ref, gp_wts = gauss_rule(etype)
            ng = len(gp_wts)
            dN_all = np.zeros((ne, ng, npe, 3))
            weights = np.zeros((ne, ng))
            for e in range(ne):
                x_ref = mesh.nodes[block.elements[e]]
                for g in range(ng):
                    _, dNdxi = shape_function(etype, gp_ref[g])
                    J = x_ref.T @ dNdxi
                    det_J = np.linalg.det(J)
                    if abs(det_J) < 1e-20:
                        continue
                    J_inv = np.linalg.inv(J)
                    dN_all[e, g] = dNdxi @ J_inv
                    weights[e, g] = abs(det_J) * gp_wts[g]
            edata = ElementIntegrationData(
                dN=dN_all, weights=weights, n_gauss=ng,
                nodes_per_elem=npe, element_type=etype,
            )
        else:
            # Hex8, Tet10 — reuse _precompute_general via adapter
            adapter_mesh = type('_Mesh', (), {
                'nodes': mesh.nodes,
                'elements': block.elements,
                'element_type': etype,
                'n_elements': block.n_elements,
                'n_nodes': mesh.n_nodes,
            })()
            edata = _precompute_general(adapter_mesh)

        block_data.append(edata)
        block_elements.append(block.elements)

    return CompositeIntegrationData(
        block_data=block_data,
        block_elements=block_elements,
        n_global_nodes=mesh.n_nodes,
    )


# ═══════════════════════════════════════════════════════════════
# Batch deformation gradient & polar decomposition
# ═══════════════════════════════════════════════════════════════

def _batch_deformation_gradients(x, elements, dN):
    """Compute F for all elements. dN is (ne, nodes_per_elem, 3)."""
    x_def = x[elements]                                    # (ne, npe, 3)
    F_all = np.einsum('eai,eaj->eij', x_def, dN)          # (ne, 3, 3)
    return F_all


def _batch_polar_decomposition(F_all):
    """Polar decomposition F = R @ S (batch SVD)."""
    ne = F_all.shape[0]
    bad = ~np.all(np.isfinite(F_all.reshape(ne, -1)), axis=1)
    if np.any(bad):
        F_all[bad] = np.eye(3)

    U, sigma, Vt = np.linalg.svd(F_all)
    det_UV = np.linalg.det(U) * np.linalg.det(Vt)
    flip = det_UV < 0
    if np.any(flip):
        U[flip, :, 2] *= -1
        sigma[flip, 2] *= -1

    sigma = np.maximum(sigma, 0.1)
    R_all = U @ Vt
    S_all = np.einsum('eji,ej,ejk->eik', Vt, sigma, Vt)
    return R_all, S_all


def _batch_corotational_stress(F_all, R_all, S_all, mu, lam):
    """First Piola-Kirchhoff stress P = R @ (2μ(S-I) + λ tr(S-I) I)."""
    I3 = np.eye(3)[np.newaxis]
    eps = S_all - I3
    trace_eps = np.trace(eps, axis1=1, axis2=2)
    T = 2.0 * mu * eps + lam * trace_eps[:, None, None] * I3
    return R_all @ T


def _batch_corotational_plastic_stress(
    F_all, R_all, S_all, eps_p_in, mu, lam, sigma_Y, hardening,
):
    """Vectorised J2 radial-return for an entire element batch.

    Mirrors :meth:`CorotationalPlastic.compute_stress` element-wise but
    operates on (ne, 3, 3) arrays without a Python loop.

    Returns ``(P_all, eps_p_new)`` — first Piola stress per element and
    the updated plastic strain (caller writes back to per-element store).
    """
    ne = F_all.shape[0]
    I3 = np.eye(3)[np.newaxis]                         # (1, 3, 3)

    # Trial elastic strain in corotated frame: sym(S - I) - eps_p.
    eps_e_trial = 0.5 * (S_all + np.transpose(S_all, (0, 2, 1))) - I3 - eps_p_in
    tr_e = np.trace(eps_e_trial, axis1=1, axis2=2)     # (ne,)
    sigma_trial = (
        2.0 * mu * eps_e_trial
        + lam * tr_e[:, None, None] * I3
    )

    # Deviatoric trial stress and its Frobenius norm per element.
    tr_sig = np.trace(sigma_trial, axis1=1, axis2=2)
    dev = sigma_trial - (tr_sig / 3.0)[:, None, None] * I3
    norm_dev = np.linalg.norm(dev, axis=(1, 2))        # (ne,)

    # Effective yield radius (linear isotropic hardening).
    eps_p_eq = np.sqrt(2.0 / 3.0) * np.linalg.norm(eps_p_in, axis=(1, 2))
    sigma_Y_eff = sigma_Y + hardening * eps_p_eq
    f = norm_dev - np.sqrt(2.0 / 3.0) * sigma_Y_eff    # (ne,)

    yielded = (f > 0.0) & (norm_dev > 1e-16)

    # Δγ and flow direction n; both zero where elastic.
    safe_norm = np.where(norm_dev > 1e-16, norm_dev, 1.0)
    dgamma = np.where(yielded, f / (2.0 * mu + 2.0 * hardening / 3.0), 0.0)
    n = np.where(
        yielded[:, None, None],
        dev / safe_norm[:, None, None],
        np.zeros_like(dev),
    )

    sigma_new = sigma_trial - 2.0 * mu * dgamma[:, None, None] * n
    eps_p_new = eps_p_in + dgamma[:, None, None] * n

    P_all = np.einsum("eij,ejk->eik", R_all, sigma_new)
    return P_all, eps_p_new


def _batch_neohookean_stress(F_all, mu, lam):
    ne = F_all.shape[0]
    P_all = np.zeros_like(F_all)
    for e in range(ne):
        F = F_all[e]
        J = max(np.linalg.det(F), 1e-10)
        F_inv_T = np.linalg.inv(F).T
        P_all[e] = mu * (F - F_inv_T) + lam * np.log(J) * F_inv_T
    return P_all


# ═══════════════════════════════════════════════════════════════
# Force assembly
# ═══════════════════════════════════════════════════════════════

def assemble_forces(
    mesh, x, material, dN_all, volumes,
    return_intermediates=False, eps_p=None,
):
    """Assemble global internal force vector.

    Parameters
    ----------
    mesh : TetMesh or FEMesh
    dN_all : (ne,4,3) for Tet4  OR  ElementIntegrationData for Hex8/Tet10
    volumes : (ne,) for Tet4  OR  ignored for general (weights in dN_all)
    eps_p : (ne, 3, 3) per-element plastic strain, REQUIRED if material is
        CorotationalPlastic. The function returns the *updated* ``eps_p``
        alongside the force vector (and intermediates) — caller decides
        whether to commit it (Newton iters typically discard until
        convergence).

    Plastic Tet4 path returns ``(f, eps_p_new)`` or
    ``(f, F_all, R_all, S_all, eps_p_new)`` with ``return_intermediates``.
    Plastic with general/composite meshes is not yet wired (raises).
    """
    if isinstance(material, CorotationalPlastic):
        if isinstance(dN_all, (ElementIntegrationData, CompositeIntegrationData)):
            raise NotImplementedError(
                "CorotationalPlastic + general/composite mesh not wired yet. "
                "Use TetMesh / Tet4 elements."
            )
        return _assemble_forces_tet4(
            mesh, x, material, dN_all, volumes, return_intermediates,
            eps_p_in=eps_p,
        )
    if isinstance(dN_all, CompositeIntegrationData):
        return _assemble_forces_composite(mesh, x, material, dN_all, return_intermediates)
    if isinstance(dN_all, ElementIntegrationData):
        return _assemble_forces_general(mesh, x, material, dN_all, return_intermediates)
    return _assemble_forces_tet4(mesh, x, material, dN_all, volumes, return_intermediates)


def _assemble_forces_tet4(mesh, x, material, dN_all, volumes, return_intermediates,
                          eps_p_in=None):
    """Optimized Tet4 force assembly (single Gauss point, vectorized).

    For :class:`CorotationalPlastic` materials, ``eps_p_in`` (per-element
    3×3) must be supplied; the trial-elastic / radial-return projection
    runs and the updated ``eps_p_new`` is returned alongside ``f``.
    """
    ne = mesh.n_elements
    n_dof = mesh.n_nodes * 3

    F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)
    R_all = S_all = None
    eps_p_new = None

    if isinstance(material, CorotationalPlastic):
        if eps_p_in is None:
            eps_p_in = np.zeros((ne, 3, 3), dtype=np.float64)
        R_all, S_all = _batch_polar_decomposition(F_all)
        P_all, eps_p_new = _batch_corotational_plastic_stress(
            F_all, R_all, S_all, eps_p_in,
            material.mu, material.lam,
            material.yield_stress, material.hardening,
        )
    elif isinstance(material, CorotationalElastic):
        R_all, S_all = _batch_polar_decomposition(F_all)
        P_all = _batch_corotational_stress(F_all, R_all, S_all, material.mu, material.lam)
    else:
        P_all = _batch_neohookean_stress(F_all, material.mu, material.lam)

    H_all = volumes[:, None, None] * np.einsum('eij,ekj->eik', P_all, dN_all)

    f = np.zeros(n_dof)
    for a in range(4):
        np.add.at(f.reshape(-1, 3), mesh.elements[:, a], -H_all[:, :, a])

    if return_intermediates:
        if eps_p_new is not None:
            return f, F_all, R_all, S_all, eps_p_new
        return f, F_all, R_all, S_all
    if eps_p_new is not None:
        return f, eps_p_new
    return f


def _assemble_forces_general(mesh, x, material, edata: ElementIntegrationData, return_intermediates):
    """General force assembly with multi-point Gauss integration."""
    ne = mesh.n_elements
    npe = edata.nodes_per_elem
    ng = edata.n_gauss
    n_dof = mesh.n_nodes * 3

    f = np.zeros(n_dof)
    # Accumulate R_all from first Gauss point (for stiffness reuse)
    R_first = None
    S_first = None
    F_first = None

    for g in range(ng):
        dN_g = edata.dN[:, g, :, :]    # (ne, npe, 3)
        w_g = edata.weights[:, g]       # (ne,)

        F_all = _batch_deformation_gradients(x, mesh.elements, dN_g)

        if isinstance(material, CorotationalElastic):
            R_all, S_all = _batch_polar_decomposition(F_all)
            P_all = _batch_corotational_stress(F_all, R_all, S_all, material.mu, material.lam)
        else:
            P_all = _batch_neohookean_stress(F_all, material.mu, material.lam)

        if g == 0:
            R_first, S_first, F_first = R_all, S_all, F_all

        # H[e, i, a] = w * P[e,i,j] * dN[e,a,j]
        H_all = w_g[:, None, None] * np.einsum('eij,eaj->eia', P_all, dN_g)

        for a in range(npe):
            np.add.at(f.reshape(-1, 3), mesh.elements[:, a], -H_all[:, :, a])

    if return_intermediates:
        return f, F_first, R_first, S_first
    return f


def _assemble_forces_composite(mesh, x, material, cdata: CompositeIntegrationData, return_intermediates):
    """Composite force assembly: accumulate over all blocks."""
    n_dof = mesh.n_nodes * 3
    f = np.zeros(n_dof)

    for edata, elements in zip(cdata.block_data, cdata.block_elements):
        adapter = _BlockAdapter(mesh.nodes, elements, mesh.n_nodes)
        f_block = _assemble_forces_general(adapter, x, material, edata, return_intermediates=False)
        f += f_block

    if return_intermediates:
        return f, None, None, None
    return f


# ═══════════════════════════════════════════════════════════════
# Stiffness assembly
# ═══════════════════════════════════════════════════════════════

def assemble_stiffness(
    mesh, x, material, dN_all, volumes, R_all=None,
):
    """Assemble global stiffness matrix K."""
    if isinstance(dN_all, CompositeIntegrationData):
        return _assemble_stiffness_composite(mesh, x, material, dN_all)
    if isinstance(dN_all, ElementIntegrationData):
        return _assemble_stiffness_general(mesh, x, material, dN_all, R_all)
    return _assemble_stiffness_tet4(mesh, x, material, dN_all, volumes, R_all)


def _assemble_stiffness_tet4(mesh, x, material, dN_all, volumes, R_all):
    """Tet4 stiffness (single Gauss point, vectorized)."""
    ne = mesh.n_elements
    n_dof = mesh.n_nodes * 3

    if isinstance(material, (CorotationalElastic, CorotationalPlastic)):
        # Plastic uses the elastic tangent as a modified-Newton approximation:
        # forces use the true projected stress (radial return), but the
        # Hessian stays the constant elastic K. Loses quadratic Newton
        # convergence inside the plastic zone but stays stable, and avoids
        # the cost of the consistent C^{ep} tangent.
        if R_all is None:
            F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)
            R_all, _ = _batch_polar_decomposition(F_all)
        Ke_all = _batch_corotational_stiffness(R_all, dN_all, volumes, material.mu, material.lam, 4)
    else:
        Ke_all = _batch_numerical_stiffness_tet4(mesh, x, material, dN_all, volumes)

    ndof_e = 12
    elem_dofs = np.repeat(mesh.elements * 3, 3, axis=1) + np.tile([0, 1, 2], 4)
    row_idx = np.repeat(elem_dofs[:, :, np.newaxis], ndof_e, axis=2)
    col_idx = np.repeat(elem_dofs[:, np.newaxis, :], ndof_e, axis=1)

    K = sp.coo_matrix((Ke_all.ravel(), (row_idx.ravel(), col_idx.ravel())), shape=(n_dof, n_dof))
    return K.tocsr()


def _assemble_stiffness_general(mesh, x, material, edata: ElementIntegrationData, R_all_hint):
    """General stiffness assembly with multi-point Gauss integration."""
    ne = mesh.n_elements
    npe = edata.nodes_per_elem
    ng = edata.n_gauss
    n_dof = mesh.n_nodes * 3
    ndof_e = npe * 3

    Ke_all = np.zeros((ne, ndof_e, ndof_e))

    for g in range(ng):
        dN_g = edata.dN[:, g, :, :]  # (ne, npe, 3)
        w_g = edata.weights[:, g]     # (ne,)

        if isinstance(material, CorotationalElastic):
            F_all = _batch_deformation_gradients(x, mesh.elements, dN_g)
            R_all, _ = _batch_polar_decomposition(F_all)
            Ke_g = _batch_corotational_stiffness(R_all, dN_g, w_g, material.mu, material.lam, npe)
        else:
            # Numerical stiffness per Gauss point (fallback)
            Ke_g = _batch_numerical_stiffness_general(
                mesh, x, material, dN_g, w_g, npe
            )

        Ke_all += Ke_g

    # Build sparse matrix
    elem_dofs = np.repeat(mesh.elements * 3, 3, axis=1) + np.tile([0, 1, 2], npe)
    row_idx = np.repeat(elem_dofs[:, :, np.newaxis], ndof_e, axis=2)
    col_idx = np.repeat(elem_dofs[:, np.newaxis, :], ndof_e, axis=1)

    K = sp.coo_matrix((Ke_all.ravel(), (row_idx.ravel(), col_idx.ravel())), shape=(n_dof, n_dof))
    return K.tocsr()


def _assemble_stiffness_composite(mesh, x, material, cdata: CompositeIntegrationData):
    """Composite stiffness assembly: sum sparse K from each block."""
    n_dof = mesh.n_nodes * 3
    K = sp.csr_matrix((n_dof, n_dof))

    for edata, elements in zip(cdata.block_data, cdata.block_elements):
        adapter = _BlockAdapter(mesh.nodes, elements, mesh.n_nodes)
        K_block = _assemble_stiffness_general(adapter, x, material, edata, R_all_hint=None)
        K = K + K_block

    return K


def _batch_corotational_stiffness(R_all, dN_all, weights, mu, lam, npe):
    """Corotational stiffness for any element (vectorized over elements)."""
    ne = R_all.shape[0]
    ndof_e = npe * 3
    Ke_all = np.zeros((ne, ndof_e, ndof_e))

    for a in range(npe):
        for b in range(npe):
            dNa = dN_all[:, a, :]  # (ne, 3)
            dNb = dN_all[:, b, :]  # (ne, 3)
            dot_ab = np.einsum('ei,ei->e', dNa, dNb)

            I3 = np.eye(3)[np.newaxis]
            K_block = (
                mu * dot_ab[:, None, None] * I3
                + mu * np.einsum('ei,ej->eij', dNb, dNa)
                + lam * np.einsum('ei,ej->eij', dNa, dNb)
            )
            K_block *= weights[:, None, None]

            # Rotate: R @ K_block @ R^T
            K_block = np.einsum('eij,ejk,elk->eil', R_all, K_block, R_all)

            i0, i1 = a * 3, a * 3 + 3
            j0, j1 = b * 3, b * 3 + 3
            Ke_all[:, i0:i1, j0:j1] = K_block

    return Ke_all


def _batch_numerical_stiffness_tet4(mesh, x, material, dN_all, volumes):
    """Numerical stiffness for Tet4 (Neo-Hookean fallback)."""
    ne = mesh.n_elements
    Ke_all = np.zeros((ne, 12, 12))
    eps = 1e-7
    for e in range(ne):
        nodes_e = mesh.elements[e]
        dN = dN_all[e]; vol = volumes[e]
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


def _batch_numerical_stiffness_general(mesh, x, material, dN_g, w_g, npe):
    """Numerical stiffness for general elements at a single Gauss point."""
    ne = mesh.n_elements
    ndof_e = npe * 3
    Ke_all = np.zeros((ne, ndof_e, ndof_e))
    eps = 1e-7
    for e in range(ne):
        nodes_e = mesh.elements[e]
        dN = dN_g[e]; w = w_g[e]
        if w < 1e-20:
            continue
        x_def = x[nodes_e].copy()
        F0 = x_def.T @ dN
        J0 = max(np.linalg.det(F0), 1e-10)
        F_inv_T0 = np.linalg.inv(F0).T
        P0 = material.mu * (F0 - F_inv_T0) + material.lam * np.log(J0) * F_inv_T0
        f0 = (-w * (P0 @ dN.T)).T.ravel()
        for j in range(ndof_e):
            a, d = divmod(j, 3)
            x_pert = x_def.copy()
            x_pert[a, d] += eps
            F_p = x_pert.T @ dN
            J_p = max(np.linalg.det(F_p), 1e-10)
            F_inv_T_p = np.linalg.inv(F_p).T
            P_p = material.mu * (F_p - F_inv_T_p) + material.lam * np.log(J_p) * F_inv_T_p
            f_p = (-w * (P_p @ dN.T)).T.ravel()
            Ke_all[e, :, j] = -(f_p - f0) / eps
        Ke_all[e] = 0.5 * (Ke_all[e] + Ke_all[e].T)
    return Ke_all


# ═══════════════════════════════════════════════════════════════
# Mass matrix
# ═══════════════════════════════════════════════════════════════

def assemble_mass_matrix(mesh, density, volumes) -> sp.csr_matrix:
    """Assemble lumped mass matrix (diagonal).

    Parameters
    ----------
    mesh : TetMesh or FEMesh
    density : material density
    volumes : (ne,) for Tet4  OR  ElementIntegrationData
    """
    n_dof = mesh.n_nodes * 3
    diag = np.zeros(n_dof)

    if isinstance(volumes, CompositeIntegrationData):
        cdata = volumes
        for edata, elements in zip(cdata.block_data, cdata.block_elements):
            elem_vol = edata.weights.sum(axis=1)  # (ne_block,)
            npe = edata.nodes_per_elem
            elem_node_mass = density * elem_vol / npe
            for a in range(npe):
                node_indices = elements[:, a]
                np.add.at(diag[::3], node_indices, elem_node_mass)
                np.add.at(diag[1::3], node_indices, elem_node_mass)
                np.add.at(diag[2::3], node_indices, elem_node_mass)
        return sp.diags(diag, format="csr")

    if isinstance(volumes, ElementIntegrationData):
        edata = volumes
        # Total weight per element = sum of Gauss weights
        elem_vol = edata.weights.sum(axis=1)  # (ne,)
        npe = edata.nodes_per_elem
        elem_node_mass = density * elem_vol / npe
        for a in range(npe):
            node_indices = mesh.elements[:, a]
            np.add.at(diag[::3], node_indices, elem_node_mass)
            np.add.at(diag[1::3], node_indices, elem_node_mass)
            np.add.at(diag[2::3], node_indices, elem_node_mass)
    else:
        # Tet4 legacy
        npe = 4
        elem_node_mass = density * volumes / npe
        for a in range(npe):
            node_indices = mesh.elements[:, a]
            np.add.at(diag[::3], node_indices, elem_node_mass)
            np.add.at(diag[1::3], node_indices, elem_node_mass)
            np.add.at(diag[2::3], node_indices, elem_node_mass)

    return sp.diags(diag, format="csr")


# ═══════════════════════════════════════════════════════════════
# Von Mises stress
# ═══════════════════════════════════════════════════════════════

def batch_von_mises(mesh, x, material, dN_all, volumes) -> np.ndarray:
    """Per-node averaged von Mises stress."""
    if isinstance(dN_all, CompositeIntegrationData):
        return _von_mises_composite(mesh, x, material, dN_all)
    if isinstance(dN_all, ElementIntegrationData):
        return _von_mises_general(mesh, x, material, dN_all)
    return _von_mises_tet4(mesh, x, material, dN_all)


def _von_mises_tet4(mesh, x, material, dN_all):
    F_all = _batch_deformation_gradients(x, mesh.elements, dN_all)
    R_all, S_all = _batch_polar_decomposition(F_all)

    I3 = np.eye(3)[np.newaxis]
    eps = S_all - I3
    trace_eps = np.trace(eps, axis1=1, axis2=2)
    sigma = 2.0 * material.mu * eps + material.lam * trace_eps[:, None, None] * I3

    trace_sigma = np.trace(sigma, axis1=1, axis2=2)
    dev = sigma - (trace_sigma[:, None, None] / 3.0) * I3
    vm = np.sqrt(1.5 * np.sum(dev**2, axis=(1, 2)))

    stress_sum = np.zeros(mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    for a in range(4):
        np.add.at(stress_sum, mesh.elements[:, a], vm)
        np.add.at(count, mesh.elements[:, a], 1.0)
    count[count == 0] = 1.0
    return stress_sum / count


def _von_mises_general(mesh, x, material, edata: ElementIntegrationData):
    """Von Mises for general elements (use first Gauss point as representative)."""
    dN_g0 = edata.dN[:, 0, :, :]  # first Gauss point
    F_all = _batch_deformation_gradients(x, mesh.elements, dN_g0)
    R_all, S_all = _batch_polar_decomposition(F_all)

    I3 = np.eye(3)[np.newaxis]
    eps = S_all - I3
    trace_eps = np.trace(eps, axis1=1, axis2=2)
    sigma = 2.0 * material.mu * eps + material.lam * trace_eps[:, None, None] * I3

    trace_sigma = np.trace(sigma, axis1=1, axis2=2)
    dev = sigma - (trace_sigma[:, None, None] / 3.0) * I3
    vm = np.sqrt(1.5 * np.sum(dev**2, axis=(1, 2)))

    npe = edata.nodes_per_elem
    stress_sum = np.zeros(mesh.n_nodes)
    count = np.zeros(mesh.n_nodes)
    for a in range(npe):
        np.add.at(stress_sum, mesh.elements[:, a], vm)
        np.add.at(count, mesh.elements[:, a], 1.0)
    count[count == 0] = 1.0
    return stress_sum / count


def _von_mises_composite(mesh, x, material, cdata: CompositeIntegrationData):
    """Von Mises for composite meshes: accumulate stress from all blocks."""
    n_nodes = mesh.n_nodes
    stress_sum = np.zeros(n_nodes)
    count = np.zeros(n_nodes)

    for edata, elements in zip(cdata.block_data, cdata.block_elements):
        dN_g0 = edata.dN[:, 0, :, :]  # first Gauss point
        F_all = _batch_deformation_gradients(x, elements, dN_g0)
        R_all, S_all = _batch_polar_decomposition(F_all)

        I3 = np.eye(3)[np.newaxis]
        eps = S_all - I3
        trace_eps = np.trace(eps, axis1=1, axis2=2)
        sigma = 2.0 * material.mu * eps + material.lam * trace_eps[:, None, None] * I3

        trace_sigma = np.trace(sigma, axis1=1, axis2=2)
        dev = sigma - (trace_sigma[:, None, None] / 3.0) * I3
        vm = np.sqrt(1.5 * np.sum(dev**2, axis=(1, 2)))

        npe = edata.nodes_per_elem
        for a in range(npe):
            np.add.at(stress_sum, elements[:, a], vm)
            np.add.at(count, elements[:, a], 1.0)

    count[count == 0] = 1.0
    return stress_sum / count
