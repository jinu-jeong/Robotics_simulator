// Craig-Bampton modal-dynamics step — port of
// ``CraigBamptonBody._step_anchored`` and ``_step_free``.
//
// The body's *precomputed* CB basis (Phi_CB, M_r, A_r_inv, C_q,
// M_diag, fixed/free DOF lists, …) stays on the Python side; here
// we just consume them as Eigen::Ref<…> views per call, so the
// inner dense matvecs run in compiled code without paying NumPy's
// per-``@`` dispatch.
//
// The data is read row-major from numpy (C order); using
// ``RowMajor`` Eigen matrices on the C++ side keeps the underlying
// buffers identical (no transpose / copy at the boundary).

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <tuple>

namespace robosim {

using MatXRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Anchored step (some nodes Dirichlet-fixed). Mirrors
// ``CraigBamptonBody._step_anchored``.
//
//   x      : (n_nodes, 3)   current world positions   [READ]
//   v      : (n_nodes, 3)   current velocities         [READ]
//   nodes  : (n_nodes, 3)   reference rest config      [READ]
//   Phi_CB : (n_dof, n_r)   reduction basis            [READ]
//   M_r    : (n_r, n_r)     reduced mass               [READ]
//   A_r_inv: (n_r, n_r)     precomputed inverse system [READ]
//   C_q    : (n_r, n_free)  q_r ← C_q · u_free         [READ]
//   M_diag : (n_dof,)       full mass diagonal         [READ]
//   free_dofs  : (n_free,)  global free DOF indices    [READ]
//   fixed_dofs : (n_fixed,) global fixed DOF indices   [READ]
//   gravity    : (3,)
//   extra_force_flat : (n_dof,) optional extra force per DOF
//                       (length 0 ⇒ none)
//   dt, damping, m_total : scalars
//
// Returns:
//   x_new : (n_nodes, 3)
//   v_new : (n_nodes, 3)
//   q_r_new : (n_r,)
std::tuple<MatXRM, MatXRM, Eigen::VectorXd>
cb_step_anchored(const Eigen::Ref<const MatXRM>& x,
                 const Eigen::Ref<const MatXRM>& v,
                 const Eigen::Ref<const MatXRM>& nodes,
                 const Eigen::Ref<const MatXRM>& Phi_CB,
                 const Eigen::Ref<const MatXRM>& M_r,
                 const Eigen::Ref<const MatXRM>& A_r_inv,
                 const Eigen::Ref<const MatXRM>& C_q,
                 const Eigen::Ref<const Eigen::VectorXd>& M_diag,
                 const Eigen::Ref<const Eigen::VectorXi>& free_dofs,
                 const Eigen::Ref<const Eigen::VectorXi>& fixed_dofs,
                 const Eigen::Ref<const Eigen::Vector3d>& gravity,
                 const Eigen::Ref<const Eigen::VectorXd>& extra_force_flat,
                 double dt,
                 double damping);

// Free-floating step (no Dirichlet constraint — Kabsch rotation +
// centroid bookkeeping). Mirrors ``CraigBamptonBody._step_free``.
//
//   x_ref_body : (n_nodes, 3)  body-frame reference (mesh.nodes − t_ref)
//   m_total    : total mass for the centroid integrator
std::tuple<MatXRM, MatXRM, Eigen::VectorXd>
cb_step_free(const Eigen::Ref<const MatXRM>& x,
             const Eigen::Ref<const MatXRM>& v,
             const Eigen::Ref<const MatXRM>& x_ref_body,
             const Eigen::Ref<const MatXRM>& Phi_CB,
             const Eigen::Ref<const MatXRM>& M_r,
             const Eigen::Ref<const MatXRM>& A_r_inv,
             const Eigen::Ref<const Eigen::VectorXd>& M_diag,
             const Eigen::Ref<const Eigen::Vector3d>& gravity,
             const Eigen::Ref<const Eigen::VectorXd>& extra_force_flat,
             double dt,
             double damping,
             double m_total);

}  // namespace robosim
