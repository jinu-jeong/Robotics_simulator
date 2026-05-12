// Craig-Bampton step kernels. Direct port of ``_step_anchored`` /
// ``_step_free`` — same dense matvecs, same implicit-Euler form,
// same Kabsch SVD for the free-body rotation.

#include "robosim/cb.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace robosim {

namespace {

// Gather rows from a (n_dof,) buffer flattened from (n_nodes, 3) into a
// (n_free,) vector. Used for both M_diag and the displacement
// extraction. Faster than ``buf[idx]`` numpy fancy-indexing on the
// Python side because it stays in straight-line C++.
inline Eigen::VectorXd gather(
        const Eigen::Ref<const Eigen::VectorXd>& src,
        const Eigen::Ref<const Eigen::VectorXi>& idx) {
    const int n = idx.size();
    Eigen::VectorXd out(n);
    for (int i = 0; i < n; ++i) out[i] = src[idx[i]];
    return out;
}

// Build (n_dof,) gravity body force: per-DOF, mass[d] * g[d % 3].
inline Eigen::VectorXd build_gravity_force(
        const Eigen::Ref<const Eigen::VectorXd>& M_diag,
        const Eigen::Ref<const Eigen::Vector3d>& gravity,
        const Eigen::Ref<const Eigen::VectorXd>& extra) {
    const int n_dof = M_diag.size();
    Eigen::VectorXd f = Eigen::VectorXd::Zero(n_dof);
    for (int i = 0; i < n_dof; i += 3) {
        f[i + 0] = M_diag[i + 0] * gravity[0];
        f[i + 1] = M_diag[i + 1] * gravity[1];
        f[i + 2] = M_diag[i + 2] * gravity[2];
    }
    if (extra.size() == n_dof) f += extra;
    return f;
}

// Kabsch rotation: argmin_R || x_centered - x_ref_body · R^T ||_F.
// Matches the Python helper bit-for-bit (including the det sign fix).
inline Eigen::Matrix3d kabsch_rotation(
        const Eigen::Ref<const MatXRM>& x_centered,
        const Eigen::Ref<const MatXRM>& x_ref_body) {
    Eigen::Matrix3d H = x_ref_body.transpose() * x_centered;
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::Matrix3d U = svd.matrixU();
    const Eigen::Matrix3d V = svd.matrixV();
    const double d = (V * U.transpose()).determinant();
    Eigen::Matrix3d D = Eigen::Matrix3d::Identity();
    D(2, 2) = d;
    return V * D * U.transpose();
}

}  // namespace

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
                 double damping) {
    const int n_nodes = static_cast<int>(x.rows());
    const int n_dof = 3 * n_nodes;
    const int n_r = static_cast<int>(M_r.rows());
    const double dt2_inv = 1.0 / (dt * dt);

    // Flatten x/v/nodes view (row-major (n_nodes, 3) → length 3·n_nodes).
    Eigen::Map<const Eigen::VectorXd> x_flat(x.data(), n_dof);
    Eigen::Map<const Eigen::VectorXd> v_flat(v.data(), n_dof);
    Eigen::Map<const Eigen::VectorXd> x_ref_flat(nodes.data(), n_dof);

    // Displacement and velocity on the free DOFs.
    Eigen::VectorXd u_free(free_dofs.size());
    Eigen::VectorXd v_free(free_dofs.size());
    for (int i = 0; i < free_dofs.size(); ++i) {
        const int dof = free_dofs[i];
        u_free[i] = x_flat[dof] - x_ref_flat[dof];
        v_free[i] = v_flat[dof];
    }

    const Eigen::VectorXd q_r     = C_q * u_free;
    const Eigen::VectorXd q_r_dot = C_q * v_free;

    // External force on free DOFs only.
    Eigen::VectorXd f_ext = build_gravity_force(M_diag, gravity, extra_force_flat);
    Eigen::VectorXd f_ext_free(free_dofs.size());
    for (int i = 0; i < free_dofs.size(); ++i) f_ext_free[i] = f_ext[free_dofs[i]];

    // Reduced force: only the rows of Phi_CB corresponding to free
    // DOFs. We build the slice on the fly to mirror the Python ref.
    Eigen::MatrixXd Phi_free(free_dofs.size(), n_r);
    for (int i = 0; i < free_dofs.size(); ++i)
        Phi_free.row(i) = Phi_CB.row(free_dofs[i]);
    const Eigen::VectorXd f_r = Phi_free.transpose() * f_ext_free;

    // Implicit Euler: q_r_pred = q_r + dt · q_r_dot;
    // rhs = M_r · q_r_pred / dt² + f_r [+ damping · M_r · q_r / dt]
    const Eigen::VectorXd q_r_pred = q_r + dt * q_r_dot;
    Eigen::VectorXd rhs = (M_r * q_r_pred) * dt2_inv + f_r;
    if (damping > 0.0) rhs += (damping / dt) * (M_r * q_r);

    const Eigen::VectorXd q_r_new     = A_r_inv * rhs;
    const Eigen::VectorXd q_r_dot_new = (q_r_new - q_r) / dt;

    // Reconstruct: x_new = nodes + Phi · q_r_new (per-DOF), with
    // fixed DOFs re-clamped to reference.
    const Eigen::VectorXd u_full = Phi_CB * q_r_new;
    const Eigen::VectorXd u_dot_full = Phi_CB * q_r_dot_new;

    MatXRM x_new(n_nodes, 3);
    MatXRM v_new(n_nodes, 3);
    {
        // x = nodes + u (reshape (n_dof,) → (n_nodes, 3))
        Eigen::Map<Eigen::VectorXd> xn_flat(x_new.data(), n_dof);
        Eigen::Map<Eigen::VectorXd> vn_flat(v_new.data(), n_dof);
        xn_flat = x_ref_flat + u_full;
        vn_flat = u_dot_full;
    }

    // Re-clamp fixed nodes exactly at reference, zero velocity there.
    {
        Eigen::Map<Eigen::VectorXd> xn_flat(x_new.data(), n_dof);
        Eigen::Map<Eigen::VectorXd> vn_flat(v_new.data(), n_dof);
        for (int i = 0; i < fixed_dofs.size(); ++i) {
            const int dof = fixed_dofs[i];
            xn_flat[dof] = x_ref_flat[dof];
            vn_flat[dof] = 0.0;
        }
    }

    return std::make_tuple(std::move(x_new), std::move(v_new), q_r_new);
}

std::tuple<MatXRM, MatXRM, Eigen::VectorXd>
cb_step_free(const Eigen::Ref<const MatXRM>& x,
             const Eigen::Ref<const MatXRM>& v,
             const Eigen::Ref<const MatXRM>& x_ref_body,
             const Eigen::Ref<const MatXRM>& Phi_CB,
             const Eigen::Ref<const MatXRM>& M_r,
             const Eigen::Ref<const MatXRM>& A_r_inv,
             const Eigen::Ref<const MatXRM>& C_q,
             const Eigen::Ref<const Eigen::VectorXd>& M_diag,
             const Eigen::Ref<const Eigen::Vector3d>& gravity,
             const Eigen::Ref<const Eigen::VectorXd>& extra_force_flat,
             double dt,
             double damping,
             double m_total) {
    const int n_nodes = static_cast<int>(x.rows());
    const int n_dof = 3 * n_nodes;
    const int n_r = static_cast<int>(M_r.rows());
    const double dt2_inv = 1.0 / (dt * dt);

    // Centroid + Kabsch rotation
    const Eigen::Vector3d t_centroid = x.colwise().mean();
    MatXRM x_centered(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i)
        x_centered.row(i) = x.row(i) - t_centroid.transpose();

    const Eigen::Matrix3d R = kabsch_rotation(x_centered, x_ref_body);

    // Body-frame deformation: u_body = x_centered · R - x_ref_body
    MatXRM u_body(n_nodes, 3);
    u_body.noalias() = x_centered * R - x_ref_body;

    // Mass-weighted reduction (matches Python ``self._C_q @ u_body``
    // — C_q absorbs M_r⁻¹·Phiᵀ·diag(M_diag)). Earlier ports used
    // bare Phiᵀ here which silently dropped the mass weighting; the
    // integration then diverged after a few steps under gravity.
    Eigen::Map<const Eigen::VectorXd> u_body_flat(u_body.data(), n_dof);
    Eigen::VectorXd q_r = C_q * u_body_flat;

    // q_r_dot from rotated velocity (same C_q projection).
    const Eigen::Vector3d t_dot = v.colwise().mean();
    MatXRM v_body(n_nodes, 3);
    for (int i = 0; i < n_nodes; ++i)
        v_body.row(i) = v.row(i) - t_dot.transpose();
    v_body = v_body * R;
    Eigen::Map<const Eigen::VectorXd> v_body_flat(v_body.data(), n_dof);
    Eigen::VectorXd q_r_dot = C_q * v_body_flat;

    // External force (world frame) → body frame via R
    Eigen::VectorXd f_ext = build_gravity_force(M_diag, gravity, extra_force_flat);
    // Reshape (n_dof,) → (n_nodes, 3), rotate, reshape back.
    MatXRM f_ext_3 = Eigen::Map<MatXRM>(f_ext.data(), n_nodes, 3) * R;
    Eigen::Map<Eigen::VectorXd> f_body_flat(f_ext_3.data(), n_dof);
    Eigen::VectorXd f_r = Phi_CB.transpose() * f_body_flat;

    // Implicit Euler (same shape as anchored)
    const Eigen::VectorXd q_r_pred = q_r + dt * q_r_dot;
    Eigen::VectorXd rhs = (M_r * q_r_pred) * dt2_inv + f_r;
    if (damping > 0.0) rhs += (damping / dt) * (M_r * q_r);
    const Eigen::VectorXd q_r_new     = A_r_inv * rhs;
    const Eigen::VectorXd q_r_dot_new = (q_r_new - q_r) / dt;

    // Centroid integration (explicit Euler — total external force)
    Eigen::Vector3d f_total = Eigen::Map<MatXRM>(f_ext.data(), n_nodes, 3).colwise().sum();
    const Eigen::Vector3d t_dot_new = t_dot + dt * f_total / m_total;
    const Eigen::Vector3d t_new     = t_centroid + dt * t_dot_new;

    // Reconstruct: x = t_new + (x_ref_body + Phi·q_r_new) · R^T
    Eigen::VectorXd u_new_flat = Phi_CB * q_r_new;
    Eigen::VectorXd u_dot_new_flat = Phi_CB * q_r_dot_new;
    Eigen::Map<const MatXRM> u_new(u_new_flat.data(), n_nodes, 3);
    Eigen::Map<const MatXRM> u_dot_new(u_dot_new_flat.data(), n_nodes, 3);

    MatXRM x_new(n_nodes, 3);
    MatXRM v_new(n_nodes, 3);
    const Eigen::Matrix3d RT = R.transpose();
    for (int i = 0; i < n_nodes; ++i) {
        x_new.row(i) = t_new.transpose()
                     + (x_ref_body.row(i) + u_new.row(i)) * RT;
        v_new.row(i) = t_dot_new.transpose() + u_dot_new.row(i) * RT;
    }

    return std::make_tuple(std::move(x_new), std::move(v_new), q_r_new);
}

}  // namespace robosim
