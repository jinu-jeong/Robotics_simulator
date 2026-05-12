// ABA + gravity_torques — direct port of the NumPy algorithms with
// per-link Eigen 6×6 / 6-vector storage. Per-step state is laid out
// in std::vector<...> so the inner loops are straight-line Eigen
// expressions; no allocations after the size-determined-by-topology
// reservations at the top.

#include "robosim/rbd.hpp"

#include <Eigen/Dense>
#include <vector>

namespace robosim {

namespace {

// Build the 6×6 spatial-motion-transform matrix X(T) for ``T``: the
// "joint local transform inverse" form used in Featherstone's
// algorithms (v_child = X_joint · v_parent + S · qd).
//
// Layout:
//   X = | R        0 |
//       | -R [p]x  R |
// where R = T.rotation and p = T.translation (translation given in
// the parent frame). [p]x is the skew-symmetric matrix of p.
//
// This is the same as ``_spatial_transform_matrix`` in the Python
// reference but built directly on the inverse: instead of computing
// X(T) then inverting, we use the closed-form inverse:
//   T_inv has rotation R^T, translation -R^T·p.
//
// The Python reference passes ``X = _spatial_transform_matrix(jlt.inverse())``
// — so the matrix below already corresponds to that inverse form.
inline Mat6 spatial_transform_inv(const Mat3& R, const Vec3& t) noexcept {
    // T_inv: R' = R^T, p' = -R^T · t.
    const Mat3 Rt = R.transpose();
    const Vec3 p_inv = -Rt * t;
    // X(T_inv): build directly.
    Mat6 X = Mat6::Zero();
    X.block<3, 3>(0, 0) = Rt;
    X.block<3, 3>(3, 3) = Rt;
    // lower-left = -Rt · [p_inv]x , equivalently = [p_inv]x_postmul · Rt.
    // Use the inlined form from the Python ref:
    //   X[3+i, 0:3] = (-p_inv x R_inv-row(i)) effectively.
    // Simpler: use Eigen's skew via cross-product of basis vectors.
    Mat3 px;
    px <<  0.0,        -p_inv.z(),  p_inv.y(),
           p_inv.z(),   0.0,       -p_inv.x(),
          -p_inv.y(),   p_inv.x(),  0.0;
    X.block<3, 3>(3, 0) = px * Rt;
    return X;
}

// 6×6 motion-cross operator [v]x for ω in v[0:3], v_lin in v[3:6].
inline Mat6 spatial_cross_motion(const Vec6& v) noexcept {
    Mat6 X = Mat6::Zero();
    const double w0 = v[0], w1 = v[1], w2 = v[2];
    const double l0 = v[3], l1 = v[4], l2 = v[5];
    // [ω]x blocks (top-left, bottom-right)
    X(0, 1) = -w2; X(0, 2) =  w1;
    X(1, 0) =  w2; X(1, 2) = -w0;
    X(2, 0) = -w1; X(2, 1) =  w0;
    X(3, 4) = -w2; X(3, 5) =  w1;
    X(4, 3) =  w2; X(4, 5) = -w0;
    X(5, 3) = -w1; X(5, 4) =  w0;
    // [v_lin]x block (bottom-left)
    X(3, 1) = -l2; X(3, 2) =  l1;
    X(4, 0) =  l2; X(4, 2) = -l0;
    X(5, 0) = -l1; X(5, 1) =  l0;
    return X;
}

inline Mat6 spatial_cross_force(const Vec6& v) noexcept {
    return -spatial_cross_motion(v).transpose();
}

// Motion subspace S (6-vector) for the joint kinds we support.
// FIXED → S is empty; we encode that as the zero vector and skip
// any S-projection terms in the algorithm.
inline Vec6 motion_subspace(const JointSpec& j) noexcept {
    Vec6 s = Vec6::Zero();
    switch (j.kind) {
        case JointKind::REVOLUTE:
            s.head<3>() = j.axis;
            return s;
        case JointKind::PRISMATIC:
            s.tail<3>() = j.axis;
            return s;
        case JointKind::FIXED:
        default:
            return s;   // zero — caller treats has_dof = false
    }
}

// Unpack one row of the inertia buffer into a 6×6 matrix. The buffer
// is row-major (n_links × 36), so the data pointer is contiguous.
inline Mat6 link_inertia(const RbdTopology& topo, int link) noexcept {
    Mat6 I;
    const double* p = topo.link_inertias.row(link).data();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            I(i, j) = p[i * 6 + j];
    return I;
}

// Compute the per-joint local (origin × joint-motion) transform's
// "X(T^-1)" 6×6 spatial transform. The Python reference caches these
// across calls — the C++ kernel recomputes them per call (cheap; the
// Python overhead they were dodging doesn't apply here).
inline Mat6 joint_X_inverse(const JointSpec& j, double qj) noexcept {
    // T_local = origin · joint_motion(qj)
    Mat3 Rj, R_local;
    Vec3 tj, t_local;
    switch (j.kind) {
        case JointKind::FIXED:
            Rj.setIdentity(); tj.setZero(); break;
        case JointKind::REVOLUTE:
            Rj = from_axis_angle(j.axis, qj); tj.setZero(); break;
        case JointKind::PRISMATIC:
            Rj.setIdentity(); tj = j.axis * qj; break;
    }
    compose_Rt(j.origin_R, j.origin_t, Rj, tj, R_local, t_local);
    return spatial_transform_inv(R_local, t_local);
}

}  // namespace

Eigen::VectorXd aba(const RbdTopology& topo,
                    const Eigen::Ref<const Eigen::VectorXd>& q,
                    const Eigen::Ref<const Eigen::VectorXd>& qd,
                    const Eigen::Ref<const Eigen::VectorXd>& tau,
                    const Eigen::Ref<const Eigen::Vector3d>& gravity,
                    const Eigen::Ref<const Eigen::VectorXd>& f_ext_flat) {
    const int n_links = topo.n_links;
    const int n_dof   = topo.n_dof;

    // ── Per-link scratch (size known at entry, allocated once) ──────
    std::vector<Vec6> v(n_links, Vec6::Zero());
    std::vector<Vec6> c(n_links, Vec6::Zero());
    std::vector<Vec6> p_A(n_links, Vec6::Zero());
    std::vector<Vec6> a(n_links, Vec6::Zero());
    std::vector<Mat6> I_A(n_links, Mat6::Zero());
    std::vector<Mat6> X_parent(n_links, Mat6::Identity());
    std::vector<Vec6> U(n_links, Vec6::Zero());
    std::vector<double> d(n_links, 0.0);
    std::vector<double> u_acc(n_links, 0.0);
    std::vector<Vec6> S(n_links, Vec6::Zero());
    std::vector<bool> has_dof(n_links, false);
    std::vector<int>  dof_offset(n_links, -1);

    // Seed link inertias.
    for (int l = 0; l < n_links; ++l) I_A[l] = link_inertia(topo, l);

    // Spatial gravity in the world frame (rooted convention used by
    // the Python reference): a_grav = (0, -g).
    Vec6 a_grav = Vec6::Zero();
    a_grav[3] = -gravity[0];
    a_grav[4] = -gravity[1];
    a_grav[5] = -gravity[2];

    // ── Pass 1: forward (velocities, bias forces) ──────────────────
    for (int link : topo.tree_order) {
        const int parent = topo.parent_link[link];
        const int j_idx  = topo.joint_for_link[link];
        const JointSpec& js = topo.joints[j_idx];

        const double qj  = (js.dof_index >= 0) ? q[js.dof_index] : 0.0;
        const double qdj = (js.dof_index >= 0) ? qd[js.dof_index] : 0.0;
        S[link]          = motion_subspace(js);
        has_dof[link]    = (js.dof_index >= 0);
        dof_offset[link] = js.dof_index;

        X_parent[link] = joint_X_inverse(js, qj);

        v[link].noalias() = X_parent[link] * v[parent >= 0 ? parent : link];
        if (parent < 0) v[link].setZero();
        if (has_dof[link]) v[link].noalias() += S[link] * qdj;

        c[link].setZero();
        if (has_dof[link])
            c[link].noalias() = spatial_cross_motion(v[link]) * (S[link] * qdj);

        p_A[link].noalias() = spatial_cross_force(v[link]) * (I_A[link] * v[link]);

        // External wrench on this link (link frame, in the same
        // ``[ω, v]`` convention as v / p_A). Subtract — matches the
        // Python reference's ``p_A[i] -= f_ext[i]``.
        if (f_ext_flat.size() >= 6 * (link + 1)) {
            for (int k = 0; k < 6; ++k)
                p_A[link][k] -= f_ext_flat[6 * link + k];
        }
    }

    // ── Pass 2: backward (articulated inertias / bias forces) ───────
    for (auto it = topo.tree_order.rbegin(); it != topo.tree_order.rend(); ++it) {
        const int link   = *it;
        const int parent = topo.parent_link[link];

        Mat6 I_a;
        Vec6 p_a;
        if (has_dof[link]) {
            U[link].noalias() = I_A[link] * S[link];
            d[link]           = S[link].dot(U[link]);
            u_acc[link]       = tau[dof_offset[link]] - S[link].dot(p_A[link]);

            if (std::abs(d[link]) > 1e-15) {
                I_a.noalias() = I_A[link] - (U[link] * U[link].transpose()) / d[link];
                p_a.noalias() = p_A[link] + I_a * c[link]
                              + U[link] * (u_acc[link] / d[link]);
            } else {
                I_a = I_A[link];
                p_a.noalias() = p_A[link] + I_a * c[link];
            }
        } else {
            I_a = I_A[link];
            p_a.noalias() = p_A[link] + I_a * c[link];
        }

        if (parent >= 0) {
            const Mat6& Xp = X_parent[link];
            I_A[parent].noalias() += Xp.transpose() * I_a * Xp;
            p_A[parent].noalias() += Xp.transpose() * p_a;
        }
    }

    // ── Pass 3: forward (accelerations) ────────────────────────────
    // Root acceleration = -gravity (so gravity appears as an inertial
    // bias force when this is pulled down the chain).
    for (int l = 0; l < n_links; ++l)
        if (topo.parent_link[l] == -1) a[l] = a_grav;

    Eigen::VectorXd qdd = Eigen::VectorXd::Zero(n_dof);
    for (int link : topo.tree_order) {
        const int parent = topo.parent_link[link];
        const Vec6 a_prime = X_parent[link] * a[parent] + c[link];

        if (has_dof[link] && std::abs(d[link]) > 1e-15) {
            const double qdd_i = (u_acc[link] - U[link].dot(a_prime)) / d[link];
            qdd[dof_offset[link]] = qdd_i;
            a[link] = a_prime + S[link] * qdd_i;
        } else {
            a[link] = a_prime;
        }
    }

    return qdd;
}

Eigen::VectorXd gravity_torques(const RbdTopology& topo,
                                const Eigen::Ref<const Eigen::VectorXd>& q,
                                const Eigen::Ref<const Eigen::Vector3d>& gravity) {
    // Specialised RNEA(q, 0, 0, g): no Coriolis, no acceleration —
    // only the gravity bias propagated forward, force projected back.
    const int n_links = topo.n_links;
    const int n_dof   = topo.n_dof;

    std::vector<Mat6> X_parent(n_links, Mat6::Identity());
    std::vector<Vec6> a(n_links, Vec6::Zero());
    std::vector<Vec6> f(n_links, Vec6::Zero());
    std::vector<Vec6> S(n_links, Vec6::Zero());
    std::vector<bool> has_dof(n_links, false);
    std::vector<int>  dof_offset(n_links, -1);

    Vec6 a_grav = Vec6::Zero();
    a_grav[3] = -gravity[0];
    a_grav[4] = -gravity[1];
    a_grav[5] = -gravity[2];

    // Pass 1: forward. a_i = X · a_parent (root = -g). f_i = I_i · a_i.
    for (int l = 0; l < n_links; ++l)
        if (topo.parent_link[l] == -1) a[l] = a_grav;

    for (int link : topo.tree_order) {
        const int parent = topo.parent_link[link];
        const int j_idx  = topo.joint_for_link[link];
        const JointSpec& js = topo.joints[j_idx];

        const double qj  = (js.dof_index >= 0) ? q[js.dof_index] : 0.0;
        S[link]          = motion_subspace(js);
        has_dof[link]    = (js.dof_index >= 0);
        dof_offset[link] = js.dof_index;

        X_parent[link] = joint_X_inverse(js, qj);
        a[link].noalias() = X_parent[link] * a[parent >= 0 ? parent : link];
        if (parent < 0) a[link].setZero();   // root has no inbound
        // Root really shouldn't be in tree_order, but safe-guard.

        f[link].noalias() = link_inertia(topo, link) * a[link];
    }

    // Pass 2: backward. tau_i = S^T · f_i, then accumulate into parent.
    Eigen::VectorXd tau = Eigen::VectorXd::Zero(n_dof);
    for (auto it = topo.tree_order.rbegin(); it != topo.tree_order.rend(); ++it) {
        const int link   = *it;
        const int parent = topo.parent_link[link];

        if (has_dof[link]) tau[dof_offset[link]] = S[link].dot(f[link]);

        if (parent >= 0)
            f[parent].noalias() += X_parent[link].transpose() * f[link];
    }
    return tau;
}

}  // namespace robosim
