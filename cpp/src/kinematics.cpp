// Forward-kinematics implementation. See header for the contract.

#include "robosim/kinematics.hpp"

#include <Eigen/Dense>

namespace robosim {

namespace {

// joint_motion(joint, q) → (R, t). Compact mirror of
// ``robosim/math/spatial.py::joint_transform`` for the three joint
// kinds the URDF parser produces.
inline void joint_motion(const JointSpec& j, double q,
                         Eigen::Matrix3d& R, Eigen::Vector3d& t) noexcept {
    switch (j.kind) {
        case JointKind::FIXED:
            R.setIdentity();
            t.setZero();
            return;
        case JointKind::REVOLUTE:
            R = from_axis_angle(j.axis, q);
            t.setZero();
            return;
        case JointKind::PRISMATIC:
            R.setIdentity();
            t = j.axis * q;
            return;
    }
    R.setIdentity();
    t.setZero();
}

}  // namespace

// Pack/unpack helpers — R_out stores each link's 3×3 as a flat
// length-9 row in row-major order. Keeping the conversion local
// avoids leaking Eigen::Map specifics into the header.
// R_out is row-major (n_links, 9) — row l is the 3×3 flattened
// row-major. ``row(l).data()`` is contiguous length 9 on the
// row-major layout, so a plain ``Map`` works.
inline Eigen::Matrix3d row_to_mat3(const Eigen::Ref<const MatRMXd>& R_out, int l) {
    const double* p = R_out.row(l).data();
    Eigen::Matrix3d m;
    m << p[0], p[1], p[2],
         p[3], p[4], p[5],
         p[6], p[7], p[8];
    return m;
}

inline void mat3_to_row(Eigen::Ref<MatRMXd> R_out, int l, const Eigen::Matrix3d& m) {
    double* p = R_out.row(l).data();
    p[0] = m(0, 0); p[1] = m(0, 1); p[2] = m(0, 2);
    p[3] = m(1, 0); p[4] = m(1, 1); p[5] = m(1, 2);
    p[6] = m(2, 0); p[7] = m(2, 1); p[8] = m(2, 2);
}

void forward_kinematics(const Topology& topo,
                        const Eigen::Ref<const Eigen::VectorXd>& q,
                        Eigen::Ref<MatRMXd> R_out,
                        Eigen::Ref<MatRMXd> t_out) {
    const int n_links = topo.n_links;

    // Seed every link with identity — joints later overwrite child
    // rows, but orphan/unreached links stay at identity (matches the
    // Python reference's default).
    const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    for (int l = 0; l < n_links; ++l) {
        mat3_to_row(R_out, l, I3);
        t_out.row(l).setZero();
    }

    Eigen::Matrix3d Rj, R_local, Rp, R_child;
    Eigen::Vector3d tj, t_local, tp, t_child;
    for (const auto& j : topo.joints) {
        const double qj = (j.dof_index >= 0) ? q[j.dof_index] : 0.0;
        joint_motion(j, qj, Rj, tj);

        // T_local = origin · joint_motion
        compose_Rt(j.origin_R, j.origin_t, Rj, tj, R_local, t_local);

        // Read parent transform (-1 = root → identity, which is what
        // the seeded rows hold; we still go through the same row-fetch
        // path to keep the inner loop straight-line).
        if (j.parent_link >= 0) {
            Rp = row_to_mat3(R_out, j.parent_link);
            tp = t_out.row(j.parent_link).transpose();
        } else {
            Rp.setIdentity();
            tp.setZero();
        }

        compose_Rt(Rp, tp, R_local, t_local, R_child, t_child);

        mat3_to_row(R_out, j.child_link, R_child);
        t_out.row(j.child_link) = t_child.transpose();
    }
}


void link_world_velocities(const RbdTopology& topo,
                           const Eigen::Ref<const Eigen::VectorXd>& qd,
                           const Eigen::Ref<const MatRMXd>& R_flat,
                           const Eigen::Ref<const MatRMXd>& t_arr,
                           Eigen::Ref<MatRMXd> omega_out,
                           Eigen::Ref<MatRMXd> v_out) {
    const int n_links = topo.n_links;
    omega_out.setZero();
    v_out.setZero();

    // tree_order: BFS from roots excluding roots → parent always
    // processed before child (roots already have zero rows).
    for (int child_idx : topo.tree_order) {
        const int parent_idx = topo.parent_link[child_idx];
        if (parent_idx < 0) continue;

        Eigen::Vector3d omega_p = omega_out.row(parent_idx).transpose();
        Eigen::Vector3d v_p     = v_out.row(parent_idx).transpose();
        Eigen::Vector3d origin_p = t_arr.row(parent_idx).transpose();
        Eigen::Vector3d origin_c = t_arr.row(child_idx).transpose();
        Eigen::Vector3d d = origin_c - origin_p;
        Eigen::Vector3d v_c = v_p + omega_p.cross(d);
        Eigen::Vector3d omega_c = omega_p;

        const int j_idx = topo.joint_for_link[child_idx];
        if (j_idx >= 0) {
            const JointSpec& joint = topo.joints[j_idx];
            if (joint.kind != JointKind::FIXED) {
                const double qd_j = qd[joint.dof_index];
                Eigen::Matrix3d R_c = row_to_mat3(R_flat, child_idx);
                Eigen::Vector3d axis_world = R_c * joint.axis;
                if (joint.kind == JointKind::REVOLUTE)
                    omega_c = omega_p + qd_j * axis_world;
                else if (joint.kind == JointKind::PRISMATIC)
                    v_c = v_c + qd_j * axis_world;
            }
        }
        omega_out.row(child_idx) = omega_c.transpose();
        v_out    .row(child_idx) = v_c.transpose();
    }
}


Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
batch_point_velocities(const Eigen::Ref<const Eigen::VectorXi>& link_indices,
                       const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& points,
                       const Eigen::Ref<const MatRMXd>& omega_arr,
                       const Eigen::Ref<const MatRMXd>& v_origin_arr,
                       const Eigen::Ref<const MatRMXd>& t_arr) {
    const int N = static_cast<int>(link_indices.size());
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> out(N, 3);
    for (int i = 0; i < N; ++i) {
        const int li = link_indices[i];
        Eigen::Vector3d omega   = omega_arr   .row(li).transpose();
        Eigen::Vector3d v_orig  = v_origin_arr.row(li).transpose();
        Eigen::Vector3d origin  = t_arr       .row(li).transpose();
        Eigen::Vector3d pt      = points      .row(i).transpose();
        Eigen::Vector3d r = pt - origin;
        Eigen::Vector3d v = v_orig + omega.cross(r);
        out.row(i) = v.transpose();
    }
    return out;
}

}  // namespace robosim
