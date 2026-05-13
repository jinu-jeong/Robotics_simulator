// RigidSceneStep — substep-level scene orchestration.
// See header for the contract. The inner loop mirrors the Python
// path in ``robosim/scene/runner.py`` substep-by-substep, but each
// kernel call stays inside this translation unit (no Python ↔ C++
// boundary inside ``step()``).

#include "robosim/scene_step.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace robosim {

namespace {

constexpr double GROUND_UP_Z_TOL = 0.0;  // axis-aligned-up if normal == +Z

inline bool ground_is_up_z(const Eigen::Vector3d& n) {
    return n[0] == 0.0 && n[1] == 0.0 && n[2] > GROUND_UP_Z_TOL;
}

// Compose two SE(3) transforms: out = a · b.
inline void compose_T(const Eigen::Matrix3d& Ra, const Eigen::Vector3d& ta,
                      const Eigen::Matrix3d& Rb, const Eigen::Vector3d& tb,
                      Eigen::Matrix3d& Rout, Eigen::Vector3d& tout) {
    Rout.noalias() = Ra * Rb;
    tout.noalias() = Ra * tb + ta;
}

}  // namespace


RigidSceneStep::RigidSceneStep() = default;
RigidSceneStep::~RigidSceneStep() = default;


int RigidSceneStep::add_robot(RbdTopology* topo,
                              const Eigen::Ref<const Eigen::Vector3d>& gravity,
                              bool is_free) {
    RobotEntry r;
    r.topo = topo;
    r.n_links = topo->n_links;
    r.n_dof = topo->n_dof;
    r.q  = Eigen::VectorXd::Zero(r.n_dof);
    r.qd = Eigen::VectorXd::Zero(r.n_dof);
    r.tau = Eigen::VectorXd::Zero(r.n_dof);
    r.gravity = gravity;
    r.is_free = is_free;

    r.R_flat = MatRMXd::Zero(r.n_links, 9);
    r.t_arr  = MatRMXd::Zero(r.n_links, 3);
    r.omega  = MatRMXd::Zero(r.n_links, 3);
    r.v_orig = MatRMXd::Zero(r.n_links, 3);
    r.f_ext_flat = Eigen::VectorXd::Zero(r.n_links * 6);

    robots_.push_back(std::move(r));
    return static_cast<int>(robots_.size()) - 1;
}

void RigidSceneStep::set_pd_control(int idx,
        const Eigen::Ref<const Eigen::VectorXd>& kp,
        const Eigen::Ref<const Eigen::VectorXd>& kd) {
    auto& r = robots_.at(idx);
    r.pd_enabled = true;
    r.kp = kp;
    r.kd = kd;
    if (r.q_target.size() == 0) r.q_target = Eigen::VectorXd::Zero(r.n_dof);
}

void RigidSceneStep::set_q_target(int idx,
        const Eigen::Ref<const Eigen::VectorXd>& q_target) {
    robots_.at(idx).q_target = q_target;
}

void RigidSceneStep::set_tau(int idx,
        const Eigen::Ref<const Eigen::VectorXd>& tau) {
    robots_.at(idx).tau = tau;
}

void RigidSceneStep::set_state(int idx,
        const Eigen::Ref<const Eigen::VectorXd>& q,
        const Eigen::Ref<const Eigen::VectorXd>& qd) {
    auto& r = robots_.at(idx);
    r.q  = q;
    r.qd = qd;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
RigidSceneStep::get_state(int idx) const {
    const auto& r = robots_.at(idx);
    return {r.q, r.qd};
}

int RigidSceneStep::add_box(int robot_idx, int link_idx,
        const Eigen::Ref<const Eigen::Vector3d>& half_extents,
        const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
        const Eigen::Ref<const Eigen::Vector3d>& origin_t) {
    CollisionBodyEntry b;
    b.robot_idx = robot_idx;
    b.link_idx  = link_idx;
    b.body_id   = next_body_id_++;
    b.geom      = ColGeom::BOX;
    b.dim       = half_extents;
    b.origin_R  = origin_R;
    b.origin_t  = origin_t;
    bodies_.push_back(std::move(b));
    return bodies_.back().body_id;
}

int RigidSceneStep::add_sphere(int robot_idx, int link_idx, double radius,
        const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
        const Eigen::Ref<const Eigen::Vector3d>& origin_t) {
    CollisionBodyEntry b;
    b.robot_idx = robot_idx;
    b.link_idx  = link_idx;
    b.body_id   = next_body_id_++;
    b.geom      = ColGeom::SPHERE;
    b.dim       = Eigen::Vector3d(radius, 0, 0);
    b.origin_R  = origin_R;
    b.origin_t  = origin_t;
    bodies_.push_back(std::move(b));
    return bodies_.back().body_id;
}

int RigidSceneStep::add_cylinder(int robot_idx, int link_idx,
        double radius, double length,
        const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
        const Eigen::Ref<const Eigen::Vector3d>& origin_t) {
    CollisionBodyEntry b;
    b.robot_idx = robot_idx;
    b.link_idx  = link_idx;
    b.body_id   = next_body_id_++;
    b.geom      = ColGeom::CYLINDER;
    b.dim       = Eigen::Vector3d(radius, length, 0);
    b.origin_R  = origin_R;
    b.origin_t  = origin_t;
    bodies_.push_back(std::move(b));
    return bodies_.back().body_id;
}

void RigidSceneStep::add_filter(int a, int b) {
    filters_.insert({std::min(a, b), std::max(a, b)});
}

void RigidSceneStep::set_ground(double h,
        const Eigen::Ref<const Eigen::Vector3d>& normal) {
    ground_h_ = h;
    ground_n_ = normal;
    if (ground_n_.norm() > 0) ground_n_.normalize();
}

void RigidSceneStep::set_contact_params(double k, double c, double mu,
                                        double eps, double max_pen) {
    k_ = k; c_ = c; mu_ = mu; fric_eps_ = eps; max_pen_ = max_pen;
}


void RigidSceneStep::apply_pd_to_tau_(int idx, Eigen::VectorXd& tau_out) {
    const auto& r = robots_[idx];
    if (!r.pd_enabled) return;
    // τ_pd = kp · (q_target − q) − kd · qd. Componentwise.
    for (int i = 0; i < r.n_dof; ++i) {
        const double err = r.q_target[i] - r.q[i];
        tau_out[i] += r.kp[i] * err - r.kd[i] * r.qd[i];
    }
}


void RigidSceneStep::update_body_transforms_() {
    body_t_world_.resize(bodies_.size());
    body_R_world_.resize(bodies_.size());

    for (size_t i = 0; i < bodies_.size(); ++i) {
        const auto& b = bodies_[i];
        const auto& r = robots_[b.robot_idx];
        // Pull link world transform from packed FK output.
        Eigen::Matrix3d R_link;
        const double* rp = r.R_flat.row(b.link_idx).data();
        R_link << rp[0], rp[1], rp[2],
                  rp[3], rp[4], rp[5],
                  rp[6], rp[7], rp[8];
        Eigen::Vector3d t_link = r.t_arr.row(b.link_idx).transpose();

        // body_T_world = link_T_world ∘ origin
        compose_T(R_link, t_link, b.origin_R, b.origin_t,
                  body_R_world_[i], body_t_world_[i]);
    }
}


void RigidSceneStep::compute_body_aabbs_() {
    body_aabb_min_.resize(bodies_.size());
    body_aabb_max_.resize(bodies_.size());

    for (size_t i = 0; i < bodies_.size(); ++i) {
        const auto& b = bodies_[i];
        const auto& R = body_R_world_[i];
        const auto& t = body_t_world_[i];
        Eigen::Vector3d extent;

        if (b.geom == ColGeom::SPHERE) {
            const double r = b.dim[0];
            extent = Eigen::Vector3d(r, r, r);
        } else if (b.geom == ColGeom::BOX) {
            // OBB world-AABB: extent_i = sum_j |R_ij| · he_j
            for (int ax = 0; ax < 3; ++ax) {
                extent[ax] = std::abs(R(ax, 0)) * b.dim[0]
                           + std::abs(R(ax, 1)) * b.dim[1]
                           + std::abs(R(ax, 2)) * b.dim[2];
            }
        } else {  // CYLINDER
            const double r = b.dim[0], hl = 0.5 * b.dim[1];
            Eigen::Vector3d axis = R.col(2);  // symmetry axis in world
            for (int ax = 0; ax < 3; ++ax) {
                const double c2 = axis[ax] * axis[ax];
                extent[ax] = hl * std::abs(axis[ax])
                           + r * std::sqrt(std::max(0.0, 1.0 - c2));
            }
        }

        body_aabb_min_[i] = t - extent;
        body_aabb_max_[i] = t + extent;
    }
}


// Per-contact info collected by detection — kept inline for cache locality.
struct ContactRec {
    Eigen::Vector3d pa, pb, n;   // point on A, point on B, normal B→A (world)
    double pen;
    int body_a;                  // index in bodies_ vector
    int body_b;                  // -1 ⇒ ground
};


// Compute world-frame point velocity of a point attached to a link.
inline Eigen::Vector3d link_point_velocity(const RobotEntry& r,
                                            int link_idx,
                                            const Eigen::Vector3d& point_world) {
    Eigen::Vector3d omega = r.omega.row(link_idx).transpose();
    Eigen::Vector3d v_o   = r.v_orig.row(link_idx).transpose();
    Eigen::Vector3d t_lk  = r.t_arr.row(link_idx).transpose();
    Eigen::Vector3d d = point_world - t_lk;
    return v_o + omega.cross(d);
}


void RigidSceneStep::run_one_substep_(double dt) {
    // ── 1) PD + FK + link velocities per robot ──
    for (size_t r_idx = 0; r_idx < robots_.size(); ++r_idx) {
        auto& r = robots_[r_idx];

        // Effective tau = baseline + PD.
        Eigen::VectorXd tau_eff = r.tau;
        apply_pd_to_tau_(static_cast<int>(r_idx), tau_eff);
        // Stash effective tau in r.tau for use by ABA later in this
        // substep (we don't mutate the baseline; this is the substep-
        // resolved value, which becomes the dynamics input).
        // Actually: keep baseline intact and use tau_eff locally.

        // FK
        forward_kinematics(*r.topo, r.q, r.R_flat, r.t_arr);

        // link velocities
        link_world_velocities(*r.topo, r.qd, r.R_flat, r.t_arr,
                              r.omega, r.v_orig);

        // tau_eff stored for ABA — we'll reuse via shadow storage:
        // overwrite r.tau here is fine when PD is enabled because the
        // user-supplied tau gets re-set per outer step anyway.
        if (r.pd_enabled) r.tau = tau_eff;
    }

    // ── 2) Update collision body transforms + AABBs ──
    update_body_transforms_();
    compute_body_aabbs_();

    // ── 3) Detect contacts (ground + body-body, brute force) ──
    std::vector<ContactRec> contacts;
    contacts.reserve(64);

    const bool have_ground = ground_n_.norm() > 0;
    const bool ground_up_z = have_ground && ground_is_up_z(ground_n_);

    for (size_t i = 0; i < bodies_.size(); ++i) {
        if (!have_ground) break;
        if (ground_up_z && body_aabb_min_[i][2] > ground_h_) continue;
        const auto& b = bodies_[i];
        ContactBuf buf(0, 10);
        if (b.geom == ColGeom::SPHERE) {
            buf = sphere_ground(body_t_world_[i], b.dim[0],
                                 ground_h_, ground_n_);
        } else if (b.geom == ColGeom::BOX) {
            buf = box_ground(body_t_world_[i], body_R_world_[i], b.dim,
                              ground_h_, ground_n_);
        } else if (b.geom == ColGeom::CYLINDER) {
            buf = cylinder_ground(body_t_world_[i], body_R_world_[i],
                                   b.dim[0], 0.5 * b.dim[1],
                                   ground_h_, ground_n_, /*n_ring=*/8);
        }
        for (int row = 0; row < buf.rows(); ++row) {
            ContactRec cr;
            cr.pa = buf.row(row).segment<3>(0).transpose();
            cr.pb = buf.row(row).segment<3>(3).transpose();
            cr.n  = buf.row(row).segment<3>(6).transpose();
            cr.pen = buf(row, 9);
            cr.body_a = static_cast<int>(i);
            cr.body_b = -1;
            contacts.push_back(cr);
        }
    }

    // Body-body
    const int N = static_cast<int>(bodies_.size());
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const int bid_a = bodies_[i].body_id;
            const int bid_b = bodies_[j].body_id;
            const std::pair<int,int> key{std::min(bid_a, bid_b), std::max(bid_a, bid_b)};
            if (filters_.count(key)) continue;

            // AABB overlap
            const auto& amn = body_aabb_min_[i]; const auto& amx = body_aabb_max_[i];
            const auto& bmn = body_aabb_min_[j]; const auto& bmx = body_aabb_max_[j];
            if (amx[0] < bmn[0] || bmx[0] < amn[0]) continue;
            if (amx[1] < bmn[1] || bmx[1] < amn[1]) continue;
            if (amx[2] < bmn[2] || bmx[2] < amn[2]) continue;

            const auto& ba = bodies_[i];
            const auto& bb = bodies_[j];

            ContactBuf buf(0, 10);
            // We only narrow-phase the geometry pairs the Python ref
            // supports: box-box, box-sphere, sphere-sphere.
            if (ba.geom == ColGeom::BOX && bb.geom == ColGeom::BOX) {
                buf = box_box(body_t_world_[i], body_R_world_[i], ba.dim,
                               body_t_world_[j], body_R_world_[j], bb.dim);
            } else if (ba.geom == ColGeom::SPHERE && bb.geom == ColGeom::SPHERE) {
                buf = sphere_sphere(body_t_world_[i], ba.dim[0],
                                     body_t_world_[j], bb.dim[0]);
            } else if (ba.geom == ColGeom::BOX && bb.geom == ColGeom::SPHERE) {
                buf = box_sphere(body_t_world_[i], body_R_world_[i], ba.dim,
                                  body_t_world_[j], bb.dim[0]);
            } else if (ba.geom == ColGeom::SPHERE && bb.geom == ColGeom::BOX) {
                buf = box_sphere(body_t_world_[j], body_R_world_[j], bb.dim,
                                  body_t_world_[i], ba.dim[0]);
                // Flip: normal points B→A in box_sphere(box, sphere), but
                // here body_i is the sphere. We want normal pointing
                // from j (box) → i (sphere) which IS the swap convention.
                // Actually: box_sphere returns normal from box→sphere.
                // For our pair (a=sphere, b=box), normal B→A == box→sphere,
                // which matches what's returned. Swap point_a/point_b
                // so cp.point_a sits on body_a (sphere).
                for (int row = 0; row < buf.rows(); ++row) {
                    Eigen::Vector3d pa = buf.row(row).segment<3>(0).transpose();
                    Eigen::Vector3d pb = buf.row(row).segment<3>(3).transpose();
                    buf.row(row).segment<3>(0) = pb.transpose();
                    buf.row(row).segment<3>(3) = pa.transpose();
                }
            }
            // (cylinder narrow-phase pairs not implemented in C++ —
            // expected to be filtered out via finger adjacency.)

            for (int row = 0; row < buf.rows(); ++row) {
                ContactRec cr;
                cr.pa = buf.row(row).segment<3>(0).transpose();
                cr.pb = buf.row(row).segment<3>(3).transpose();
                cr.n  = buf.row(row).segment<3>(6).transpose();
                cr.pen = buf(row, 9);
                cr.body_a = i;
                cr.body_b = j;
                contacts.push_back(cr);
            }
        }
    }

    // ── 4) Compute per-contact forces, accumulate per-(robot, link) wrenches ──
    for (auto& r : robots_) r.f_ext_flat.setZero();

    // Two-pass: count contacts per (robot, link) so effective_mass is
    // shared the same way as the Python penalty solver.
    std::vector<int> per_robot_link_count(0);
    // Flatten index: robot_idx * max_links + link_idx is unwieldy when
    // robots have different n_links; use a parallel vector mapping
    // global linear index built per substep.
    // Simpler: walk contacts twice — first to count, second to compute.

    // count
    struct ContactSide {
        int robot, link;
    };
    auto get_side = [&](int body_idx, bool side_a, const ContactRec&) {
        ContactSide s;
        const auto& b = bodies_[body_idx];
        s.robot = b.robot_idx;
        s.link  = b.link_idx;
        (void)side_a;
        return s;
    };

    // Count contacts per (robot, link).
    std::vector<std::vector<int>> cnt(robots_.size());
    for (size_t i = 0; i < robots_.size(); ++i)
        cnt[i].assign(robots_[i].n_links, 0);

    for (const auto& c : contacts) {
        auto sa = get_side(c.body_a, true,  c);
        cnt[sa.robot][sa.link]++;
        if (c.body_b >= 0) {
            auto sb = get_side(c.body_b, false, c);
            cnt[sb.robot][sb.link]++;
        }
    }

    // Compute forces & accumulate.
    for (const auto& c : contacts) {
        // body_a side
        auto sa = get_side(c.body_a, true, c);
        auto& ra = robots_[sa.robot];

        Eigen::Vector3d v_a = link_point_velocity(ra, sa.link, c.pa);
        Eigen::Vector3d v_b = Eigen::Vector3d::Zero();
        if (c.body_b >= 0) {
            auto sb = get_side(c.body_b, false, c);
            const auto& rb = robots_[sb.robot];
            v_b = link_point_velocity(rb, sb.link, c.pb);
        }

        // Effective mass = link_mass / n_contacts_at_link. RobotEntry
        // doesn't carry per-link mass directly here — it lives in the
        // RbdTopology's spatial inertias. Just use -1 (no critical-
        // damping cap) for v1; matches Python with effective_mass=None.
        Eigen::VectorXd f = contact_force_penalty(
            c.n, v_a, v_b, c.pen,
            k_, c_, mu_, fric_eps_, max_pen_,
            /*effective_mass=*/-1.0);
        if (f.size() == 0) continue;  // separating fast enough
        Eigen::Vector3d force = f.head<3>();

        // body_a side: convert world force to link-frame spatial wrench.
        // T_inv = (R^T, -R^T t). f_link = R^T · force. p_link = R^T (cp.pa - t).
        // tau_link = p_link × f_link. Wrench is [tau_link; f_link].
        Eigen::Matrix3d R_link_a;
        {
            const double* rp = ra.R_flat.row(sa.link).data();
            R_link_a << rp[0], rp[1], rp[2],
                        rp[3], rp[4], rp[5],
                        rp[6], rp[7], rp[8];
        }
        Eigen::Vector3d t_link_a = ra.t_arr.row(sa.link).transpose();
        Eigen::Vector3d f_link_a = R_link_a.transpose() * force;
        Eigen::Vector3d p_link_a = R_link_a.transpose() * (c.pa - t_link_a);
        Eigen::Vector3d tau_link_a = p_link_a.cross(f_link_a);
        const int base_a = sa.link * 6;
        ra.f_ext_flat.segment<3>(base_a + 0) += tau_link_a;
        ra.f_ext_flat.segment<3>(base_a + 3) += f_link_a;

        // body_b side: Newton's 3rd law — opposite force at cp.point_b.
        if (c.body_b >= 0) {
            auto sb = get_side(c.body_b, false, c);
            auto& rb = robots_[sb.robot];
            Eigen::Vector3d force_b = -force;
            Eigen::Matrix3d R_link_b;
            const double* rp = rb.R_flat.row(sb.link).data();
            R_link_b << rp[0], rp[1], rp[2],
                        rp[3], rp[4], rp[5],
                        rp[6], rp[7], rp[8];
            Eigen::Vector3d t_link_b = rb.t_arr.row(sb.link).transpose();
            Eigen::Vector3d f_link_b = R_link_b.transpose() * force_b;
            Eigen::Vector3d p_link_b = R_link_b.transpose() * (c.pb - t_link_b);
            Eigen::Vector3d tau_link_b = p_link_b.cross(f_link_b);
            const int base_b = sb.link * 6;
            rb.f_ext_flat.segment<3>(base_b + 0) += tau_link_b;
            rb.f_ext_flat.segment<3>(base_b + 3) += f_link_b;
        }
    }

    // ── 5) ABA per robot + semi-implicit Euler ──
    for (auto& r : robots_) {
        Eigen::VectorXd qdd = aba(*r.topo, r.q, r.qd, r.tau,
                                  r.gravity, r.f_ext_flat);
        r.qd.noalias() += dt * qdd;
        r.q .noalias() += dt * r.qd;
    }
}


void RigidSceneStep::step(double dt, int n_substeps) {
    for (int s = 0; s < n_substeps; ++s) run_one_substep_(dt);
}

}  // namespace robosim
