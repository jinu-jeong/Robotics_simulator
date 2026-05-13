// Contact narrow-phase kernels (C++ port of robosim/physics/contact/sdf.py).
//
// Mirrors the Python reference *exactly* in numerical logic so the
// existing Python tests stay valid. Anything that diverges (different
// thresholds, different tie-break order in SAT, ...) would surface as
// a parity test failure.

#include "robosim/contact.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace robosim {

namespace {

constexpr double GROUND_UP_Z[3] = {0.0, 0.0, 1.0};

inline bool is_up_z(const Eigen::Ref<const Eigen::Vector3d>& n) {
    return n[0] == 0.0 && n[1] == 0.0 && n[2] > 0.0;
}

inline void emit_row(ContactBuf& out, int row,
                     const Eigen::Vector3d& pa,
                     const Eigen::Vector3d& pb,
                     const Eigen::Vector3d& n,
                     double pen) {
    out(row, 0) = pa[0]; out(row, 1) = pa[1]; out(row, 2) = pa[2];
    out(row, 3) = pb[0]; out(row, 4) = pb[1]; out(row, 5) = pb[2];
    out(row, 6) = n[0];  out(row, 7) = n[1];  out(row, 8) = n[2];
    out(row, 9) = pen;
}

}  // namespace


ContactBuf sphere_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                         double radius,
                         double ground_height,
                         const Eigen::Ref<const Eigen::Vector3d>& ground_normal) {
    Eigen::Vector3d n = ground_normal;
    const double nrm = n.norm();
    if (nrm > 0) n /= nrm;

    const double dist = center.dot(n) - ground_height;
    const double pen  = radius - dist;
    if (pen <= 0.0) return ContactBuf(0, 10);

    ContactBuf out(1, 10);
    Eigen::Vector3d pa = center - radius * n;
    Eigen::Vector3d pb = center - dist   * n;
    emit_row(out, 0, pa, pb, n, pen);
    return out;
}


ContactBuf sphere_sphere(const Eigen::Ref<const Eigen::Vector3d>& c1, double r1,
                         const Eigen::Ref<const Eigen::Vector3d>& c2, double r2) {
    Eigen::Vector3d diff = c1 - c2;
    const double dist = diff.norm();
    const double pen  = r1 + r2 - dist;
    if (pen <= 0.0 || dist < 1e-12) return ContactBuf(0, 10);

    Eigen::Vector3d n = diff / dist;
    ContactBuf out(1, 10);
    Eigen::Vector3d pa = c1 - r1 * n;
    Eigen::Vector3d pb = c2 + r2 * n;
    emit_row(out, 0, pa, pb, n, pen);
    return out;
}


ContactBuf box_sphere(const Eigen::Ref<const Eigen::Vector3d>& box_center,
                      const Eigen::Ref<const Mat3RM>& box_rot,
                      const Eigen::Ref<const Eigen::Vector3d>& box_half,
                      const Eigen::Ref<const Eigen::Vector3d>& sphere_center,
                      double sphere_radius) {
    Eigen::Vector3d local = box_rot.transpose() * (sphere_center - box_center);
    Eigen::Vector3d closest_local;
    for (int i = 0; i < 3; ++i)
        closest_local[i] = std::clamp(local[i], -box_half[i], box_half[i]);

    Eigen::Vector3d closest_world = box_rot * closest_local + box_center;
    Eigen::Vector3d diff = sphere_center - closest_world;
    const double dist = diff.norm();
    if (dist > sphere_radius || dist < 1e-12) return ContactBuf(0, 10);

    Eigen::Vector3d n = diff / dist;
    const double pen  = sphere_radius - dist;
    Eigen::Vector3d pa = closest_world;
    Eigen::Vector3d pb = sphere_center - sphere_radius * n;
    ContactBuf out(1, 10);
    emit_row(out, 0, pa, pb, n, pen);
    return out;
}


ContactBuf box_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                      const Eigen::Ref<const Mat3RM>& rotation,
                      const Eigen::Ref<const Eigen::Vector3d>& half_extents,
                      double ground_height,
                      const Eigen::Ref<const Eigen::Vector3d>& ground_normal) {
    // Eight box corners in world frame.
    static const double SIGNS[8][3] = {
        {-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},
        {-1,-1, 1},{1,-1, 1},{1,1, 1},{-1,1, 1},
    };

    Eigen::Vector3d local;
    Eigen::Vector3d world[8];
    for (int k = 0; k < 8; ++k) {
        local[0] = SIGNS[k][0] * half_extents[0];
        local[1] = SIGNS[k][1] * half_extents[1];
        local[2] = SIGNS[k][2] * half_extents[2];
        world[k] = rotation * local + center;
    }

    if (is_up_z(ground_normal)) {
        // Axis-aligned-up ground: only z matters.
        int n_hit = 0;
        int hit_idx[8];
        for (int k = 0; k < 8; ++k) {
            if (world[k][2] < ground_height) hit_idx[n_hit++] = k;
        }
        if (n_hit == 0) return ContactBuf(0, 10);
        ContactBuf out(n_hit, 10);
        Eigen::Vector3d up(0.0, 0.0, 1.0);
        for (int i = 0; i < n_hit; ++i) {
            const Eigen::Vector3d& v = world[hit_idx[i]];
            const double dist = v[2] - ground_height;
            Eigen::Vector3d pb = v;
            pb[2] = ground_height;
            emit_row(out, i, v, pb, up, -dist);
        }
        return out;
    }

    Eigen::Vector3d n = ground_normal;
    const double nrm = n.norm();
    if (nrm > 0) n /= nrm;

    int n_hit = 0;
    int hit_idx[8];
    double hit_dist[8];
    for (int k = 0; k < 8; ++k) {
        const double dist = world[k].dot(n) - ground_height;
        if (dist < 0.0) { hit_idx[n_hit] = k; hit_dist[n_hit] = dist; ++n_hit; }
    }
    if (n_hit == 0) return ContactBuf(0, 10);
    ContactBuf out(n_hit, 10);
    for (int i = 0; i < n_hit; ++i) {
        const Eigen::Vector3d& v = world[hit_idx[i]];
        const double d = hit_dist[i];
        emit_row(out, i, v, v - d * n, n, -d);
    }
    return out;
}


namespace {

// Sutherland-Hodgman half-space clip — same logic as the Python ref.
// Keeps vertices with dot(v, plane_normal) <= plane_d.
void clip_polygon(std::vector<Eigen::Vector3d>& poly,
                  const Eigen::Vector3d& pn, double pd) {
    if (poly.empty()) return;
    std::vector<Eigen::Vector3d> result;
    result.reserve(poly.size() + 1);
    const int n = static_cast<int>(poly.size());
    for (int i = 0; i < n; ++i) {
        const auto& curr = poly[i];
        const auto& nxt  = poly[(i + 1) % n];
        const double dc = curr.dot(pn) - pd;
        const double dn = nxt .dot(pn) - pd;
        if (dc <= 0.0) result.push_back(curr);
        if ((dc > 0.0) != (dn > 0.0)) {
            const double t = dc / (dc - dn);
            result.push_back(curr + t * (nxt - curr));
        }
    }
    poly.swap(result);
}

double support_dist(const Eigen::Vector3d& half_ext,
                    const Eigen::Vector3d ax[3],
                    const Eigen::Vector3d& direction) {
    double s = 0.0;
    for (int i = 0; i < 3; ++i)
        s += half_ext[i] * std::abs(ax[i].dot(direction));
    return s;
}

}  // namespace


ContactBuf box_box(const Eigen::Ref<const Eigen::Vector3d>& center_a,
                   const Eigen::Ref<const Mat3RM>& rot_a,
                   const Eigen::Ref<const Eigen::Vector3d>& half_a,
                   const Eigen::Ref<const Eigen::Vector3d>& center_b,
                   const Eigen::Ref<const Mat3RM>& rot_b,
                   const Eigen::Ref<const Eigen::Vector3d>& half_b) {
    // Rows of rot.transpose() = local axes in world frame.
    Eigen::Vector3d ax_a[3], ax_b[3];
    for (int i = 0; i < 3; ++i) {
        ax_a[i] = rot_a.col(i);   // rot_a row-major: ax_a[i] = rot_a.T.row(i) = rot_a.col(i)
        ax_b[i] = rot_b.col(i);
    }
    const Eigen::Vector3d d = center_b - center_a;

    // ── Phase 1: SAT ──
    Eigen::Vector3d axes[15];
    int n_axes = 0;
    for (int i = 0; i < 3; ++i) axes[n_axes++] = ax_a[i];
    for (int i = 0; i < 3; ++i) axes[n_axes++] = ax_b[i];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Eigen::Vector3d c = ax_a[i].cross(ax_b[j]);
            const double nm = c.norm();
            if (nm > 0.01) axes[n_axes++] = c / nm;
        }
    }

    double min_pen = std::numeric_limits<double>::infinity();
    Eigen::Vector3d min_axis = Eigen::Vector3d::Zero();
    bool have_axis = false;
    for (int a = 0; a < n_axes; ++a) {
        const Eigen::Vector3d& axis = axes[a];
        const double proj_a =
            half_a[0] * std::abs(ax_a[0].dot(axis)) +
            half_a[1] * std::abs(ax_a[1].dot(axis)) +
            half_a[2] * std::abs(ax_a[2].dot(axis));
        const double proj_b =
            half_b[0] * std::abs(ax_b[0].dot(axis)) +
            half_b[1] * std::abs(ax_b[1].dot(axis)) +
            half_b[2] * std::abs(ax_b[2].dot(axis));
        const double d_axis = d.dot(axis);
        const double dist   = std::abs(d_axis);
        const double pen    = proj_a + proj_b - dist;
        if (pen <= 0.0) return ContactBuf(0, 10);
        if (pen < min_pen) {
            min_pen  = pen;
            min_axis = (d_axis < 0.0) ? axis : Eigen::Vector3d(-axis);
            have_axis = true;
        }
    }
    if (!have_axis) return ContactBuf(0, 10);

    const Eigen::Vector3d normal = min_axis;
    const double penetration     = min_pen;

    // ── Phase 2: contact manifold via face clipping ──
    double dots_a[3] = {ax_a[0].dot(normal), ax_a[1].dot(normal), ax_a[2].dot(normal)};
    int ref_idx = 0;
    if (std::abs(dots_a[1]) > std::abs(dots_a[ref_idx])) ref_idx = 1;
    if (std::abs(dots_a[2]) > std::abs(dots_a[ref_idx])) ref_idx = 2;
    Eigen::Vector3d ref_fn     = (dots_a[ref_idx] > 0.0) ? Eigen::Vector3d(-ax_a[ref_idx])
                                                         : ax_a[ref_idx];
    Eigen::Vector3d ref_center = center_a + half_a[ref_idx] * ref_fn;

    double dots_b[3] = {ax_b[0].dot(normal), ax_b[1].dot(normal), ax_b[2].dot(normal)};
    int inc_idx = 0;
    if (std::abs(dots_b[1]) > std::abs(dots_b[inc_idx])) inc_idx = 1;
    if (std::abs(dots_b[2]) > std::abs(dots_b[inc_idx])) inc_idx = 2;
    Eigen::Vector3d inc_fn     = (dots_b[inc_idx] > 0.0) ? ax_b[inc_idx]
                                                         : Eigen::Vector3d(-ax_b[inc_idx]);
    Eigen::Vector3d inc_center = center_b + half_b[inc_idx] * inc_fn;

    int other_b[2], k = 0;
    for (int i = 0; i < 3; ++i) if (i != inc_idx) other_b[k++] = i;
    const Eigen::Vector3d& ub = ax_b[other_b[0]];
    const Eigen::Vector3d& vb = ax_b[other_b[1]];
    const double hub = half_b[other_b[0]];
    const double hvb = half_b[other_b[1]];

    std::vector<Eigen::Vector3d> polygon;
    polygon.reserve(8);
    polygon.push_back(inc_center + hub * ub + hvb * vb);
    polygon.push_back(inc_center - hub * ub + hvb * vb);
    polygon.push_back(inc_center - hub * ub - hvb * vb);
    polygon.push_back(inc_center + hub * ub - hvb * vb);

    int other_a[2]; k = 0;
    for (int i = 0; i < 3; ++i) if (i != ref_idx) other_a[k++] = i;
    const Eigen::Vector3d& ua = ax_a[other_a[0]];
    const Eigen::Vector3d& va = ax_a[other_a[1]];
    const double hua = half_a[other_a[0]];
    const double hva = half_a[other_a[1]];

    const Eigen::Vector3d sn[4] = { ua, -ua, va, -va };
    const double sd[4] = {
        ref_center.dot( ua) + hua,
        ref_center.dot(-ua) + hua,
        ref_center.dot( va) + hva,
        ref_center.dot(-va) + hva,
    };
    for (int p = 0; p < 4; ++p) {
        clip_polygon(polygon, sn[p], sd[p]);
        if (polygon.empty()) return ContactBuf(0, 10);
    }

    const double ref_plane_d = ref_center.dot(ref_fn);
    constexpr double TOL = 5e-4;
    struct Hit { Eigen::Vector3d pa, pb; double depth; };
    std::vector<Hit> hits;
    hits.reserve(polygon.size());
    for (const auto& p : polygon) {
        const double sd_p = p.dot(ref_fn) - ref_plane_d;
        const double depth = -sd_p;
        if (depth >= -TOL) {
            Eigen::Vector3d pa = p - sd_p * ref_fn;
            hits.push_back({pa, p, std::max(0.0, depth)});
        }
    }

    if (hits.empty()) {
        // Degenerate fallback — single centre-point contact.
        const double sa = support_dist(half_a, ax_a, normal);
        const double sb = support_dist(half_b, ax_b, -normal);
        Eigen::Vector3d pa = center_a - sa * normal;
        Eigen::Vector3d pb = center_b - sb * normal;  // center_b + sb*(-normal) → ...
        ContactBuf out(1, 10);
        emit_row(out, 0, pa, pb, normal, penetration);
        return out;
    }

    // Cap to 4 deepest, mirroring the Python manifold reduction.
    if (hits.size() > 4) {
        std::partial_sort(hits.begin(), hits.begin() + 4, hits.end(),
                          [](const Hit& x, const Hit& y){ return x.depth > y.depth; });
        hits.resize(4);
    }

    ContactBuf out(static_cast<int>(hits.size()), 10);
    for (int i = 0; i < static_cast<int>(hits.size()); ++i)
        emit_row(out, i, hits[i].pa, hits[i].pb, normal, hits[i].depth);
    return out;
}


ContactBuf cylinder_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                           const Eigen::Ref<const Mat3RM>& rotation,
                           double radius, double half_length,
                           double ground_height,
                           const Eigen::Ref<const Eigen::Vector3d>& ground_normal,
                           int n_ring) {
    // Build local ring (top + bottom) once per call. Match Python ref.
    std::vector<Eigen::Vector3d> world(2 * static_cast<size_t>(n_ring));
    const double inv = 2.0 * M_PI / static_cast<double>(n_ring);
    for (int k = 0; k < n_ring; ++k) {
        const double a = inv * k;
        const double cx = std::cos(a) * radius;
        const double sy = std::sin(a) * radius;
        Eigen::Vector3d loc_b(cx, sy, -half_length);
        Eigen::Vector3d loc_t(cx, sy,  half_length);
        world[k]                       = rotation * loc_b + center;
        world[static_cast<size_t>(n_ring) + k] = rotation * loc_t + center;
    }

    if (is_up_z(ground_normal)) {
        std::vector<int> hits; hits.reserve(world.size());
        for (size_t k = 0; k < world.size(); ++k)
            if (world[k][2] < ground_height) hits.push_back(static_cast<int>(k));
        if (hits.empty()) return ContactBuf(0, 10);
        ContactBuf out(static_cast<int>(hits.size()), 10);
        Eigen::Vector3d up(0.0, 0.0, 1.0);
        for (size_t i = 0; i < hits.size(); ++i) {
            const Eigen::Vector3d& v = world[hits[i]];
            const double dist = v[2] - ground_height;
            Eigen::Vector3d pb = v;
            pb[2] = ground_height;
            emit_row(out, static_cast<int>(i), v, pb, up, -dist);
        }
        return out;
    }

    Eigen::Vector3d n = ground_normal;
    const double nrm = n.norm();
    if (nrm > 0) n /= nrm;
    std::vector<std::pair<int, double>> hits;
    for (size_t k = 0; k < world.size(); ++k) {
        const double dist = world[k].dot(n) - ground_height;
        if (dist < 0.0) hits.push_back({static_cast<int>(k), dist});
    }
    if (hits.empty()) return ContactBuf(0, 10);
    ContactBuf out(static_cast<int>(hits.size()), 10);
    for (size_t i = 0; i < hits.size(); ++i) {
        const Eigen::Vector3d& v = world[hits[i].first];
        const double d = hits[i].second;
        emit_row(out, static_cast<int>(i), v, v - d * n, n, -d);
    }
    return out;
}


AABBBuf aabb_box(const Eigen::Ref<const Mat3RM>& rotation,
                 const Eigen::Ref<const Eigen::Vector3d>& translation,
                 const Eigen::Ref<const Eigen::Vector3d>& half_extents) {
    // extent_i = sum_j |R_ij| * he_j  →  rows are world-axis projections.
    Eigen::Vector3d extent;
    for (int i = 0; i < 3; ++i) {
        extent[i] = std::abs(rotation(i, 0)) * half_extents[0]
                  + std::abs(rotation(i, 1)) * half_extents[1]
                  + std::abs(rotation(i, 2)) * half_extents[2];
    }
    AABBBuf out;
    out.row(0) = (translation - extent).transpose();
    out.row(1) = (translation + extent).transpose();
    return out;
}


AABBBuf aabb_sphere(const Eigen::Ref<const Eigen::Vector3d>& translation,
                    double radius) {
    AABBBuf out;
    Eigen::Vector3d r(radius, radius, radius);
    out.row(0) = (translation - r).transpose();
    out.row(1) = (translation + r).transpose();
    return out;
}


AABBBuf aabb_cylinder(const Eigen::Ref<const Mat3RM>& rotation,
                      const Eigen::Ref<const Eigen::Vector3d>& translation,
                      double radius, double length) {
    const double hl = 0.5 * length;
    Eigen::Vector3d axis = rotation.col(2);     // symmetry axis in world
    Eigen::Vector3d cos2(axis[0]*axis[0], axis[1]*axis[1], axis[2]*axis[2]);
    Eigen::Vector3d extent;
    for (int i = 0; i < 3; ++i) {
        const double r_term = std::max(0.0, 1.0 - cos2[i]);
        extent[i] = hl * std::abs(axis[i]) + radius * std::sqrt(r_term);
    }
    AABBBuf out;
    out.row(0) = (translation - extent).transpose();
    out.row(1) = (translation + extent).transpose();
    return out;
}


Eigen::VectorXd contact_force_penalty(
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    const Eigen::Ref<const Eigen::Vector3d>& v_a,
    const Eigen::Ref<const Eigen::Vector3d>& v_b,
    double penetration,
    double stiffness, double damping,
    double mu, double friction_eps,
    double max_penetration,
    double effective_mass) {
    if (penetration <= 0.0) return Eigen::VectorXd();

    Eigen::Vector3d v_rel = v_a - v_b;
    const double v_n = v_rel.dot(normal);
    const double d   = std::min(penetration, max_penetration);

    double c = damping;
    if (effective_mass > 0.0) {
        const double c_crit = 2.0 * std::sqrt(stiffness * effective_mass);
        if (c_crit < c) c = c_crit;
    }
    const double fn_mag = stiffness * d - c * v_n;
    if (fn_mag <= 0.0) return Eigen::VectorXd();

    Eigen::Vector3d f_normal = fn_mag * normal;
    Eigen::Vector3d v_t = v_rel - v_n * normal;
    const double v_t_norm = v_t.norm();
    const double f_max    = mu * fn_mag;

    Eigen::Vector3d f_friction = Eigen::Vector3d::Zero();
    if (v_t_norm > 1e-12) {
        const double scale = std::min(1.0, v_t_norm / friction_eps);
        f_friction = -f_max * scale * (v_t / v_t_norm);
    }
    Eigen::Vector3d f = f_normal + f_friction;
    Eigen::VectorXd out(4);
    out << f[0], f[1], f[2], fn_mag;
    return out;
}

}  // namespace robosim
