// SE(3) rigid-body transforms — hot-path helpers.
//
// Mirrors the operations in ``robosim/math/transforms.py`` that are
// called per joint, per step (compose / inverse / from_axis_angle /
// apply_point / apply_vector / cross3 / skew / rotation_x/y/z).
//
// Everything here is header-only ``inline`` to let the FK chain in
// Stage 2 fold these into a single translation unit and avoid call
// overhead. Eigen is the storage type for both the C++ side and the
// pybind11 ↔ numpy binding (zero-copy via ``Eigen::Ref``).

#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace robosim {

using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

// ── Skew / cross3 ──────────────────────────────────────────────────────────

inline Vec3 cross3(const Vec3& a, const Vec3& b) noexcept {
    return Vec3(a.y() * b.z() - a.z() * b.y(),
                a.z() * b.x() - a.x() * b.z(),
                a.x() * b.y() - a.y() * b.x());
}

inline Mat3 skew(const Vec3& v) noexcept {
    Mat3 S;
    S <<  0.0,   -v.z(),  v.y(),
          v.z(),  0.0,   -v.x(),
         -v.y(),  v.x(),  0.0;
    return S;
}

// ── Elementary axis rotations ──────────────────────────────────────────────

inline Mat3 rotation_x(double a) noexcept {
    const double c = std::cos(a), s = std::sin(a);
    Mat3 R;
    R << 1.0, 0.0, 0.0,
         0.0, c,  -s,
         0.0, s,   c;
    return R;
}

inline Mat3 rotation_y(double a) noexcept {
    const double c = std::cos(a), s = std::sin(a);
    Mat3 R;
    R <<  c, 0.0, s,
         0.0, 1.0, 0.0,
         -s, 0.0, c;
    return R;
}

inline Mat3 rotation_z(double a) noexcept {
    const double c = std::cos(a), s = std::sin(a);
    Mat3 R;
    R << c,  -s, 0.0,
         s,   c, 0.0,
         0.0, 0.0, 1.0;
    return R;
}

// ── Axis-angle (Rodrigues) ─────────────────────────────────────────────────
// Mirrors the explicit element-wise form in transforms.py; the
// branch on |axis|^2 == 0 matches identity() so revolute joints with
// axis=0 are quietly treated as fixed (parses survive bad URDFs).

inline Mat3 from_axis_angle(const Vec3& axis, double angle) noexcept {
    double x = axis.x(), y = axis.y(), z = axis.z();
    const double n2 = x * x + y * y + z * z;
    if (n2 < 1e-24) {
        return Mat3::Identity();
    }
    if (std::abs(n2 - 1.0) > 1e-9) {
        const double inv = 1.0 / std::sqrt(n2);
        x *= inv; y *= inv; z *= inv;
    }
    const double s = std::sin(angle);
    const double c = std::cos(angle);
    const double C = 1.0 - c;
    Mat3 R;
    R(0, 0) = c + x * x * C;
    R(0, 1) = x * y * C - z * s;
    R(0, 2) = x * z * C + y * s;
    R(1, 0) = y * x * C + z * s;
    R(1, 1) = c + y * y * C;
    R(1, 2) = y * z * C - x * s;
    R(2, 0) = z * x * C - y * s;
    R(2, 1) = z * y * C + x * s;
    R(2, 2) = c + z * z * C;
    return R;
}

// ── SE(3) compose / inverse / apply ────────────────────────────────────────
//
// Transforms are passed as (R, t) pairs; the Python ``Transform``
// dataclass and its ``_fast()`` shortcut already use this layout, so
// the binding can borrow numpy buffers without copying.

inline void compose_Rt(const Mat3& R1, const Vec3& t1,
                       const Mat3& R2, const Vec3& t2,
                       Mat3& Rout, Vec3& tout) noexcept {
    Rout.noalias() = R1 * R2;
    tout.noalias() = R1 * t2 + t1;
}

inline void inverse_Rt(const Mat3& R, const Vec3& t,
                       Mat3& Rout, Vec3& tout) noexcept {
    Rout.noalias() = R.transpose();
    tout.noalias() = -Rout * t;
}

inline Vec3 apply_point(const Mat3& R, const Vec3& t, const Vec3& p) noexcept {
    return R * p + t;
}

inline Vec3 apply_vector(const Mat3& R, const Vec3& v) noexcept {
    return R * v;
}

}  // namespace robosim
