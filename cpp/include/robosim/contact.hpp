// Contact narrow-phase kernels — port of robosim/physics/contact/sdf.py
// hot paths.
//
// Each kernel returns a *contact buffer* — a packed (N, 10) RowMajor
// numpy-friendly matrix with columns:
//   [0:3]  point_a (world)
//   [3:6]  point_b (world)
//   [6:9]  normal (B -> A)
//   [9]    penetration (positive)
//
// Python wrappers reconstruct ContactPoint dataclasses from each row.
// Returning a packed array keeps the boundary cheap (one malloc per
// call) and matches what Python's vectorised code already produces.
//
// N is variable per call: 0 (no contact), 1 (sphere/box-sphere), up to
// 4 (box-box manifold) or 8 (box-ground vertices).

#pragma once

#include <Eigen/Dense>

namespace robosim {

using ContactBuf = Eigen::Matrix<double, Eigen::Dynamic, 10, Eigen::RowMajor>;
using Mat3RM    = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

// ── Single-contact kernels ────────────────────────────────────────
// Return a (0, 10) matrix when there's no contact, (1, 10) otherwise.

ContactBuf sphere_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                         double radius,
                         double ground_height,
                         const Eigen::Ref<const Eigen::Vector3d>& ground_normal);

ContactBuf sphere_sphere(const Eigen::Ref<const Eigen::Vector3d>& c1, double r1,
                         const Eigen::Ref<const Eigen::Vector3d>& c2, double r2);

ContactBuf box_sphere(const Eigen::Ref<const Eigen::Vector3d>& box_center,
                      const Eigen::Ref<const Mat3RM>& box_rot,
                      const Eigen::Ref<const Eigen::Vector3d>& box_half,
                      const Eigen::Ref<const Eigen::Vector3d>& sphere_center,
                      double sphere_radius);

// ── Multi-contact kernels ─────────────────────────────────────────

ContactBuf box_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                      const Eigen::Ref<const Mat3RM>& rotation,
                      const Eigen::Ref<const Eigen::Vector3d>& half_extents,
                      double ground_height,
                      const Eigen::Ref<const Eigen::Vector3d>& ground_normal);

ContactBuf box_box(const Eigen::Ref<const Eigen::Vector3d>& center_a,
                   const Eigen::Ref<const Mat3RM>& rot_a,
                   const Eigen::Ref<const Eigen::Vector3d>& half_a,
                   const Eigen::Ref<const Eigen::Vector3d>& center_b,
                   const Eigen::Ref<const Mat3RM>& rot_b,
                   const Eigen::Ref<const Eigen::Vector3d>& half_b);

ContactBuf cylinder_ground(const Eigen::Ref<const Eigen::Vector3d>& center,
                           const Eigen::Ref<const Mat3RM>& rotation,
                           double radius, double half_length,
                           double ground_height,
                           const Eigen::Ref<const Eigen::Vector3d>& ground_normal,
                           int n_ring);

// AABB helpers — return (min_pt, max_pt) packed as (2, 3).
using AABBBuf = Eigen::Matrix<double, 2, 3, Eigen::RowMajor>;

AABBBuf aabb_box(const Eigen::Ref<const Mat3RM>& rotation,
                 const Eigen::Ref<const Eigen::Vector3d>& translation,
                 const Eigen::Ref<const Eigen::Vector3d>& half_extents);

AABBBuf aabb_sphere(const Eigen::Ref<const Eigen::Vector3d>& translation,
                    double radius);

AABBBuf aabb_cylinder(const Eigen::Ref<const Mat3RM>& rotation,
                      const Eigen::Ref<const Eigen::Vector3d>& translation,
                      double radius, double length);

// Penalty + kinetic Coulomb contact force.
// Inputs:
//   normal       (3,) B->A unit normal
//   v_a, v_b     (3,) point velocities
//   penetration  scalar
//   stiffness, damping, mu, friction_eps, max_pen
//   effective_mass : -1 to disable critical-damping cap
// Returns:
//   (4,) [fx, fy, fz, fn_mag]  if contact produces a force
//   (0,)                       if separating fast enough / depth ≤ 0
Eigen::VectorXd contact_force_penalty(
    const Eigen::Ref<const Eigen::Vector3d>& normal,
    const Eigen::Ref<const Eigen::Vector3d>& v_a,
    const Eigen::Ref<const Eigen::Vector3d>& v_b,
    double penetration,
    double stiffness, double damping,
    double mu, double friction_eps,
    double max_penetration,
    double effective_mass);

}  // namespace robosim
