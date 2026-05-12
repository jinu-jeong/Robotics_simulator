// Rigid-body dynamics — ABA (forward) and gravity-torque shortcut.
//
// Mirrors ``robosim/physics/rbd/algorithms.py::aba`` and
// ``gravity_torques`` so the Python solver can keep its public API
// while ducking the per-joint Python overhead. Both kernels read a
// pre-built :class:`RbdTopology` once and operate on raw numpy
// arrays from there on.

#pragma once

#include "robosim/kinematics.hpp"

#include <Eigen/Dense>

namespace robosim {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Vec6 = Eigen::Matrix<double, 6, 1>;

// Forward dynamics: qdd = M(q)^{-1} (tau - C(q,qd)).
//
// Inputs:
//   topo    : built once
//   q       : (n_dof,)
//   qd      : (n_dof,)
//   tau     : (n_dof,)
//   gravity : (3,)
// Returns: (n_dof,) qdd.
// ``f_ext`` is a flat (n_links * 6,) vector of spatial wrenches per
// link (link i at offset 6*i). Pass a length-0 vector or an
// all-zero vector when no external forces are applied. Layout
// matches the Python ``dict[link_idx -> (6,) ndarray]`` after the
// bridge densifies it.
Eigen::VectorXd aba(const RbdTopology& topo,
                    const Eigen::Ref<const Eigen::VectorXd>& q,
                    const Eigen::Ref<const Eigen::VectorXd>& qd,
                    const Eigen::Ref<const Eigen::VectorXd>& tau,
                    const Eigen::Ref<const Eigen::Vector3d>& gravity,
                    const Eigen::Ref<const Eigen::VectorXd>& f_ext_flat);

// Gravity compensation torques: τ_g = RNEA(q, 0, 0, gravity).
// Returns (n_dof,) τ_g.
Eigen::VectorXd gravity_torques(const RbdTopology& topo,
                                const Eigen::Ref<const Eigen::VectorXd>& q,
                                const Eigen::Ref<const Eigen::Vector3d>& gravity);

}  // namespace robosim
