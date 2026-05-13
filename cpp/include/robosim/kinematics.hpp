// Forward kinematics for an articulated rigid body — single C++ call
// replacing the per-link Python BFS in ``robosim/model/robot.py``.
//
// The Python side passes the *topology* (parent map, joint axes, joint
// types, origin transforms, dof indices) once via ``build_topology``,
// then calls ``forward_kinematics(q)`` every step. Output is two
// dense buffers: per-link rotation (n_links × 3 × 3) and per-link
// translation (n_links × 3) packed contiguously so the Python wrapper
// can hand each row pair to a ``Transform._fast`` shortcut.

#pragma once

#include "robosim/transform.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace robosim {

enum class JointKind : std::int8_t {
    FIXED = 0,
    REVOLUTE = 1,
    PRISMATIC = 2,
};

// Per-joint precomputed data. Stored in BFS-friendly order: the
// builder sorts joints so that parent links are always laid out
// before their children, letting the FK loop pull T_world[parent]
// without further bookkeeping.
struct JointSpec {
    int parent_link;   // -1 for joints whose parent link is a root
    int child_link;
    JointKind kind;
    int dof_index;     // index into q; -1 if fixed
    Eigen::Vector3d axis;       // unit axis (already normalised)
    Eigen::Matrix3d origin_R;   // joint-origin rotation (in parent frame)
    Eigen::Vector3d origin_t;   // joint-origin translation
};

struct Topology {
    int n_links = 0;
    int n_dof = 0;
    // Ordered so that for every joint j, joints[j].parent_link has its
    // world transform already filled by the time the FK loop hits j.
    std::vector<JointSpec> joints;
    // Roots = links with parent_link == -1 in any joint OR no incoming
    // joint at all; we just identify them and seed identity.
    std::vector<int> root_links;
};

// Extension of Topology with the per-link spatial inertia + tree
// traversal order needed by RBD (ABA, RNEA, gravity_torques).
//
// Built once from the Python Robot at first call; mutating the
// model means rebuilding (same contract as the bare Topology).
struct RbdTopology : public Topology {
    // (n_links,) parent link index — -1 for roots.
    std::vector<int> parent_link;
    // (n_links,) joint index that has this link as its child — -1
    // for roots (no inbound joint).
    std::vector<int> joint_for_link;
    // Tree order (BFS from each root, excluding root): the iteration
    // order RBD's 3 passes follow. Reverse() of this is the backward
    // pass.
    std::vector<int> tree_order;
    // (n_links,) 6×6 spatial inertia of each link's body in its own
    // link frame. Row-major flatten so the bind code can hand us a
    // numpy array with shape (n_links, 36) and avoid the per-row
    // stride dance.
    Eigen::Matrix<double, Eigen::Dynamic, 36, Eigen::RowMajor> link_inertias;
};


// Row-major dense buffers used for the per-link output. Row-major so
// that ``R_out.row(l).data()`` is contiguous (length 9) and trivially
// reshape-able to (3, 3) on the Python side.
using MatRMXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Compute world-frame (R, t) for every link given q.
//
// Inputs:
//   topo : built once
//   q    : (n_dof,)
// Outputs:
//   R_out : (n_links, 9) row-major — row l = flatten(R_l) in row-major
//   t_out : (n_links, 3) row-major
//
// Roots are initialised to identity. Joints are walked in topology
// order — guaranteed parent-before-child by build_topology.
void forward_kinematics(const Topology& topo,
                        const Eigen::Ref<const Eigen::VectorXd>& q,
                        Eigen::Ref<MatRMXd> R_out,
                        Eigen::Ref<MatRMXd> t_out);

// Per-link world-frame spatial velocity (ω, v_origin).
//
// Uses an :class:`RbdTopology` for the per-link parent + joint-for-link
// tables plus `tree_order`. Mirrors
// ``robosim/model/robot.py::link_world_velocities`` exactly:
//   v_c = v_p + ω_p × (origin_c − origin_p)
//   ω_c = ω_p (+ qd · axis_world for revolute)
//   v_c += qd · axis_world for prismatic
//
// Inputs:
//   topo    : built RbdTopology
//   qd      : (n_dof,)
//   R_flat  : (n_links, 9) row-major flatten of FK rotations
//   t_arr   : (n_links, 3) FK translations
// Outputs (row-major, pre-sized):
//   omega_out : (n_links, 3)
//   v_out     : (n_links, 3)
void link_world_velocities(const RbdTopology& topo,
                           const Eigen::Ref<const Eigen::VectorXd>& qd,
                           const Eigen::Ref<const MatRMXd>& R_flat,
                           const Eigen::Ref<const MatRMXd>& t_arr,
                           Eigen::Ref<MatRMXd> omega_out,
                           Eigen::Ref<MatRMXd> v_out);

// Batched: world-frame velocity of N body-fixed points.
//   v[i] = v_origin[link_i] + omega[link_i] × (point_i − origin_link_i)
// link_indices : (N,)  int32
// points       : (N, 3)
// omega/v_arr  : (n_links, 3) from link_world_velocities
// t_arr        : (n_links, 3) from FK
// Returns      : (N, 3) point velocities
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
batch_point_velocities(const Eigen::Ref<const Eigen::VectorXi>& link_indices,
                       const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>& points,
                       const Eigen::Ref<const MatRMXd>& omega_arr,
                       const Eigen::Ref<const MatRMXd>& v_origin_arr,
                       const Eigen::Ref<const MatRMXd>& t_arr);

}  // namespace robosim
