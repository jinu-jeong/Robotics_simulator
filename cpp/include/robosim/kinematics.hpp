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

}  // namespace robosim
