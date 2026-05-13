// Substep-level scene orchestration in C++ — fuses what is currently
// many small kernel calls per substep (FK, link_vel, contact detection,
// contact force, ABA, integrate) into ONE Python ↔ C++ boundary cross
// per ``Scene.step()`` call, regardless of n_substeps.
//
// Scope (v1): rigid-only scenes with one ground plane and a fixed
// collection of OBB/sphere/cylinder collision bodies attached to
// articulated robot links. FEM / CB bodies stay on the Python side.
//
// Bookkeeping rationale: the data registered up-front (collision
// bodies, filters, PD setup, gravity) doesn't change inside the
// substep loop, so we lock it in via ``add_*`` methods at Python side
// initialisation, then call ``step(dt, n_substeps)`` per outer tick.

#pragma once

#include "robosim/kinematics.hpp"
#include "robosim/rbd.hpp"
#include "robosim/contact.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace robosim {

enum class ColGeom : std::int8_t {
    SPHERE   = 0,
    BOX      = 1,
    CYLINDER = 2,
};

struct CollisionBodyEntry {
    int robot_idx;
    int link_idx;
    int body_id;           // global unique id (matches Python detector ids)
    ColGeom geom;
    // Box: dim = (hx, hy, hz). Sphere: dim[0] = r. Cylinder: dim[0]=r, dim[1]=length.
    Eigen::Vector3d dim;
    Eigen::Matrix3d origin_R;
    Eigen::Vector3d origin_t;
};

struct RobotEntry {
    RbdTopology* topo;
    int n_links;
    int n_dof;

    Eigen::VectorXd q;
    Eigen::VectorXd qd;
    Eigen::VectorXd tau;          // baseline external joint torque (e.g. supplied per outer step)
    Eigen::Vector3d gravity;

    // PD control (optional). When pd_enabled and step is called with
    // an applicable q_target, tau_effective = tau + PD.
    bool pd_enabled = false;
    Eigen::VectorXd q_target;
    Eigen::VectorXd kp;
    Eigen::VectorXd kd;

    // Whether this robot represents a single free-floating body for
    // which we should not run PD (just dynamics under gravity + contact).
    bool is_free = false;

    // Per-substep scratch (resized at add_robot).
    MatRMXd R_flat;   // (n_links, 9)
    MatRMXd t_arr;    // (n_links, 3)
    MatRMXd omega;    // (n_links, 3)
    MatRMXd v_orig;   // (n_links, 3)
    Eigen::VectorXd f_ext_flat;  // (n_links*6,) wrench accumulator
};


// Public Python-side handle. Holds all robots, collision bodies, and
// the global ground+contact params. ``step()`` runs the substep loop.
class RigidSceneStep {
 public:
    RigidSceneStep();
    ~RigidSceneStep();

    // ── Registration (called once from Python at scene build time) ──

    // Add a robot. Returns its index. Caller must keep the topo alive
    // for the lifetime of the scene_step.
    int add_robot(RbdTopology* topo,
                  const Eigen::Ref<const Eigen::Vector3d>& gravity,
                  bool is_free = false);

    void set_pd_control(int robot_idx,
                        const Eigen::Ref<const Eigen::VectorXd>& kp,
                        const Eigen::Ref<const Eigen::VectorXd>& kd);

    // Set q_target for the upcoming step (Python computes the target
    // each outer tick).
    void set_q_target(int robot_idx,
                      const Eigen::Ref<const Eigen::VectorXd>& q_target);

    void set_tau(int robot_idx,
                 const Eigen::Ref<const Eigen::VectorXd>& tau);

    void set_state(int robot_idx,
                   const Eigen::Ref<const Eigen::VectorXd>& q,
                   const Eigen::Ref<const Eigen::VectorXd>& qd);

    std::pair<Eigen::VectorXd, Eigen::VectorXd> get_state(int robot_idx) const;

    // Add a collision body attached to robot_idx's link_idx.
    // Returns a global body_id.
    int add_box(int robot_idx, int link_idx,
                const Eigen::Ref<const Eigen::Vector3d>& half_extents,
                const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
                const Eigen::Ref<const Eigen::Vector3d>& origin_t);
    int add_sphere(int robot_idx, int link_idx, double radius,
                   const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
                   const Eigen::Ref<const Eigen::Vector3d>& origin_t);
    int add_cylinder(int robot_idx, int link_idx,
                     double radius, double length,
                     const Eigen::Ref<const Eigen::Matrix3d>& origin_R,
                     const Eigen::Ref<const Eigen::Vector3d>& origin_t);

    // Disable collision between this pair of body_ids.
    void add_filter(int body_id_a, int body_id_b);

    void set_ground(double height,
                    const Eigen::Ref<const Eigen::Vector3d>& normal);

    void set_contact_params(double stiffness, double damping,
                            double mu, double friction_eps,
                            double max_penetration);

    // ── The main entry point. Runs ``n_substeps`` substeps. ──
    void step(double dt, int n_substeps);

    // ── Diagnostics ──
    int n_robots() const { return static_cast<int>(robots_.size()); }
    int n_bodies() const { return static_cast<int>(bodies_.size()); }

 private:
    std::vector<RobotEntry> robots_;
    std::vector<CollisionBodyEntry> bodies_;
    int next_body_id_ = 0;

    // Ground (optional). zero-normal ⇒ no ground.
    double ground_h_ = 0.0;
    Eigen::Vector3d ground_n_ = Eigen::Vector3d::Zero();

    // Contact params.
    double k_ = 1e5, c_ = 1e3, mu_ = 0.5, fric_eps_ = 1e-3, max_pen_ = 1e-2;

    std::set<std::pair<int, int>> filters_;

    // Scratch reused across substeps. Sized at first step().
    std::vector<Eigen::Vector3d> body_t_world_;
    std::vector<Eigen::Matrix3d> body_R_world_;
    std::vector<Eigen::Vector3d> body_aabb_min_, body_aabb_max_;

    void run_one_substep_(double dt);
    void update_body_transforms_();
    void compute_body_aabbs_();
    void apply_pd_to_tau_(int robot_idx, Eigen::VectorXd& tau_out);
};

}  // namespace robosim
