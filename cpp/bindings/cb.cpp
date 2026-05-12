// pybind11 bindings for the CB step kernels.

#include "robosim/cb.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace robosim {

void register_cb(py::module_& m) {
    auto c = m.def_submodule("cb",
        "Craig-Bampton modal step (anchored / free). The Python "
        "``CraigBamptonBody`` keeps ownership of the precomputed CB "
        "basis (Phi, M_r, A_r_inv, …); these entry points consume "
        "those arrays as zero-copy Eigen views per call.");

    c.def("step_anchored",
          [](const Eigen::Ref<const MatXRM>& x,
             const Eigen::Ref<const MatXRM>& v,
             const Eigen::Ref<const MatXRM>& nodes,
             const Eigen::Ref<const MatXRM>& Phi_CB,
             const Eigen::Ref<const MatXRM>& M_r,
             const Eigen::Ref<const MatXRM>& A_r_inv,
             const Eigen::Ref<const MatXRM>& C_q,
             const Eigen::Ref<const Eigen::VectorXd>& M_diag,
             const Eigen::Ref<const Eigen::VectorXi>& free_dofs,
             const Eigen::Ref<const Eigen::VectorXi>& fixed_dofs,
             const Eigen::Ref<const Eigen::Vector3d>& gravity,
             const Eigen::Ref<const Eigen::VectorXd>& extra_force_flat,
             double dt, double damping) {
              return cb_step_anchored(x, v, nodes, Phi_CB, M_r, A_r_inv,
                                       C_q, M_diag, free_dofs, fixed_dofs,
                                       gravity, extra_force_flat, dt, damping);
          },
          py::arg("x"), py::arg("v"), py::arg("nodes"),
          py::arg("Phi_CB"), py::arg("M_r"), py::arg("A_r_inv"),
          py::arg("C_q"), py::arg("M_diag"),
          py::arg("free_dofs"), py::arg("fixed_dofs"),
          py::arg("gravity"), py::arg("extra_force_flat"),
          py::arg("dt"), py::arg("damping"),
          "Anchored CB step — returns (x_new, v_new, q_r_new).");

    c.def("step_free",
          [](const Eigen::Ref<const MatXRM>& x,
             const Eigen::Ref<const MatXRM>& v,
             const Eigen::Ref<const MatXRM>& x_ref_body,
             const Eigen::Ref<const MatXRM>& Phi_CB,
             const Eigen::Ref<const MatXRM>& M_r,
             const Eigen::Ref<const MatXRM>& A_r_inv,
             const Eigen::Ref<const Eigen::VectorXd>& M_diag,
             const Eigen::Ref<const Eigen::Vector3d>& gravity,
             const Eigen::Ref<const Eigen::VectorXd>& extra_force_flat,
             double dt, double damping, double m_total) {
              return cb_step_free(x, v, x_ref_body, Phi_CB, M_r, A_r_inv,
                                   M_diag, gravity, extra_force_flat,
                                   dt, damping, m_total);
          },
          py::arg("x"), py::arg("v"), py::arg("x_ref_body"),
          py::arg("Phi_CB"), py::arg("M_r"), py::arg("A_r_inv"),
          py::arg("M_diag"), py::arg("gravity"),
          py::arg("extra_force_flat"),
          py::arg("dt"), py::arg("damping"), py::arg("m_total"),
          "Free-floating CB step (Kabsch rotation + centroid). "
          "Returns (x_new, v_new, q_r_new).");
}

}  // namespace robosim
