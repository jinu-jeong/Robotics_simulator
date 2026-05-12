// pybind11 bindings for ABA + gravity_torques. Exposes
// ``robosim._cpp.rbd`` with the :class:`RbdTopology` extension of
// the bare :class:`Topology` and the two dynamics entry points.

#include "robosim/rbd.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace robosim {

void register_rbd(py::module_& m) {
    auto r = m.def_submodule("rbd",
        "Rigid-body dynamics — ABA (forward dynamics) and "
        "gravity_torques (RNEA shortcut with qd=qdd=0). Both consume "
        "an :class:`RbdTopology` built from the Python Robot via "
        "``robosim.model._cpp_bridge.build_rbd_topology``.");

    py::class_<RbdTopology, Topology>(r, "RbdTopology")
        .def(py::init<>())
        .def_readwrite("parent_link",     &RbdTopology::parent_link)
        .def_readwrite("joint_for_link",  &RbdTopology::joint_for_link)
        .def_readwrite("tree_order",      &RbdTopology::tree_order)
        .def_readwrite("link_inertias",   &RbdTopology::link_inertias);

    r.def("aba",
          [](const RbdTopology& topo,
             const Eigen::Ref<const Eigen::VectorXd>& q,
             const Eigen::Ref<const Eigen::VectorXd>& qd,
             const Eigen::Ref<const Eigen::VectorXd>& tau,
             const Eigen::Ref<const Eigen::Vector3d>& gravity,
             const Eigen::Ref<const Eigen::VectorXd>& f_ext_flat) {
              return aba(topo, q, qd, tau, gravity, f_ext_flat);
          },
          py::arg("topo"), py::arg("q"), py::arg("qd"),
          py::arg("tau"), py::arg("gravity"),
          py::arg("f_ext_flat") = Eigen::VectorXd(),
          "Articulated Body Algorithm — forward dynamics. "
          "``f_ext_flat`` is an optional (n_links*6,) array of "
          "spatial wrenches per link (link i at offset 6*i).");

    r.def("gravity_torques",
          [](const RbdTopology& topo,
             const Eigen::Ref<const Eigen::VectorXd>& q,
             const Eigen::Ref<const Eigen::Vector3d>& gravity) {
              return gravity_torques(topo, q, gravity);
          },
          py::arg("topo"), py::arg("q"), py::arg("gravity"),
          "Gravity compensation torques τ_g = RNEA(q, 0, 0, gravity).");
}

}  // namespace robosim
