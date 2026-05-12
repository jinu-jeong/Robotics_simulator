// pybind11 bindings for the FK chain — exposes ``robosim._cpp.kin``
// with a single ``Topology`` Python-side handle and a free function
// ``forward_kinematics(topo, q) -> (R_array, t_array)``.

#include "robosim/kinematics.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

namespace py = pybind11;

namespace robosim {

void register_kinematics(py::module_& m) {
    auto k = m.def_submodule("kin",
        "Forward kinematics on a precomputed Topology. One C++ call "
        "per step replaces the per-joint Python BFS in "
        "``robosim/model/robot.py::forward_kinematics``.");

    py::enum_<JointKind>(k, "JointKind")
        .value("FIXED",     JointKind::FIXED)
        .value("REVOLUTE",  JointKind::REVOLUTE)
        .value("PRISMATIC", JointKind::PRISMATIC)
        .export_values();

    py::class_<JointSpec>(k, "JointSpec")
        .def(py::init<>())
        .def_readwrite("parent_link", &JointSpec::parent_link)
        .def_readwrite("child_link",  &JointSpec::child_link)
        .def_readwrite("kind",        &JointSpec::kind)
        .def_readwrite("dof_index",   &JointSpec::dof_index)
        .def_readwrite("axis",        &JointSpec::axis)
        .def_readwrite("origin_R",    &JointSpec::origin_R)
        .def_readwrite("origin_t",    &JointSpec::origin_t);

    py::class_<Topology>(k, "Topology")
        .def(py::init<>())
        .def_readwrite("n_links", &Topology::n_links)
        .def_readwrite("n_dof",   &Topology::n_dof)
        .def_readwrite("joints",  &Topology::joints)
        .def_readwrite("root_links", &Topology::root_links);

    k.def("forward_kinematics",
          [](const Topology& topo,
             const Eigen::Ref<const Eigen::VectorXd>& q) {
              const int n = topo.n_links;
              MatRMXd R_out(n, 9);   // row-major: row(l).data() contiguous
              MatRMXd t_out(n, 3);
              forward_kinematics(topo, q, R_out, t_out);
              return std::make_tuple(R_out, t_out);
          },
          py::arg("topo"), py::arg("q"),
          "Compute per-link world-frame (R, t).  Returns "
          "(R_flat: (n_links, 9), t_flat: (n_links, 3)) — caller reshapes "
          "each R row to (3, 3) row-major.");
}

}  // namespace robosim
