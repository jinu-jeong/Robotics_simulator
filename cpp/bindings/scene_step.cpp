// pybind11 bindings for the substep-level scene orchestrator.

#include "robosim/scene_step.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace robosim {

void register_scene_step(py::module_& m) {
    auto s = m.def_submodule("scene_step",
        "Substep-level rigid scene orchestration (Stage 9). Python "
        "registers robots + collision bodies once; ``step(dt, n)`` "
        "runs the substep loop entirely inside C++.");

    py::class_<RigidSceneStep>(s, "RigidSceneStep")
        .def(py::init<>())
        .def("add_robot",
             [](RigidSceneStep& self, RbdTopology& topo,
                const Eigen::Ref<const Eigen::Vector3d>& gravity,
                bool is_free) {
                 return self.add_robot(&topo, gravity, is_free);
             },
             py::arg("topo"), py::arg("gravity"), py::arg("is_free") = false,
             py::keep_alive<1, 2>())
        .def("set_pd_control", &RigidSceneStep::set_pd_control,
             py::arg("robot_idx"), py::arg("kp"), py::arg("kd"))
        .def("set_q_target", &RigidSceneStep::set_q_target,
             py::arg("robot_idx"), py::arg("q_target"))
        .def("set_tau", &RigidSceneStep::set_tau,
             py::arg("robot_idx"), py::arg("tau"))
        .def("set_state", &RigidSceneStep::set_state,
             py::arg("robot_idx"), py::arg("q"), py::arg("qd"))
        .def("get_state", &RigidSceneStep::get_state,
             py::arg("robot_idx"))
        .def("add_box", &RigidSceneStep::add_box,
             py::arg("robot_idx"), py::arg("link_idx"),
             py::arg("half_extents"), py::arg("origin_R"), py::arg("origin_t"))
        .def("add_sphere", &RigidSceneStep::add_sphere,
             py::arg("robot_idx"), py::arg("link_idx"),
             py::arg("radius"), py::arg("origin_R"), py::arg("origin_t"))
        .def("add_cylinder", &RigidSceneStep::add_cylinder,
             py::arg("robot_idx"), py::arg("link_idx"),
             py::arg("radius"), py::arg("length"),
             py::arg("origin_R"), py::arg("origin_t"))
        .def("add_filter", &RigidSceneStep::add_filter,
             py::arg("body_id_a"), py::arg("body_id_b"))
        .def("set_ground", &RigidSceneStep::set_ground,
             py::arg("height"), py::arg("normal"))
        .def("set_contact_params", &RigidSceneStep::set_contact_params,
             py::arg("stiffness"), py::arg("damping"), py::arg("mu"),
             py::arg("friction_eps"), py::arg("max_penetration"))
        .def("step", &RigidSceneStep::step,
             py::arg("dt"), py::arg("n_substeps"))
        .def("n_robots",  &RigidSceneStep::n_robots)
        .def("n_bodies",  &RigidSceneStep::n_bodies);
}

}  // namespace robosim
