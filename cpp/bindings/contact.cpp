// pybind11 bindings for the contact narrow-phase kernels.

#include "robosim/contact.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace robosim {

void register_contact(py::module_& m) {
    auto c = m.def_submodule("contact",
        "Contact narrow-phase kernels (sphere/box vs ground & sphere/box "
        "vs sphere/box). Each call returns a packed (N, 10) RowMajor "
        "matrix [point_a(3), point_b(3), normal(3), penetration]. "
        "N=0 ⇒ no contact.");

    c.def("sphere_ground", &sphere_ground,
          py::arg("center"), py::arg("radius"),
          py::arg("ground_height"), py::arg("ground_normal"));

    c.def("sphere_sphere", &sphere_sphere,
          py::arg("c1"), py::arg("r1"),
          py::arg("c2"), py::arg("r2"));

    c.def("box_sphere", &box_sphere,
          py::arg("box_center"), py::arg("box_rot"), py::arg("box_half"),
          py::arg("sphere_center"), py::arg("sphere_radius"));

    c.def("box_ground", &box_ground,
          py::arg("center"), py::arg("rotation"), py::arg("half_extents"),
          py::arg("ground_height"), py::arg("ground_normal"));

    c.def("box_box", &box_box,
          py::arg("center_a"), py::arg("rot_a"), py::arg("half_a"),
          py::arg("center_b"), py::arg("rot_b"), py::arg("half_b"));

    c.def("cylinder_ground", &cylinder_ground,
          py::arg("center"), py::arg("rotation"),
          py::arg("radius"), py::arg("half_length"),
          py::arg("ground_height"), py::arg("ground_normal"),
          py::arg("n_ring"));

    c.def("aabb_box", &aabb_box,
          py::arg("rotation"), py::arg("translation"), py::arg("half_extents"));
    c.def("aabb_sphere", &aabb_sphere,
          py::arg("translation"), py::arg("radius"));
    c.def("aabb_cylinder", &aabb_cylinder,
          py::arg("rotation"), py::arg("translation"),
          py::arg("radius"), py::arg("length"));

    c.def("contact_force_penalty", &contact_force_penalty,
          py::arg("normal"), py::arg("v_a"), py::arg("v_b"),
          py::arg("penetration"),
          py::arg("stiffness"), py::arg("damping"),
          py::arg("mu"), py::arg("friction_eps"),
          py::arg("max_penetration"),
          py::arg("effective_mass"));
}

}  // namespace robosim
