// pybind11 bindings for the SE(3) hot-path utilities defined in
// ``robosim/transform.hpp``.
//
// Style: free functions that take + return raw 3-vectors / 3×3
// matrices via Eigen. The Python ``Transform`` dataclass keeps its
// ``rotation``/``translation`` numpy attributes and dispatches its
// per-operation methods here (see ``robosim/math/transforms.py``);
// this keeps the Python-facing API stable.

#include "robosim/transform.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

namespace py = pybind11;

namespace robosim {

void register_transform(py::module_& m) {
    auto t = m.def_submodule("transform",
        "SE(3) hot-path helpers (cross3, skew, rotation_{x,y,z}, "
        "from_axis_angle, compose, inverse, apply_*). All operands are "
        "passed/returned as plain numpy arrays — no C++ Transform "
        "object is exposed; the Python ``Transform`` dataclass wraps "
        "these.");

    t.def("cross3",
          [](const Eigen::Ref<const Vec3>& a,
             const Eigen::Ref<const Vec3>& b) { return cross3(a, b); },
          py::arg("a"), py::arg("b"));

    t.def("skew",
          [](const Eigen::Ref<const Vec3>& v) { return skew(v); },
          py::arg("v"));

    t.def("rotation_x", &rotation_x, py::arg("angle"));
    t.def("rotation_y", &rotation_y, py::arg("angle"));
    t.def("rotation_z", &rotation_z, py::arg("angle"));

    t.def("from_axis_angle",
          [](const Eigen::Ref<const Vec3>& axis, double angle) {
              return from_axis_angle(axis, angle);
          },
          py::arg("axis"), py::arg("angle"));

    t.def("compose_Rt",
          [](const Eigen::Ref<const Mat3>& R1,
             const Eigen::Ref<const Vec3>& t1,
             const Eigen::Ref<const Mat3>& R2,
             const Eigen::Ref<const Vec3>& t2) {
              Mat3 Rout; Vec3 tout;
              compose_Rt(R1, t1, R2, t2, Rout, tout);
              return std::make_tuple(Rout, tout);
          },
          py::arg("R1"), py::arg("t1"), py::arg("R2"), py::arg("t2"),
          "Returns (R, t) = (R1·R2, R1·t2 + t1).");

    t.def("inverse_Rt",
          [](const Eigen::Ref<const Mat3>& R,
             const Eigen::Ref<const Vec3>& tt) {
              Mat3 Rout; Vec3 tout;
              inverse_Rt(R, tt, Rout, tout);
              return std::make_tuple(Rout, tout);
          },
          py::arg("R"), py::arg("t"),
          "Returns (R^T, -R^T · t).");

    t.def("apply_point",
          [](const Eigen::Ref<const Mat3>& R,
             const Eigen::Ref<const Vec3>& tt,
             const Eigen::Ref<const Vec3>& p) {
              return apply_point(R, tt, p);
          },
          py::arg("R"), py::arg("t"), py::arg("p"));

    t.def("apply_vector",
          [](const Eigen::Ref<const Mat3>& R,
             const Eigen::Ref<const Vec3>& v) {
              return apply_vector(R, v);
          },
          py::arg("R"), py::arg("v"));
}

}  // namespace robosim
