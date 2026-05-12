// RoboSim C++ extension — pybind11 module entry point.
//
// Each stage adds a submodule under the same ``_cpp`` package. Bindings
// for that submodule live in their own translation unit + a
// ``register_<name>(module&)`` hook called from PYBIND11_MODULE.
//
//   stage 0  smoke   ``add``, ``square_array``   (build pipeline sanity)
//   stage 1  transform  SE(3) hot-path helpers   (compose, inverse, FK atoms)
//   stage 2  …

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

namespace py = pybind11;

namespace robosim {
void register_transform(py::module_& m);   // bindings/transform.cpp
void register_kinematics(py::module_& m);  // bindings/kinematics.cpp
}

namespace robosim_cpp {

static double add(double a, double b) { return a + b; }

static Eigen::VectorXd square_array(const Eigen::Ref<const Eigen::VectorXd>& x) {
    return x.array().square();
}

}  // namespace robosim_cpp

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "RoboSim C++ extension (pybind11 + Eigen). Hot-path solvers "
              "live in submodules added stage-by-stage; this module's "
              "Python-side entry points stay 1:1 with the pure-Python "
              "reference so tests can run on either backend.";

    m.attr("__stage__") = 2;
    m.def("add", &robosim_cpp::add,
          "Stage-0 smoke test: scalar addition.",
          py::arg("a"), py::arg("b"));
    m.def("square_array", &robosim_cpp::square_array,
          "Stage-0 numpy round-trip: elementwise square of a 1-D array.",
          py::arg("x"));

    robosim::register_transform(m);
    robosim::register_kinematics(m);
}
