// RoboSim C++ extension — pybind11 module entry point.
//
// Stage 0 (this file): hello-world `add(a, b)` plus a numpy round-trip
// (`square_array`) so we verify both the build pipeline AND the
// Eigen ↔ numpy buffer wiring before any real physics lands here.
//
// Subsequent stages append namespaces (rbd, contact, fem) under the
// same module; each gets its own translation unit + a register_*()
// hook called from PYBIND11_MODULE below.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Dense>

namespace py = pybind11;

namespace robosim_cpp {

// ── Stage-0 sanity ─────────────────────────────────────────────────────────

static double add(double a, double b) { return a + b; }

// Numpy round-trip: takes a (N,) double array, returns elementwise
// squared. Used to confirm the Eigen::Map ↔ py::array path is zero-copy
// on the input side and that allocation on the output side works.
static Eigen::VectorXd square_array(const Eigen::Ref<const Eigen::VectorXd>& x) {
    return x.array().square();
}

}  // namespace robosim_cpp

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "RoboSim C++ extension (pybind11 + Eigen). Hot-path solvers "
              "live in submodules added stage-by-stage; this module's "
              "Python-side entry points stay 1:1 with the pure-Python "
              "reference so tests can run on either backend.";

    m.attr("__stage__") = 0;
    m.def("add", &robosim_cpp::add,
          "Stage-0 smoke test: scalar addition.",
          py::arg("a"), py::arg("b"));
    m.def("square_array", &robosim_cpp::square_array,
          "Stage-0 numpy round-trip: elementwise square of a 1-D array.",
          py::arg("x"));
}
