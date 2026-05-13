// pybind11 bindings for the FEM corotational assembler.

#include "robosim/fem.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace robosim {

void register_fem(py::module_& m) {
    auto f = m.def_submodule("fem",
        "FEM hot-path assembly (corotational element stiffness + force) "
        "as one C++ call per Newton iteration. Mirrors "
        "robosim/physics/fem/assembly.py's batched einsum path.");

    f.def("assemble_corotational",
          [](const Eigen::Ref<const MatRMd>& x,
             const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& elements,
             const Eigen::Ref<const MatRMd>& dN_flat,
             const Eigen::Ref<const MatRMd>& weights,
             double mu, double lam,
             int npe, int ng,
             int n_dof) {
              Eigen::VectorXd f_vec = Eigen::VectorXd::Zero(n_dof);
              const int ne = static_cast<int>(elements.rows());
              const int ndof_e = npe * 3;
              MatRMd Ke_out = MatRMd::Zero(ne, ndof_e * ndof_e);
              assemble_corotational(x, elements, dN_flat, weights,
                                    mu, lam, npe, ng, f_vec, Ke_out);
              return std::make_tuple(f_vec, Ke_out);
          },
          py::arg("x"), py::arg("elements"), py::arg("dN_flat"),
          py::arg("weights"), py::arg("mu"), py::arg("lam"),
          py::arg("npe"), py::arg("ng"), py::arg("n_dof"),
          "Assemble per-element corotational stiffness + internal force. "
          "Returns (f, Ke_all) where Ke_all is (ne, ndof_e*ndof_e).");

    f.def("batch_polar",
          [](const Eigen::Ref<const MatRMd>& F_flat) {
              const int ne = static_cast<int>(F_flat.rows()) / 3;
              MatRMd R_flat = MatRMd::Zero(ne * 3, 3);
              MatRMd S_flat = MatRMd::Zero(ne * 3, 3);
              batch_polar(F_flat, ne, R_flat, S_flat);
              return std::make_tuple(R_flat, S_flat);
          },
          py::arg("F_flat"),
          "Batched 3×3 polar decomposition via SVD. Input/Output: "
          "(ne*3, 3) row-major flatten of (ne, 3, 3). Returns (R, S).");

    f.def("batch_deformation_gradients", &batch_deformation_gradients,
          py::arg("x"), py::arg("elements"), py::arg("dN_flat"), py::arg("npe"),
          "F = x_def^T · dN per element. Returns (ne*3, 3) flat.");

    py::class_<SparseCG>(f, "SparseCG",
        "Iterative SPD solver (Conjugate Gradient + diagonal "
        "preconditioner) with warm-start support. analyze() once, "
        "then solve(data, b, x_warm) per Newton iter.",
        py::dynamic_attr())
        .def(py::init<>())
        .def("analyze", &SparseCG::analyze,
             py::arg("indptr"), py::arg("indices"),
             py::arg("values_template"), py::arg("n"))
        .def("solve", &SparseCG::solve,
             py::arg("data"), py::arg("b"), py::arg("x_warm"),
             py::arg("tol") = 1e-8, py::arg("max_iter") = 200)
        .def("ready", &SparseCG::ready)
        .def("n", &SparseCG::n);

    py::class_<SparseSPDFactor>(f, "SparseSPDFactor",
        "Cached SPD sparse factorisation. analyze() once, then "
        "factorize(data) per Newton iter and solve(b) per call.",
        py::dynamic_attr())
        .def(py::init<>())
        .def("analyze", &SparseSPDFactor::analyze,
             py::arg("indptr"), py::arg("indices"),
             py::arg("values_template"), py::arg("n"))
        .def("factorize", &SparseSPDFactor::factorize, py::arg("data"))
        .def("solve", &SparseSPDFactor::solve, py::arg("b"))
        .def("ready", &SparseSPDFactor::ready)
        .def("n", &SparseSPDFactor::n);

    f.def("sparse_spd_solve", &sparse_spd_solve,
          py::arg("indptr"), py::arg("indices"), py::arg("data"),
          py::arg("n"), py::arg("b"),
          "SPD sparse linear solve via SimplicialLDLT (LU fallback). "
          "Input CSR triple (indptr, indices, data) of an (n, n) matrix.");

    f.def("batch_corotational_stiffness",
          [](const Eigen::Ref<const MatRMd>& R_flat,
             const Eigen::Ref<const MatRMd>& dN_flat,
             const Eigen::Ref<const Eigen::VectorXd>& weights,
             double mu, double lam, int npe) {
              const int ne = static_cast<int>(weights.size());
              const int ndof_e = npe * 3;
              MatRMd Ke_out = MatRMd::Zero(ne, ndof_e * ndof_e);
              batch_corotational_stiffness(R_flat, dN_flat, weights,
                                           mu, lam, ne, npe, Ke_out);
              return Ke_out;
          },
          py::arg("R_flat"), py::arg("dN_flat"), py::arg("weights"),
          py::arg("mu"), py::arg("lam"), py::arg("npe"),
          "Per-element corotational stiffness (R · K0 · R^T) summed "
          "across the supplied npe×npe block pairs. R_flat: (ne*3, 3); "
          "dN_flat: (ne*npe, 3); weights: (ne,). Returns Ke_all (ne, "
          "ndof_e*ndof_e) row-major flatten of (ndof_e, ndof_e).");
}

}  // namespace robosim
