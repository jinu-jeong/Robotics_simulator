// FEM corotational element assembly — port of the einsum-heavy paths
// in robosim/physics/fem/assembly.py. The whole per-element pipeline
// (deformation gradient → polar decomp → stress → element force +
// stiffness) collapses into one tight C++ loop, removing the per-call
// numpy/einsum dispatch overhead that dominated the FEM profile.
//
// Input/output buffers are row-major (numpy C-order) so the boundary
// stays zero-copy.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <cstdint>
#include <memory>
#include <tuple>

namespace robosim {

// Reusable SPD sparse factorisation: pattern analysed once on
// ``analyze``, numerical factor refreshed per Newton iter via
// ``factorize``. ``solve(b)`` runs the back-substitution. Matches
// Eigen::SimplicialLDLT's two-step API while exposing a stable
// Python handle so the integrator can keep the analyzed factor
// across substeps.

class SparseSPDFactor {
 public:
    SparseSPDFactor() = default;

    // Analyze sparsity. Must be called before factorize/solve.
    // values_template carries the (zero-pattern) numerical entries —
    // Eigen needs the matrix structure but doesn't yet rely on values.
    void analyze(const Eigen::Ref<const Eigen::VectorXi>& indptr,
                 const Eigen::Ref<const Eigen::VectorXi>& indices,
                 const Eigen::Ref<const Eigen::VectorXd>& values_template,
                 int n);

    // Numerically refactorise. Caller must pass *the same* sparsity
    // pattern that was analyzed; only ``data`` changes.
    void factorize(const Eigen::Ref<const Eigen::VectorXd>& data);

    // Solve A x = b using the cached factorisation.
    Eigen::VectorXd solve(const Eigen::Ref<const Eigen::VectorXd>& b);

    bool ready() const { return ready_; }
    int n() const { return n_; }

 private:
    using SparseMat = Eigen::SparseMatrix<double>;
    SparseMat A_;
    Eigen::SimplicialLDLT<SparseMat> ldlt_;
    // LU fallback for when matrix turns out to not be SPD after a
    // numerical refresh (e.g. plastic softening). Kept as an opaque
    // ptr to avoid the cost when the SPD path stays in use.
    std::unique_ptr<Eigen::SparseLU<SparseMat>> lu_;
    bool ready_ = false;
    bool spd_path_ = true;
    int n_ = 0;
    // Permutation: input CSR k-th entry → Eigen valuePtr index.
    // Built in analyze, applied in factorize so the value-only
    // refresh stays correct regardless of how Eigen reordered the
    // pattern internally.
    std::vector<int> csr_to_eigen_;
};


// Iterative SPD solver — Conjugate Gradient with Incomplete Cholesky
// preconditioner. Supports warm-starting from a previous solution
// (typically the dx from the previous Newton iteration), which is
// what makes CG competitive with direct factorisation for the
// implicit-FEM Newton loop: the matrix shifts only a little between
// iters so a good guess slashes the iteration count.
//
// Same pattern-then-values lifecycle as :class:`SparseSPDFactor`:
//   analyze(indptr, indices, values_template, n)   one-shot pattern
//   solve(data, b, x_warm, tol, max_iter)          per-call solve
//
// Returns (x, n_iters). n_iters == 0 means warm start was already at
// tolerance; n_iters == max_iter (with non-converged info) signals
// the caller may want to drop back to direct factorisation.
class SparseCG {
 public:
    SparseCG() = default;

    void analyze(const Eigen::Ref<const Eigen::VectorXi>& indptr,
                 const Eigen::Ref<const Eigen::VectorXi>& indices,
                 const Eigen::Ref<const Eigen::VectorXd>& values_template,
                 int n);

    // Returns (x, n_iters, converged_flag).
    std::tuple<Eigen::VectorXd, int, bool>
    solve(const Eigen::Ref<const Eigen::VectorXd>& data,
          const Eigen::Ref<const Eigen::VectorXd>& b,
          const Eigen::Ref<const Eigen::VectorXd>& x_warm,
          double tol,
          int max_iter);

    bool ready() const { return ready_; }
    int n() const { return n_; }

 private:
    using SparseMat = Eigen::SparseMatrix<double>;
    SparseMat A_;
    bool ready_ = false;
    int n_ = 0;
    std::vector<int> csr_to_eigen_;
};

using MatRMd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Corotational-elastic assembly. Iterates each element, integrates
// across the supplied Gauss points, accumulates internal force +
// element stiffness.
//
// Inputs (row-major numpy):
//   x        : (n_nodes, 3)        current positions
//   elements : (ne, npe)           int32 connectivity (0-indexed)
//   dN_all   : (ne, ng, npe, 3)    shape-function gradients per Gauss pt
//                                   flattened as (ne*ng*npe, 3)
//   weights  : (ne, ng)
//   mu, lam  : Lamé parameters
//   ne, ng, npe : structural dimensions (caller-derived)
//   n_nodes  : caller-derived
//
// Outputs (preallocated):
//   f        : (n_dof,) = (n_nodes*3,)   internal force vector
//   Ke_all   : (ne, ndof_e * ndof_e)     per-element stiffness, row-major
//                                         flatten of (ndof_e, ndof_e)
//
// Returns nothing; caller assembles Ke_all into the sparse global K.
void assemble_corotational(
    const Eigen::Ref<const MatRMd>& x,                   // (n_nodes, 3)
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& elements, // (ne, npe)
    const Eigen::Ref<const MatRMd>& dN_flat,             // (ne*ng*npe, 3)
    const Eigen::Ref<const MatRMd>& weights,             // (ne, ng)
    double mu, double lam,
    int npe, int ng,
    Eigen::Ref<Eigen::VectorXd> f_out,                   // (n_dof,)
    Eigen::Ref<MatRMd> Ke_out                            // (ne, ndof_e*ndof_e)
);

// Batched deformation gradient F = x_def^T · dN per element.
//   x      : (n_nodes, 3)
//   elements: (ne, npe) int32
//   dN_flat: (ne*npe, 3) row-major flatten of (ne, npe, 3)
// Returns:
//   F_flat: (ne*3, 3) row-major flatten of (ne, 3, 3)
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
batch_deformation_gradients(const Eigen::Ref<const MatRMd>& x,
                            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& elements,
                            const Eigen::Ref<const MatRMd>& dN_flat,
                            int npe);

// Batch SVD-based polar decomposition for ne 3×3 deformation gradients.
// F_flat: (ne*3, 3) row-major flatten of (ne, 3, 3).
// Outputs:
//   R_flat: (ne*3, 3)
//   S_flat: (ne*3, 3)
// Matches numpy.linalg.svd convention (A = U S Vt) with the same
// determinant-flip / singular-value-clip rules as the Python reference.
void batch_polar(const Eigen::Ref<const MatRMd>& F_flat,
                 int ne,
                 Eigen::Ref<MatRMd> R_flat,
                 Eigen::Ref<MatRMd> S_flat);

// Sparse SPD linear solve: A x = b for an SPD matrix A given in CSR
// (row-major).
//
// SimplicialLDLT keeps the cost on par with SuperLU but avoids the
// per-call SciPy dispatch + indices-sort overhead that dominated the
// FEM profile.
//
// Inputs:
//   A_indptr, A_indices, A_data  : CSR (n, n)
//   b                            : (n,)
// Output: x (n,)
//
// Returns x. Throws via pybind on factorisation failure (e.g. A not SPD).
Eigen::VectorXd sparse_spd_solve(
    const Eigen::Ref<const Eigen::VectorXi>& indptr,
    const Eigen::Ref<const Eigen::VectorXi>& indices,
    const Eigen::Ref<const Eigen::VectorXd>& data,
    int n,
    const Eigen::Ref<const Eigen::VectorXd>& b);

// Per-element corotational stiffness — port of
// ``_batch_corotational_stiffness`` (the dominant einsum-heavy kernel).
//   R_all  : (ne*3, 3)        from batch_polar
//   dN_all : (ne, npe, 3)     flattened (ne*npe, 3)
//   weights: (ne,)
//   Output: Ke_out : (ne, ndof_e*ndof_e) flat each (ndof_e, ndof_e)
void batch_corotational_stiffness(const Eigen::Ref<const MatRMd>& R_flat,
                                  const Eigen::Ref<const MatRMd>& dN_flat,
                                  const Eigen::Ref<const Eigen::VectorXd>& weights,
                                  double mu, double lam,
                                  int ne, int npe,
                                  Eigen::Ref<MatRMd> Ke_out);

}  // namespace robosim
