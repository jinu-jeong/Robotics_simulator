// FEM corotational element assembly — see header for contract.
//
// Per-element pipeline (mirrors robosim/physics/fem/assembly.py):
//   F   = sum_a (x_def_a) ⊗ (dN_a)              deformation gradient
//   R,S = polar(F)  via SVD                     polar decomposition
//   eps = S - I
//   T   = 2μ eps + λ tr(eps) I
//   P   = R · T                                 1st PK stress
//   f_e[a] = -Σ_g w_g · P · dN_a                element internal force
//   K_e[a,b] = R · K0_ab · Rᵀ                   element stiffness
//             where K0_ab = μ(dN_a·dN_b) I3 + μ dN_b dN_aᵀ + λ dN_a dN_bᵀ
//
// All einsums collapse into hand-rolled 3×3 / Vector3d arithmetic.

#include "robosim/fem.hpp"

#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <stdexcept>

namespace robosim {

namespace {

inline void polar_decomposition(const Eigen::Matrix3d& F,
                                Eigen::Matrix3d& R,
                                Eigen::Matrix3d& S) {
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U  = svd.matrixU();
    Eigen::Matrix3d V  = svd.matrixV();
    Eigen::Vector3d s  = svd.singularValues();
    // Match numpy.linalg.svd: A = U S Vt, where Vt = V.T.
    const double detUV = U.determinant() * V.transpose().determinant();
    if (detUV < 0.0) {
        U.col(2) *= -1.0;
        s[2]     *= -1.0;
    }
    // Match the Python ref's clip to keep stiffness PSD on inverted elements.
    for (int i = 0; i < 3; ++i) if (s[i] < 0.1) s[i] = 0.1;

    R = U * V.transpose();
    Eigen::Matrix3d D = s.asDiagonal();
    S = V * D * V.transpose();
}

}  // namespace


void assemble_corotational(
    const Eigen::Ref<const MatRMd>& x,
    const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& elements,
    const Eigen::Ref<const MatRMd>& dN_flat,
    const Eigen::Ref<const MatRMd>& weights,
    double mu, double lam,
    int npe, int ng,
    Eigen::Ref<Eigen::VectorXd> f_out,
    Eigen::Ref<MatRMd> Ke_out
) {
    const int ne = static_cast<int>(elements.rows());
    const int ndof_e = npe * 3;

    f_out.setZero();
    Ke_out.setZero();

    // Per-element scratch (resized once).
    Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

    for (int e = 0; e < ne; ++e) {
        // Pre-extract element node positions once.
        // x_def (npe, 3).
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> x_def(npe, 3);
        for (int a = 0; a < npe; ++a) {
            x_def.row(a) = x.row(elements(e, a));
        }

        // Accumulators across Gauss points
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ke(ndof_e, ndof_e);
        Ke.setZero();
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> fe(npe, 3);
        fe.setZero();

        for (int g = 0; g < ng; ++g) {
            const double w_g = weights(e, g);

            // Fetch dN (npe, 3) for this (e, g) — base row in flattened buffer.
            const int base = (e * ng + g) * npe;
            // Build dN matrix view.
            Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> dN(npe, 3);
            for (int a = 0; a < npe; ++a) dN.row(a) = dN_flat.row(base + a);

            // F = x_def.T @ dN_a summed?  Actually:
            // numpy: F = einsum('eai,eaj->eij', x_def, dN)  → F_ij = Σ_a x_def[a,i] · dN[a,j]
            // i.e. F = x_def.T @ dN.
            Eigen::Matrix3d F = x_def.transpose() * dN;

            // Polar decomposition + clip
            Eigen::Matrix3d R, S;
            if (F.allFinite()) {
                polar_decomposition(F, R, S);
            } else {
                R = I3; S = I3;
            }

            // Corotational stress: T = 2μ(S-I) + λ tr(S-I) I; P = R · T.
            Eigen::Matrix3d eps = S - I3;
            const double tr = eps.trace();
            Eigen::Matrix3d T = 2.0 * mu * eps + lam * tr * I3;
            Eigen::Matrix3d P = R * T;

            // f_e[a] = - w_g · P · dN_a^T  (Python: H_all = w · einsum('eij,ekj->eik', P, dN))
            //   ⇒  H[i, a] = Σ_j P_ij · dN_aj  = (P · dN^T)[i, a]
            //   then f_e[a, :] -= H[:, a]
            Eigen::Matrix3Xd H = w_g * (P * dN.transpose());  // (3, npe)
            for (int a = 0; a < npe; ++a) {
                fe.row(a) -= H.col(a).transpose();
            }

            // Stiffness K_e blocks.
            for (int a = 0; a < npe; ++a) {
                Eigen::Vector3d dNa = dN.row(a).transpose();
                for (int b = 0; b < npe; ++b) {
                    Eigen::Vector3d dNb = dN.row(b).transpose();
                    const double dot_ab = dNa.dot(dNb);
                    // K0_ab = mu·dot·I + mu·(dNb dNa^T) + lam·(dNa dNb^T)
                    Eigen::Matrix3d K0 =
                          mu * dot_ab * I3
                        + mu * (dNb * dNa.transpose())
                        + lam * (dNa * dNb.transpose());
                    K0 *= w_g;
                    Eigen::Matrix3d Kab = R * K0 * R.transpose();
                    Ke.block<3, 3>(a * 3, b * 3) += Kab;
                }
            }
        }

        // Scatter fe into the global force vector (negation already in fe).
        for (int a = 0; a < npe; ++a) {
            const int node = elements(e, a);
            f_out(3 * node + 0) += fe(a, 0);
            f_out(3 * node + 1) += fe(a, 1);
            f_out(3 * node + 2) += fe(a, 2);
        }

        // Flatten Ke into Ke_out row.
        for (int i = 0; i < ndof_e; ++i) {
            for (int j = 0; j < ndof_e; ++j) {
                Ke_out(e, i * ndof_e + j) = Ke(i, j);
            }
        }
    }
}


Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>
batch_deformation_gradients(const Eigen::Ref<const MatRMd>& x,
                            const Eigen::Ref<const Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& elements,
                            const Eigen::Ref<const MatRMd>& dN_flat,
                            int npe) {
    const int ne = static_cast<int>(elements.rows());
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> F_flat(ne * 3, 3);
    for (int e = 0; e < ne; ++e) {
        Eigen::Matrix3d F = Eigen::Matrix3d::Zero();
        for (int a = 0; a < npe; ++a) {
            const int node = elements(e, a);
            Eigen::Vector3d xa = x.row(node).transpose();
            Eigen::Vector3d dNa = dN_flat.row(e * npe + a).transpose();
            F.noalias() += xa * dNa.transpose();
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F_flat(e * 3 + i, j) = F(i, j);
    }
    return F_flat;
}


void batch_polar(const Eigen::Ref<const MatRMd>& F_flat,
                 int ne,
                 Eigen::Ref<MatRMd> R_flat,
                 Eigen::Ref<MatRMd> S_flat) {
    Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    for (int e = 0; e < ne; ++e) {
        Eigen::Matrix3d F;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                F(i, j) = F_flat(e * 3 + i, j);

        Eigen::Matrix3d R, S;
        if (F.allFinite()) {
            polar_decomposition(F, R, S);
        } else {
            R = I3; S = I3;
        }
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                R_flat(e * 3 + i, j) = R(i, j);
                S_flat(e * 3 + i, j) = S(i, j);
            }
    }
}


void batch_corotational_stiffness(const Eigen::Ref<const MatRMd>& R_flat,
                                  const Eigen::Ref<const MatRMd>& dN_flat,
                                  const Eigen::Ref<const Eigen::VectorXd>& weights,
                                  double mu, double lam,
                                  int ne, int npe,
                                  Eigen::Ref<MatRMd> Ke_out) {
    Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
    const int ndof_e = npe * 3;

    for (int e = 0; e < ne; ++e) {
        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = R_flat(e * 3 + i, j);
        const double w = weights(e);

        // Pre-extract per-element dN.
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> dN(npe, 3);
        for (int a = 0; a < npe; ++a)
            for (int k = 0; k < 3; ++k)
                dN(a, k) = dN_flat(e * npe + a, k);

        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Ke(ndof_e, ndof_e);
        Ke.setZero();
        for (int a = 0; a < npe; ++a) {
            Eigen::Vector3d dNa = dN.row(a).transpose();
            for (int b = 0; b < npe; ++b) {
                Eigen::Vector3d dNb = dN.row(b).transpose();
                const double dot_ab = dNa.dot(dNb);
                Eigen::Matrix3d K0 =
                      mu * dot_ab * I3
                    + mu * (dNb * dNa.transpose())
                    + lam * (dNa * dNb.transpose());
                K0 *= w;
                Eigen::Matrix3d Kab = R * K0 * R.transpose();
                Ke.block<3, 3>(a * 3, b * 3) = Kab;
            }
        }

        for (int i = 0; i < ndof_e; ++i)
            for (int j = 0; j < ndof_e; ++j)
                Ke_out(e, i * ndof_e + j) = Ke(i, j);
    }
}


// ── Cached SPD factorisation ────────────────────────────────────────

void SparseSPDFactor::analyze(const Eigen::Ref<const Eigen::VectorXi>& indptr,
                              const Eigen::Ref<const Eigen::VectorXi>& indices,
                              const Eigen::Ref<const Eigen::VectorXd>& values_template,
                              int n) {
    n_ = n;
    // Build SparseMatrix from triplets (handles unsorted CSR safely).
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(values_template.size());
    for (int i = 0; i < n; ++i) {
        const int s = indptr[i];
        const int e = indptr[i + 1];
        for (int k = s; k < e; ++k) {
            triplets.emplace_back(i, indices[k], values_template[k]);
        }
    }
    A_.resize(n, n);
    A_.setFromTriplets(triplets.begin(), triplets.end());
    A_.makeCompressed();

    // Build CSR-k → Eigen internal-k permutation.
    // Eigen stores CSC by default: outerIndexPtr[col]..outerIndexPtr[col+1]
    // ranges hold rows in that column. We walk input CSR (row-major) and
    // look up each (i, j) in the compressed CSC storage.
    csr_to_eigen_.assign(values_template.size(), -1);
    const int* outer = A_.outerIndexPtr();
    const int* inner = A_.innerIndexPtr();
    int csr_k = 0;
    for (int i = 0; i < n; ++i) {
        const int s = indptr[i];
        const int e = indptr[i + 1];
        for (int k = s; k < e; ++k, ++csr_k) {
            const int j = indices[k];
            // In CSC for entry (i, j) → search inner[outer[j]..outer[j+1]]
            // for row == i.
            const int cs = outer[j];
            const int ce = outer[j + 1];
            int found = -1;
            for (int p = cs; p < ce; ++p) {
                if (inner[p] == i) { found = p; break; }
            }
            csr_to_eigen_[csr_k] = found;  // -1 if structurally zero
        }
    }

    ldlt_.analyzePattern(A_);
    spd_path_ = (ldlt_.info() == Eigen::Success);
    if (!spd_path_) {
        lu_ = std::make_unique<Eigen::SparseLU<SparseMat>>();
        lu_->analyzePattern(A_);
    }
    ready_ = true;
}

void SparseSPDFactor::factorize(const Eigen::Ref<const Eigen::VectorXd>& data) {
    if (!ready_)
        throw std::runtime_error("SparseSPDFactor::factorize called before analyze");
    // Refresh numerical values in-place. Eigen's SparseMatrix exposes
    // its valuePtr directly when compressed; CSR data layout matches
    // what Python passed at analyze time (same triplet order ⇒ Eigen
    // sorts identically). Robust path: rebuild from triplets — cheap
    // because analyzePattern's symbolic factor doesn't repeat.
    if (static_cast<size_t>(data.size()) != csr_to_eigen_.size())
        throw std::runtime_error("SparseSPDFactor: nnz mismatch on factorize");
    // Scatter incoming CSR values into Eigen's internal storage via
    // the permutation built at analyze() time. Multiple CSR entries
    // mapping to the same Eigen slot (deduplicated triplets) get
    // summed — matches scipy's coo→csr sum_duplicates contract.
    double* vp = A_.valuePtr();
    for (Eigen::Index k = 0; k < A_.nonZeros(); ++k) vp[k] = 0.0;
    for (size_t k = 0; k < csr_to_eigen_.size(); ++k) {
        const int dst = csr_to_eigen_[k];
        if (dst >= 0) vp[dst] += data[k];
    }

    if (spd_path_) {
        ldlt_.factorize(A_);
        if (ldlt_.info() != Eigen::Success) {
            // Drop to LU once and stay there for this factor's lifetime.
            spd_path_ = false;
            lu_ = std::make_unique<Eigen::SparseLU<SparseMat>>();
            lu_->analyzePattern(A_);
        }
    }
    if (!spd_path_) {
        lu_->factorize(A_);
        if (lu_->info() != Eigen::Success)
            throw std::runtime_error("SparseSPDFactor: LU factorize failed");
    }
}

Eigen::VectorXd SparseSPDFactor::solve(
        const Eigen::Ref<const Eigen::VectorXd>& b) {
    if (!ready_)
        throw std::runtime_error("SparseSPDFactor::solve called before analyze");
    if (spd_path_) return ldlt_.solve(b);
    return lu_->solve(b);
}


// ── Conjugate Gradient with warm start ──────────────────────────────

void SparseCG::analyze(const Eigen::Ref<const Eigen::VectorXi>& indptr,
                       const Eigen::Ref<const Eigen::VectorXi>& indices,
                       const Eigen::Ref<const Eigen::VectorXd>& values_template,
                       int n) {
    n_ = n;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(values_template.size());
    for (int i = 0; i < n; ++i) {
        const int s = indptr[i];
        const int e = indptr[i + 1];
        for (int k = s; k < e; ++k)
            triplets.emplace_back(i, indices[k], values_template[k]);
    }
    A_.resize(n, n);
    A_.setFromTriplets(triplets.begin(), triplets.end());
    A_.makeCompressed();

    // Build CSR-k → Eigen internal-k permutation, same pattern as the
    // direct factor cache (see SparseSPDFactor::analyze).
    csr_to_eigen_.assign(values_template.size(), -1);
    const int* outer = A_.outerIndexPtr();
    const int* inner = A_.innerIndexPtr();
    int csr_k = 0;
    for (int i = 0; i < n; ++i) {
        const int s = indptr[i];
        const int e = indptr[i + 1];
        for (int k = s; k < e; ++k, ++csr_k) {
            const int j = indices[k];
            const int cs = outer[j];
            const int ce = outer[j + 1];
            int found = -1;
            for (int p = cs; p < ce; ++p)
                if (inner[p] == i) { found = p; break; }
            csr_to_eigen_[csr_k] = found;
        }
    }
    ready_ = true;
}

std::tuple<Eigen::VectorXd, int, bool>
SparseCG::solve(const Eigen::Ref<const Eigen::VectorXd>& data,
                const Eigen::Ref<const Eigen::VectorXd>& b,
                const Eigen::Ref<const Eigen::VectorXd>& x_warm,
                double tol,
                int max_iter) {
    if (!ready_) throw std::runtime_error("SparseCG::solve before analyze");
    if (static_cast<size_t>(data.size()) != csr_to_eigen_.size())
        throw std::runtime_error("SparseCG: nnz mismatch on solve");

    // Refresh matrix values in-place (scatter via permutation, summing
    // duplicates — same as SparseSPDFactor::factorize).
    double* vp = A_.valuePtr();
    for (Eigen::Index k = 0; k < A_.nonZeros(); ++k) vp[k] = 0.0;
    for (size_t k = 0; k < csr_to_eigen_.size(); ++k) {
        const int dst = csr_to_eigen_[k];
        if (dst >= 0) vp[dst] += data[k];
    }

    // Diagonal preconditioner: cheap, works well for FEM stiffness +
    // mass-matrix combinations where the diagonal dominates.
    Eigen::ConjugateGradient<SparseMat, Eigen::Lower | Eigen::Upper,
                              Eigen::DiagonalPreconditioner<double>> cg;
    cg.compute(A_);
    cg.setTolerance(tol);
    cg.setMaxIterations(max_iter);

    Eigen::VectorXd x;
    if (x_warm.size() == b.size()) {
        x = cg.solveWithGuess(b, x_warm);
    } else {
        x = cg.solve(b);
    }
    const int iters = static_cast<int>(cg.iterations());
    const bool converged = (cg.info() == Eigen::Success);
    return {x, iters, converged};
}


Eigen::VectorXd sparse_spd_solve(
    const Eigen::Ref<const Eigen::VectorXi>& indptr,
    const Eigen::Ref<const Eigen::VectorXi>& indices,
    const Eigen::Ref<const Eigen::VectorXd>& data,
    int n,
    const Eigen::Ref<const Eigen::VectorXd>& b) {
    // Build Eigen::SparseMatrix from CSR. Eigen stores CSC by default;
    // for SPD systems CSR == CSC. We construct from triplets to be
    // robust to non-sorted indices.
    using SparseMat = Eigen::SparseMatrix<double>;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(data.size());
    for (int i = 0; i < n; ++i) {
        const int s = indptr[i];
        const int e = indptr[i + 1];
        for (int k = s; k < e; ++k) {
            triplets.emplace_back(i, indices[k], data[k]);
        }
    }
    SparseMat A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Try SimplicialLDLT first (SPD fast path). If factor fails (matrix
    // not SPD due to numerical issue), fall back to SparseLU.
    Eigen::SimplicialLDLT<SparseMat> ldlt(A);
    if (ldlt.info() == Eigen::Success) {
        Eigen::VectorXd x = ldlt.solve(b);
        if (ldlt.info() == Eigen::Success) return x;
    }
    Eigen::SparseLU<SparseMat> lu(A);
    if (lu.info() != Eigen::Success)
        throw std::runtime_error("sparse_spd_solve: LU factorisation failed");
    Eigen::VectorXd x = lu.solve(b);
    if (lu.info() != Eigen::Success)
        throw std::runtime_error("sparse_spd_solve: LU solve failed");
    return x;
}

}  // namespace robosim
