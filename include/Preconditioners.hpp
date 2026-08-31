#ifndef PRECONDITIONERS_HPP
#define PRECONDITIONERS_HPP

#include "./IncludeFiles.hpp"

using namespace dealii;

// ==================================================================
// Preconditioners for the unsteady Navier-Stokes saddle point system
//
//     A = ( F   B^T )   x = ( U )   b = ( G )
//         ( -B   0  )       ( P )       ( 0 )
//
// as presented inside the paper "Parallel preconditioners for the
// unsteady Navier-Stokes equations and applications to hemodynamics
// simulations"
//
// Preconditioners implemented:
//   - PreconditionSIMPLE
//   - PreconditionApproxSIMPLE
//   - PreconditionYosida
//   - PreconditionApproxYosida
//   - PreconditionPCD
//   - PreconditionApproxPCD
//   - PreconditionBlockTriangular
//
// Naming convention for block matrices:
//   F    -> F
//   negB -> -B
//   B_t  -> B^T
//
// IMPORTANT - sign convention for Schur complement approximations:
//   Every "S" / "Schur_approx" passed around in this file is
//   assumed to already approximate the NEGATED Schur complement,
//   i.e. roughly "-S", consistently with the (2,2) block of every
//   factorization in the paper being of the form -S (or -S_tilde, -Ŝ).
//
// Lifetime note: the classes below store raw pointers to the external
// matrices passed to initialize(). The caller must guarantee those
// matrices stay alive (and are not reallocated) for as long as the
// preconditioner is used - this matters here because, per the paper,
// A and its blocks are rebuilt at every timestep.
// =================================================================

/**
 * \brief
 *
 * BlockPrecondition is an abstract base class for block preconditioners.
 * It provides a shared helper to initialize inner solvers.
 */
class BlockPrecondition
{
public:
    virtual ~BlockPrecondition() = default;
    virtual void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const = 0;

protected:
    void initialize_inner_preconditioner(
        std::shared_ptr<TrilinosWrappers::PreconditionBase> &preconditioner,
        const TrilinosWrappers::SparseMatrix &matrix, bool ilu)
    {
        if (ilu)
        {
            std::shared_ptr<TrilinosWrappers::PreconditionILU> actual_preconditioner =
                std::make_shared<TrilinosWrappers::PreconditionILU>();
            actual_preconditioner->initialize(matrix);
            preconditioner = actual_preconditioner;
        }
        else
        {
            std::shared_ptr<TrilinosWrappers::PreconditionAMG> actual_preconditioner =
                std::make_shared<TrilinosWrappers::PreconditionAMG>();
            actual_preconditioner->initialize(matrix);
            preconditioner = actual_preconditioner;
        }
    }
};

// ---------------------------------------------------------------
// Class: PreconditionSIMPLE
//
//   P_SIMPLE = ( F      0    ) ( I   D^-1 B^T )
//              ( B  -S_tilde ) ( 0   alpha I  ),
//
// S_tilde = B D^-1 B^T
// D = diag(F)
// alpha: (0, 1]
//
// F^-1 and S_tilde^-1 are applied via inner GMRES solves
// ---------------------------------------------------------------

/**
 * \brief
 *
 * The Semi-Implicit Method for Pressure Linked Equations (SIMPLE) is
 * an iterative method which first solves the momentum equation and then
 * updates the pressure field and the velocity field to conserve the mass
 * by using the continuity equation.
 * The method can be reinterpreted as if it were associated to the
 * previously introduced preconditioner.
 */
class PreconditionSIMPLE : public BlockPrecondition
{
public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                    const TrilinosWrappers::SparseMatrix &negB_,
                    const TrilinosWrappers::SparseMatrix &B_t_,
                    const TrilinosWrappers::MPI::BlockVector &vec,
                    const double &alpha_,
                    const unsigned int &maxit_,
                    const double &tol_,
                    const bool &ilu)
    {
        F = &F_;
        negB = &negB_;
        B_t = &B_t_;

        alpha = alpha_;
        maxit = maxit_;
        tol = tol_;

        negDinv.reinit(vec.block(0));
        for (unsigned int index : negDinv.locally_owned_elements())
        {
            negDinv[index] = -1.0 / F->diag_element(index);
        }

        negB->mmult(S, *B_t, negDinv);

        this->initialize_inner_preconditioner(preconditioner_F, *F, ilu);
        this->initialize_inner_preconditioner(preconditioner_S, S, ilu);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp.reinit(src);

        // F * sol1_u = src_u
        SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
        solver_F.solve(*F, tmp.block(0), src.block(0), *preconditioner_F);

        // S * sol1_p = B * sol1_u - src_p
        B_t->Tvmult(tmp.block(1), tmp.block(0));
        tmp.block(1) -= src.block(1);

        SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
        solver_S.solve(S, dst.block(1), tmp.block(1), *preconditioner_S);

        // Correct pressure
        dst.block(1) /= alpha;

        // Correct velocity  (dst_u = sol1_u - D^-1 * B^T * dst_p)
        dst.block(0) = tmp.block(0);
        B_t->vmult(tmp.block(0), dst.block(1));
        tmp.block(0).scale(negDinv);
        dst.block(0) += tmp.block(0);
    }

protected:
    double alpha;
    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *negB;
    const TrilinosWrappers::SparseMatrix *B_t;
    TrilinosWrappers::MPI::Vector negDinv;
    TrilinosWrappers::SparseMatrix S;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;
    mutable TrilinosWrappers::MPI::BlockVector tmp;
    unsigned int maxit;
    double tol;
};

// ---------------------------------------------------------------
// Class: PreconditionApproxSIMPLE
//
// Same factorization as PreconditionSIMPLE, but F^-1 and S_tilde^-1 are
// replaced by a SINGLE application of the embedded preconditioner
// (no inner GMRES loop). This is what allows using plain GMRES
// instead of FGMRES/GMRESR for the outer saddle point solve.
// initialize() is inherited unchanged from PreconditionSIMPLE.
// ---------------------------------------------------------------
class PreconditionApproxSIMPLE : public PreconditionSIMPLE
{
public:
    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp.reinit(src);

        // sol1_u ~ F^-1 * src_u  (single embedded-preconditioner apply)
        preconditioner_F->vmult(tmp.block(0), src.block(0));

        // sol1_p ~ S~^-1 * (B * sol1_u - src_p)
        B_t->Tvmult(tmp.block(1), tmp.block(0));
        tmp.block(1) -= src.block(1);
        preconditioner_S->vmult(dst.block(1), tmp.block(1));

        // Correct pressure
        dst.block(1) /= alpha;

        // Correct velocity (dst_u = sol1_u - D^-1 * B^T * dst_p)
        dst.block(0) = tmp.block(0);
        B_t->vmult(tmp.block(0), dst.block(1));
        tmp.block(0).scale(negDinv);
        dst.block(0) += tmp.block(0);
    }
};

// ---------------------------------------------------------------
// Class: PreconditionYosida
//
//   A = ( F  0 ) ( I  F^-1 B^T )
//       ( B  S ) ( 0     I     ),
//
// S ~ Dt * B * M_u^-1 * B^T is the Schur complement approximated
// M_u_matrix_ is the velocity mass matrix
// ---------------------------------------------------------------
class PreconditionYosida : public BlockPrecondition
{
public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                    const TrilinosWrappers::SparseMatrix &negB_,
                    const TrilinosWrappers::SparseMatrix &B_t_,
                    const TrilinosWrappers::SparseMatrix &M_u_,
                    const TrilinosWrappers::MPI::BlockVector &vec,
                    const double &dt_,
                    const unsigned int &maxit_,
                    const double &tol_,
                    const bool &ilu)
    {
        F = &F_;
        negB = &negB_;
        B_t = &B_t_;

        maxit = maxit_;
        tol = tol_;
        dt = dt_;

        Dinv.reinit(vec.block(0));
        for (unsigned int index : Dinv.locally_owned_elements())
        {
            // scaling vector = dt / diag(M_u)
            Dinv[index] = dt / M_u_.diag_element(index);
        }

        negB->mmult(negS, *B_t, Dinv);

        this->initialize_inner_preconditioner(preconditioner_F, *F, ilu);
        this->initialize_inner_preconditioner(preconditioner_S, negS, ilu);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp1.reinit(src);

        // F * sol1_u = src_u
        SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
        solver_F.solve(*F, tmp1.block(0), src.block(0), *preconditioner_F);

        // -S * sol1_p = src_p - B * sol1_u
        tmp1.block(1) = src.block(1);
        negB->vmult_add(tmp1.block(1), tmp1.block(0));

        SolverControl solver_control_S(maxit, tol * tmp1.block(1).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
        solver_S.solve(negS, dst.block(1), tmp1.block(1), *preconditioner_S);

        // [I, F^-1 * B^T; 0, I] * dst = sol1
        dst.block(0) = tmp1.block(0);
        B_t->vmult(tmp1.block(0), dst.block(1));

        tmp2.reinit(src.block(0));
        SolverControl solver_control_F2(maxit, tol * tmp1.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_gmres_F2(solver_control_F2);
        solver_gmres_F2.solve(*F, tmp2, tmp1.block(0), *preconditioner_F);

        dst.block(0) -= tmp2;
    }

protected:
    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *negB;
    const TrilinosWrappers::SparseMatrix *B_t;
    TrilinosWrappers::MPI::Vector Dinv;
    TrilinosWrappers::SparseMatrix negS;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;
    mutable TrilinosWrappers::MPI::BlockVector tmp1;
    mutable TrilinosWrappers::MPI::Vector tmp2;
    unsigned int maxit;
    double dt;
    double tol;
};

// ---------------------------------------------------------------
// Class: PreconditionApproxYosida
//
// Same factorization as PreconditionYosida, F^-1 and (-S)^-1 replaced
// by a single embedded-preconditioner application.
// initialize() is inherited unchanged from PreconditionYosida.
// ---------------------------------------------------------------
class PreconditionApproxYosida : public PreconditionYosida
{
public:
    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp1.reinit(src);

        // sol1_u ~ F^-1 * src_u
        preconditioner_F->vmult(tmp1.block(0), src.block(0));

        // sol1_p ~ (-S)^-1 * (src_p - B * sol1_u)
        tmp1.block(1) = src.block(1);
        negB->vmult_add(tmp1.block(1), tmp1.block(0));
        preconditioner_S->vmult(dst.block(1), tmp1.block(1));

        // dst_u = sol1_u - F^-1 * B^T * dst_p
        dst.block(0) = tmp1.block(0);
        B_t->vmult(tmp1.block(0), dst.block(1));

        tmp2.reinit(src.block(0));
        preconditioner_F->vmult(tmp2, tmp1.block(0));

        dst.block(0) -= tmp2;
    }
};

// ---------------------------------------------------------------
// Class: PreconditionPCD
//
//   P_PCD = ( F    B^T                )
//           ( 0   -M_p * F_p^-1 * A_p )
//
// The Schur complement approximation here is NOT a single sparse
// matrix but a composite operator (A_p multiply -> F_p solve ->
// M_p multiply), so it cannot be handled by PreconditionBlockTriangular
// and needs this dedicated class. F^-1, A_p^-1 and M_p^-1 are applied
// via inner GMRES solves.
// ---------------------------------------------------------------
class PreconditionPCD : public BlockPrecondition
{
public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                    const TrilinosWrappers::SparseMatrix &B_t_,
                    const TrilinosWrappers::SparseMatrix &Ap_,
                    const TrilinosWrappers::SparseMatrix &Mp_,
                    const TrilinosWrappers::SparseMatrix &Fp_,
                    const unsigned int &maxit_,
                    const double &tol_,
                    const bool &ilu)
    {
        F = &F_;
        B_t = &B_t_;
        Ap = &Ap_;
        Mp = &Mp_;
        Fp = &Fp_; // never inverted

        maxit = maxit_;
        tol = tol_;

        this->initialize_inner_preconditioner(preconditioner_F, *F, ilu);
        this->initialize_inner_preconditioner(preconditioner_Ap, *Ap, ilu);
        this->initialize_inner_preconditioner(preconditioner_Mp, *Mp, ilu);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp.reinit(src);

        // Solving (-M_p F_p^-1 A_p) * dst_p = src_p is equivalent to
        // dst_p = -A_p^-1 * F_p * M_p^-1 * src_p, applied in three steps:

        // q1 = M_p^-1 * src_p
        TrilinosWrappers::MPI::Vector q1(src.block(1));
        SolverControl solver_control_Mp(maxit, tol * src.block(1).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_Mp(solver_control_Mp);
        solver_Mp.solve(*Mp, q1, src.block(1), *preconditioner_Mp);

        // q2 = F_p * q1  (plain matrix-vector product, never inverted)
        TrilinosWrappers::MPI::Vector q2(src.block(1));
        Fp->vmult(q2, q1);

        // q3 = A_p^-1 * q2
        TrilinosWrappers::MPI::Vector q3(src.block(1));
        SolverControl solver_control_Ap(maxit, tol * q2.l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_Ap(solver_control_Ap);
        solver_Ap.solve(*Ap, q3, q2, *preconditioner_Ap);

        // dst_p = -q3
        dst.block(1) = q3;
        dst.block(1) *= -1.0;

        // tmp_u = src_u - B^T * dst_p
        TrilinosWrappers::MPI::Vector Btp(src.block(0));
        B_t->vmult(Btp, dst.block(1));
        tmp.block(0) = src.block(0);
        tmp.block(0) -= Btp;

        // F * dst_u = tmp_u
        SolverControl solver_control_F(maxit, tol * tmp.block(0).l2_norm());
        SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
        solver_F.solve(*F, dst.block(0), tmp.block(0), *preconditioner_F);
    }

protected:
    const TrilinosWrappers::SparseMatrix *F;   // F
    const TrilinosWrappers::SparseMatrix *B_t; // B^T
    const TrilinosWrappers::SparseMatrix *Ap;  // pressure pseudo-Laplacian
    const TrilinosWrappers::SparseMatrix *Mp;  // pressure mass matrix
    const TrilinosWrappers::SparseMatrix *Fp;  // pressure convection-diffusion op.
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_Ap;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_Mp;
    mutable TrilinosWrappers::MPI::BlockVector tmp;
    unsigned int maxit;
    double tol;
};

// ---------------------------------------------------------------
// Class: PreconditionApproxPCD
//
// Same factorization as PreconditionPCD, F^-1, A_p^-1 and M_p^-1
// replaced by a single embedded-preconditioner application each.
// Per Sec. 6 of the paper, M_p^-1 is best approximated by the
// inverse of the LUMPED diagonal of M_p (fast, embarrassingly
// parallel); build preconditioner_Mp accordingly if you want to
// reproduce the paper's exact aPCD setup.
// initialize() is inherited unchanged from PreconditionPCD.
// ---------------------------------------------------------------
class PreconditionApproxPCD : public PreconditionPCD
{
public:
    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp.reinit(src);

        // q1 ~ M_p^-1 * src_p
        TrilinosWrappers::MPI::Vector q1(src.block(1));
        preconditioner_Mp->vmult(q1, src.block(1));

        // q2 = F_p * q1
        TrilinosWrappers::MPI::Vector q2(src.block(1));
        Fp->vmult(q2, q1);

        // q3 ~ A_p^-1 * q2
        TrilinosWrappers::MPI::Vector q3(src.block(1));
        preconditioner_Ap->vmult(q3, q2);

        // dst_p = -q3
        dst.block(1) = q3;
        dst.block(1) *= -1.0;

        // tmp_u = src_u - B^T * dst_p
        TrilinosWrappers::MPI::Vector Btp(src.block(0));
        B_t->vmult(Btp, dst.block(1));
        tmp.block(0) = src.block(0);
        tmp.block(0) -= Btp;

        // dst_u ~ F^-1 * tmp_u
        preconditioner_F->vmult(dst.block(0), tmp.block(0));
    }
};

// ---------------------------------------------------------------
// Class: PreconditionBlockTriangular
//
// Description:
//   Generic upper/lower block triangular preconditioner for a
//   caller-supplied single-matrix Schur complement approximation
//   (e.g. an LSC-type approximation). Not one of the named methods
//   of the paper, but kept as a reusable building block: it is NOT
//   suitable for PCD, whose Schur approximation is a 3-operator
//   composite (use PreconditionPCD / PreconditionApproxPCD instead).
// ---------------------------------------------------------------
class PreconditionBlockTriangular : public BlockPrecondition
{
public:
    void initialize(const TrilinosWrappers::SparseMatrix &F_,
                    const TrilinosWrappers::SparseMatrix &negB_,
                    const TrilinosWrappers::SparseMatrix &B_t_,
                    const TrilinosWrappers::SparseMatrix &Schur_approx_,
                    const unsigned int &maxit_,
                    const double &tol_,
                    const bool &ilu,
                    const bool &is_upper_ = true)
    {
        F = &F_;
        neg_B = &negB_;
        B_t = &B_t_;
        Schur_approx = &Schur_approx_; // e.g. obtained via PCD/LSC-type approximations

        maxit = maxit_;
        tol = tol_;
        is_upper = is_upper_;

        this->initialize_inner_preconditioner(preconditioner_F, *F, ilu);
        this->initialize_inner_preconditioner(preconditioner_S, *Schur_approx, ilu);
    }

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const override
    {
        tmp.reinit(src);

        if (is_upper)
        {
            // Upper Triangular Block:
            // [ F  B^T ]^-1 approx [ F  0 ]^-1 * [ I  -B^T * S^-1 ]
            // [ 0  -S  ]           [ 0 -S ]      [ 0     I        ]

            // -S * dst_p = src_p
            SolverControl solver_control_S(maxit, tol * src.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(*Schur_approx, dst.block(1), src.block(1), *preconditioner_S);

            // tmp_u = src_u - B^T * dst_p
            TrilinosWrappers::MPI::Vector Btp(src.block(0));
            B_t->vmult(Btp, dst.block(1));
            tmp.block(0) = src.block(0);
            tmp.block(0) -= Btp;

            // F * dst_u = tmp_u
            SolverControl solver_control_F(maxit, tol * tmp.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
            solver_F.solve(*F, dst.block(0), tmp.block(0), *preconditioner_F);
        }
        else
        {
            // Lower Triangular Block:
            // [ F   0 ]^-1
            // [ B  -S ]

            // F * dst_u = src_u
            SolverControl solver_control_F(maxit, tol * src.block(0).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_F(solver_control_F);
            solver_F.solve(*F, dst.block(0), src.block(0), *preconditioner_F);

            // tmp_p = src_p - B * dst_u
            tmp.block(1) = src.block(1);
            neg_B->vmult_add(tmp.block(1), dst.block(0));

            // 3. Solve -S * dst_p = tmp_p
            SolverControl solver_control_S(maxit, tol * tmp.block(1).l2_norm());
            SolverGMRES<TrilinosWrappers::MPI::Vector> solver_S(solver_control_S);
            solver_S.solve(*Schur_approx, dst.block(1), tmp.block(1), *preconditioner_S);
        }
    }

private:
    bool is_upper;
    const TrilinosWrappers::SparseMatrix *F;
    const TrilinosWrappers::SparseMatrix *neg_B;
    const TrilinosWrappers::SparseMatrix *B_t;
    const TrilinosWrappers::SparseMatrix *Schur_approx;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_F;
    std::shared_ptr<TrilinosWrappers::PreconditionBase> preconditioner_S;
    mutable TrilinosWrappers::MPI::BlockVector tmp;
    unsigned int maxit;
    double tol;
};

#endif