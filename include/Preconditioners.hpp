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
//   - PreconditionYosida
//   - PreconditionBlockTriangular
//
// Naming convention for block matrices:
//   F    -> F
//   negB -> -B
//   B_t  -> B^T
//
//
// Lifetime note: the classes below store raw pointers to the external
// matrices passed to initialize(). The caller must guarantee those
// matrices stay alive (and are not reallocated) for as long as the
// preconditioner is used - this matters here because, per the paper,
// A and its blocks are rebuilt at every timestep.
// =================================================================


// Block-diagonal preconditioner. Its an abstract base class for block preconditioners
// It provides a shared helper to initialize inner solver


// Identity preconditioner.
class PreconditionIdentity
{
public:
    // Application of the preconditioner: we just copy the input vector (src)
    // into the output vector (dst).
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
        dst = src;
    }

protected:
};


class PreconditionBlockDiagonal
{
public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_)
    {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass = &pressure_mass_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    { // CG solver
        SolverControl solver_control_velocity(1000,
                                              1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
            solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                 dst.block(0),
                                 src.block(0),
                                 preconditioner_velocity);

        SolverControl solver_control_pressure(1000,
                                              1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
            solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                 dst.block(1),
                                 src.block(1),
                                 preconditioner_pressure);
    }

protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;
};

// Block-triangular preconditioner.
class PreconditionBlockTriangular
{
public:
    // Initialize the preconditioner, given the velocity stiffness matrix, the
    // pressure mass matrix.
    void
    initialize(const TrilinosWrappers::SparseMatrix &velocity_stiffness_,
               const TrilinosWrappers::SparseMatrix &pressure_mass_,
               const TrilinosWrappers::SparseMatrix &B_)
    {
        velocity_stiffness = &velocity_stiffness_;
        pressure_mass = &pressure_mass_;
        B = &B_;

        preconditioner_velocity.initialize(velocity_stiffness_);
        preconditioner_pressure.initialize(pressure_mass_);
    }

    // Application of the preconditioner.
    void
    vmult(TrilinosWrappers::MPI::BlockVector &dst,
          const TrilinosWrappers::MPI::BlockVector &src) const
    {
        SolverControl solver_control_velocity(1000,
                                              1e-2 * src.block(0).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_velocity(
            solver_control_velocity);
        solver_cg_velocity.solve(*velocity_stiffness,
                                 dst.block(0),
                                 src.block(0),
                                 preconditioner_velocity);

        tmp.reinit(src.block(1));
        B->vmult(tmp, dst.block(0));
        tmp.sadd(-1.0, src.block(1));

        SolverControl solver_control_pressure(1000,
                                              1e-2 * src.block(1).l2_norm());
        SolverCG<TrilinosWrappers::MPI::Vector> solver_cg_pressure(
            solver_control_pressure);
        solver_cg_pressure.solve(*pressure_mass,
                                 dst.block(1),
                                 tmp,
                                 preconditioner_pressure);
    }

protected:
    // Velocity stiffness matrix.
    const TrilinosWrappers::SparseMatrix *velocity_stiffness;

    // Preconditioner used for the velocity block.
    TrilinosWrappers::PreconditionILU preconditioner_velocity;

    // Pressure mass matrix.
    const TrilinosWrappers::SparseMatrix *pressure_mass;

    // Preconditioner used for the pressure block.
    TrilinosWrappers::PreconditionILU preconditioner_pressure;

    // B matrix.
    const TrilinosWrappers::SparseMatrix *B;

    // Temporary vector.
    mutable TrilinosWrappers::MPI::Vector tmp;
};


// ==================================================================
// Physical Preconditioners for the unsteady Navier-Stokes
// (SIMPLE and Yosida implementations)
// =================================================================

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

#endif