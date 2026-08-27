#ifndef PRECONDITIONERS_HPP
#define PRECONDITIONERS_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <functional>

using namespace dealii;

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

// Block-diagonal preconditioner.
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

#endif
