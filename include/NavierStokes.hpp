#ifndef NAVIER_STOKES_HPP
#define NAVIER_STOKES_HPP

#include "./Preconditioners.hpp"

using namespace dealii;

// Class implementing a solver for the Navier-Stokes problem.
// Dealing the problem with template in order to have dim = 2 or dim = 3
template <unsigned int dim>
class NavierStokes
{
static_assert(dim == 2 || dim == 3, "Navier Stokes has been implemented only for 2D and 3D case") public :

    // Function for inlet velocity. This actually returns an object with four
    // components (one for each velocity component, and one for the pressure), but
    // then only the first three are really used (see the component mask when
    // applying boundary conditions at the end of assembly). If we only return
    // three components, however, we may get an error message due to this function
    // being incompatible with the finite element space.

    // create enum (?)
    class InletVelocity : public Function<dim>
    {
    public:
        InletVelocity()
            : Function<dim>(dim + 1)
        {
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
        }

    protected:
    };

    // Since we're working with block matrices, we need to make our own
    // preconditioner class. A preconditioner class can be any class that exposes
    // a vmult method that applies the inverse of the preconditioner.

    // Constructor.
    NavierStokes(const std::string &mesh_file_name_,
                 const unsigned int &degree_velocity_,
                 const unsigned int &degree_pressure_,
                 const double &final_time_,
                 const double &time_step_size_)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
          mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
          pcout(std::cout, mpi_rank == 0),
          mesh_file_name(mesh_file_name_),
          degree_velocity(degree_velocity_), // velocity space
          degree_pressure(degree_pressure_), // pressure space
          final_time(final_time_)
              time_step_size(time_step_size_),
          mesh(MPI_COMM_WORLD)
    {
    }

    // Setup system.
    void
    setup();

    // Assemble system. We also assemble the pressure mass matrix (needed for the
    // preconditioner).
    void
    assemble();

    // Solve system.
    void
    solve();

    // Output results.
    void
    output();

protected:
    // MPI parallel. /////////////////////////////////////////////////////////////

    // Number of MPI processes.
    const unsigned int mpi_size;

    // This MPI process.
    const unsigned int mpi_rank;

    // Parallel output stream.
    ConditionalOStream pcout;

    // Problem definition. ///////////////////////////////////////////////////////

    // Kinematic viscosity [m2/s].
    const double kinematic_viscosity = 1e-3;

    // Density
    const double density = 1.;

    // Inlet velocity
    InletVelocity inlet_velocity;

    // Forcing term
    Functions::ZeroFunction<dim> forcing term;

    // Final time
    const double final_time;

    // Discretization. ///////////////////////////////////////////////////////////

    // Mesh file name.
    const std::string mesh_file_name;

    // Polynomial degree used for velocity.
    const unsigned int degree_velocity;

    // Polynomial degree used for pressure.
    const unsigned int degree_pressure;

    // Time step
    const double time_step_size;

    // Initial condition.
    Functions::ZeroFunction<dim> initial_solution_function;

    // Dirichlet lifting function (g(x)).
    Functions::ZeroFunction<dim> dirichlet_lifting_function;

    // Neumann data function (h(x)).
    Functions::ZeroFunction<dim> neumann_data_function;

    // Mesh.
    parallel::fullydistributed::Triangulation<dim> mesh;

    // Finite element space.
    std::unique_ptr<FiniteElement<dim>> fe;

    // Quadrature formula.
    std::unique_ptr<Quadrature<dim>> quadrature;

    // Quadrature formula for face integrals.
    std::unique_ptr<Quadrature<dim - 1>> quadrature_face;

    // DoF handler.
    DoFHandler<dim> dof_handler;

    // DoFs owned by current process.
    IndexSet locally_owned_dofs;

    // DoFs owned by current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_owned_dofs;

    // DoFs relevant to the current process (including ghost DoFs).
    IndexSet locally_relevant_dofs;

    // DoFs relevant to current process in the velocity and pressure blocks.
    std::vector<IndexSet> block_relevant_dofs;

    // System matrix.
    TrilinosWrappers::BlockSparseMatrix system_matrix;

    // Mass matrix
    TrilinosWrappers::BlockSparseMatrix mass_matrix;

    // Stifness matrix
    TrilinosWrappers::BlockSparseMatrix stifness_matrix;

    // Convection matrix
    TrilinosWrappers::BlockSparseMatrix convection_matrix;

    // Pressure mass matrix, needed for preconditioning. We use a block matrix for
    // convenience, but in practice we only look at the pressure-pressure block.
    TrilinosWrappers::BlockSparseMatrix pressure_mass;

    // Right-hand side vector in the linear system. [U P]
    TrilinosWrappers::MPI::BlockVector system_rhs;

    // System solution (without ghost elements).
    TrilinosWrappers::MPI::BlockVector solution_owned;

    // System solution (including ghost elements).
    TrilinosWrappers::MPI::BlockVector solution;

    // System previous solution
    TrilinosWrappers::MPI::BlockVector previous_solution;
};

#endif