#ifndef NAVIER_STOKES_HPP
#define NAVIER_STOKES_HPP

#include "./Preconditioners.hpp"

using namespace dealii;

enum class InflowRegime
{
    Steady,
    Unsteady
};

// Class implementing a solver for the Navier-Stokes problem.
// Dealing the problem with template in order to have dim = 2 or dim = 3
template <unsigned int dim>
class NavierStokes
{

static_assert(dim == 2 || dim == 3, "Navier Stokes has been implemented only for 2D and 3D case")

    public :

    static constexpr types::boundary_id id_inlet = 1;
    static constexpr types::boundary_id id_outlet = 2;
    static constexpr types::boundary_id id_walls = 3;
    static constexpr types::boundary_id id_obstacle = 4;

    class InletVelocity : public Function<dim>
    {
    public:
        InletVelocity(const InflowRegime regime_ = InflowRegime::Steady, const double peak_velocity_)
            : Function<dim>(dim + 1), regime(regime_), peak_velocity(peak_velocity_)
        {
        }

        virtual void
        vector_value(const Point<dim> &p, Vector<double> &values) const override
        {
            values[0] = inflow_profile(p);
            for (unsigned int i = 1; i < dim; ++i)
            {
                values[i] = 0.0;
            }
        }

        virtual double
        value(const Point<dim> &p, const unsigned int component = 0) const override
        {
            if (component == 0)
            {
                return inflow_profile(p);
            }
            else
            {
                return 0.0;
            }
        }

        double getMeanVelocity() const
        {
            const double mean_coefficient = (dim == 2) ? (2.0 / 3.0) : (4.0 / 9.0);
            return mean_coefficient * peak_velocity * temporal_envelope(this->get_time());
        }

    protected:
        double temporal_envelope(double time) const
        {
            switch (regime)
            {
            case InflowRegime::Steady:
                return 1.;
            case InflowRegime::Unsteady:
                return std::sin(M_PI * time / 8.0);
            }
        }

        double inflow_profile(const Point<dim> &p) const
        {
            const double time_component = temporal_envelope(this->get_time());

            if constexpr (dim == 2)
            {
                return 4. * peak_velocity * p[1] * (channel_height - p[1]) * time_component / (channel_height * channel_height);
            }
            else
            {
                return 16. * peak_velocity * p[1] * p[2] * (channel_heigth - p[1]) * (channel_height - p[2]) * time_component /
                       (channel_height * channel_height * channel_height * channel_height);
            }
        }
        const double peak_velocity;
        const InflowRegime regime;
        const double height_channel = 0.41;
    };

    // Constructor.
    NavierStokes(const std::string &mesh_file_name_,
                 const unsigned int &degree_velocity_,
                 const unsigned int &degree_pressure_,
                 const double &final_time_,
                 const double &time_step_size_,
                 const double &peak_velocity,
                 const InflowRegime regime_ = InflowRegime::Steady)
        : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
          mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)),
          pcout(std::cout, mpi_rank == 0),
          mesh_file_name(mesh_file_name_),
          degree_velocity(degree_velocity_),
          degree_pressure(degree_pressure_),
          final_time(final_time_),
          time_step_size(time_step_size_),
          inlet_velocity(regime_, peak_velocity_),
          mesh(MPI_COMM_WORLD)
    {
    }

    // Setup system.
    void
    setup();

    // Assemble system.
    void
    assemble(const double &time);

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