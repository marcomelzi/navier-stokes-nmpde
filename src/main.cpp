#include "../include/NavierStokes.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int size;
    MPI_Comm_size(mpi_comm, &size);

    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    // Mesh File
    const std::string mesh_file_name = "../mesh/Navier_Stokes_2D_coarse.msh";

    // Taylor Hood Elements
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    InflowRegime regime = InflowRegime::Unsteady;
    Preconditioner preconditioner = Preconditioner::YOSIDA;

    // Time variables
    const double T = 8.0;
    const double dt = 0.002;

    dealii::Timer timer;
    timer.restart();

    NavierStokes<2> problem(mesh_file_name, degree_velocity, degree_pressure, T, dt, regime, preconditioner);

    problem.setup();
    problem.run();

    timer.stop();

    if (rank == 0)
        std::cout << "Time taken to solve Navier Stokes problem: " << timer.wall_time() << " seconds" << std::endl;

    return 0;
}