#include "../include/NavierStokes.hpp"

// Main function.
int main(int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    int size;
    MPI_Comm_size(mpi_comm, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Mesh File
    const std::string mesh_file_name = "../mesh/Navier_Stokes_2D_coarse.msh";

    // Taylor Hood Elements
    const unsigned int degree_velocity = 2;
    const unsigned int degree_pressure = 1;

    // Time variables

    // Stokes problem(mesh_file_name, degree_velocity, degree_pressure);

    return 0;
}