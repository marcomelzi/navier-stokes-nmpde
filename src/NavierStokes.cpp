#include "../include/NavierStokes.hpp"

// setup ()
template <unsigned int dim>
void NavierStokes<dim>::setup()
{
    // Create the mesh.
    {
        pcout << "Initializing the mesh" << std::endl;

        Triangulation<dim> mesh_serial;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(mesh_serial);

        std::ifstream grid_in_file(mesh_file_name);
        grid_in.read_msh(grid_in_file);

        GridTools::partition_triangulation(mpi_size, mesh_serial);
        const auto construction_data = TriangulationDescription::Utilities::
            create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
        mesh.create_triangulation(construction_data);

        pcout << "  Number of elements = " << mesh.n_global_active_cells()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the finite element space.
    {
        pcout << "Initializing the finite element space" << std::endl;

        const FE_SimplexP<dim> fe_scalar_velocity(degree_velocity);
        const FE_SimplexP<dim> fe_scalar_pressure(degree_pressure);
        fe = std::make_unique<FESystem<dim>>(fe_scalar_velocity,
                                             dim,
                                             fe_scalar_pressure,
                                             1);

        pcout << "  Velocity degree:           = " << fe_scalar_velocity.degree
              << std::endl;
        pcout << "  Pressure degree:           = " << fe_scalar_pressure.degree
              << std::endl;
        pcout << "  DoFs per cell              = " << fe->dofs_per_cell
              << std::endl;

        quadrature = std::make_unique<QGaussSimplex<dim>>(fe->degree + 1);

        pcout << "  Quadrature points per cell = " << quadrature->size()
              << std::endl;

        quadrature_face = std::make_unique<QGaussSimplex<dim - 1>>(fe->degree + 1);

        pcout << "  Quadrature points per face = " << quadrature_face->size()
              << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the DoF handler.
    {
        pcout << "Initializing the DoF handler" << std::endl;

        dof_handler.reinit(mesh);
        dof_handler.distribute_dofs(*fe);

        // We want to reorder DoFs so that all velocity DoFs come first, and then
        // all pressure DoFs.
        std::vector<unsigned int> block_component(dim + 1, 0);
        block_component[dim] = 1;
        DoFRenumbering::component_wise(dof_handler, block_component);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        // Besides the locally owned and locally relevant indices for the whole
        // system (velocity and pressure), we will also need those for the
        // individual velocity and pressure blocks.
        std::vector<types::global_dof_index> dofs_per_block =
            DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
        const unsigned int n_u = dofs_per_block[0];
        const unsigned int n_p = dofs_per_block[1];

        block_owned_dofs.resize(2);
        block_relevant_dofs.resize(2);
        block_owned_dofs[0] = locally_owned_dofs.get_view(0, n_u);
        block_owned_dofs[1] = locally_owned_dofs.get_view(n_u, n_u + n_p);
        block_relevant_dofs[0] = locally_relevant_dofs.get_view(0, n_u);
        block_relevant_dofs[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

        pcout << "  Number of DoFs: " << std::endl;
        pcout << "    velocity = " << n_u << std::endl;
        pcout << "    pressure = " << n_p << std::endl;
        pcout << "    total    = " << n_u + n_p << std::endl;
    }

    pcout << "-----------------------------------------------" << std::endl;

    // Initialize the linear system.
    {
        pcout << "Initializing the linear system" << std::endl;

        pcout << "  Initializing the sparsity pattern" << std::endl;

        // Velocity DoFs interact with other velocity DoFs (the weak formulation has
        // terms involving u times v), and pressure DoFs interact with velocity DoFs
        // (there are terms involving p times v or u times q). However, pressure
        // DoFs do not interact with other pressure DoFs (there are no terms
        // involving p times q). We build a table to store this information, so that
        // the sparsity pattern can be built accordingly.
        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::none;
                else // other combinations
                    coupling[c][d] = DoFTools::always;
            }
        }

        TrilinosWrappers::BlockSparsityPattern sparsity(block_owned_dofs,
                                                        MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler, coupling, sparsity);
        sparsity.compress();

        // We also build a sparsity pattern for the pressure mass matrix.
        for (unsigned int c = 0; c < dim + 1; ++c)
        {
            for (unsigned int d = 0; d < dim + 1; ++d)
            {
                if (c == dim && d == dim) // pressure-pressure term
                    coupling[c][d] = DoFTools::always;
                else // other combinations
                    coupling[c][d] = DoFTools::none;
            }
        }
        TrilinosWrappers::BlockSparsityPattern sparsity_pressure_mass(
            block_owned_dofs, MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(dof_handler,
                                        coupling,
                                        sparsity_pressure_mass);
        sparsity_pressure_mass.compress();

        pcout << "  Initializing the matrices" << std::endl;
        system_matrix.reinit(sparsity);
        mass_matrix.reinit(sparsity);
        convection_matrix.reinit(sparsity);
        stiffness_matrix.reinit(sparsity);
        pressure_mass.reinit(sparsity_pressure_mass);

        pcout << "  Initializing the system right-hand side" << std::endl;
        system_rhs.reinit(block_owned_dofs, MPI_COMM_WORLD);
        pcout << "  Initializing the solution vector" << std::endl;
        solution_owned.reinit(block_owned_dofs, MPI_COMM_WORLD);
        solution.reinit(block_owned_dofs, block_relevant_dofs, MPI_COMM_WORLD);
    }
}

template <unsigned int dim>
void NavierStokes<dim>::assemble(const double &time)
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_convection_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_pressure_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix = 0.0;
    mass_matrix = 0.0;
    stiffness_matrix = 0.0;
    convection_matrix = 0.0;
    system_rhs = 0.0;
    pressure_mass = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    // Store the current velocity value
    std::vector<Tensor<1, dim>> current_velocity_values(n_q);
    // Store the current velocity gradient value
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q);
    // Store the current velocity divergence value
    std::vector<double> current_velocity_divergence(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_matrix = 0.0;
        cell_mass_matrix = 0.0;
        cell_stiffness_matrix = 0.0;
        cell_convection_matrix = 0.0;
        cell_rhs = 0.0;
        cell_pressure_mass_matrix = 0.0;

        // Retrieve the current solution values.
        fe_values[velocity].get_function_values(solution, current_velocity_values);
        // Retrieve the current solution gradient values
        fe_values[velocity].get_function_gradients(solution, current_velocity_gradients);
        // Retrieve the current solution divergence values
        fe_values[velocity].get_function_divergences(solution, current_velocity_divergence);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            Vector<double> forcing_term_loc(dim);
            forcing_term.vector_value(fe_values.quadrature_point(q), forcing_term_loc);

            Tensor<1, dim> forcing_term_tensor;

            for (unsigned int i = 0; i < dim; ++i)
            {
                forcing_term_tensor[i] = forcing_term_loc[i];
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Viscosity term
                    cell_stiffness_matrix(i, j) += kinematic_viscosity * scalar_product(fe_values[velocity].gradient(i, q), fe_values[velocity].gradient(j, q)) * fe_values.JxW(q);

                    // Time derivative discretization.
                    cell_mass_matrix(i, j) += scalar_product(fe_values[velocity].value(i, q), fe_values[velocity].value(j, q)) / time_step_size * fe_values.JxW(q);

                    // Convective term
                    cell_convection_matrix(i, j) += scalar_product(fe_values[velocity].gradient(j, q) * current_velocity_values[q], fe_values[velocity].value(i, q)) * fe_values.JxW(q);

                    // Temam Stabilization term
                    cell_convection_matrix(i, j) += 0.5 * current_velocity_divergence[q] * scalar_product(fe_values[velocity].value(i, q), fe_values[velocity].value(j, q)) * fe_values.JxW(q);

                    // Pressure term in the momentum equation.
                    cell_matrix(i, j) -= fe_values[pressure].value(j, q) * fe_values[velocity].divergence(i, q) * fe_values.JxW(q);

                    // Pressure term in the continuity equation.
                    cell_matrix(i, j) += fe_values[pressure].value(i, q) * fe_values[velocity].divergence(j, q) * fe_values.JxW(q);

                    // Pressure mass matrix.
                    cell_pressure_mass_matrix(i, j) += fe_values[pressure].value(i, q) * fe_values[pressure].value(j, q) / kinematic_viscosity * fe_values.JxW(q);
                }

                // Time derivative on the right hand side
                cell_rhs(i) += scalar_product(current_velocity_values[q], fe_values[velocity].value(i, q)) * fe_values.JxW(q) / time_step_size;
            }
        }

        // Boundary integral for Neumann BCs absent

        cell->get_dof_indices(dof_indices);

        system_matrix.add(dof_indices, cell_matrix);
        mass_matrix.add(dof_indices, cell_mass_matrix);
        convection_matrix.add(dof_indices, cell_convection_matrix);
        stiffness_matrix.add(dof_indices, cell_stiffness_matrix);
        system_rhs.add(dof_indices, cell_rhs);
        pressure_mass.add(dof_indices, cell_pressure_mass_matrix);
    }

    system_matrix.compress(VectorOperation::add);
    mass_matrix.compress(VectorOperation::add);
    convection_matrix.compress(VectorOperation::add);
    stiffness_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);

    // Create the System Matrix M + A + C(u_n) + B
    system_matrix.add(1., mass_matrix);
    system_matrix.add(1., convection_matrix);
    system_matrix.add(1., stiffness_matrix);

    // Dirichlet boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        // velocity only component + mask
        std::vector<bool> velocity_mask_vec(dim + 1, true);
        velocity_mask_vec[dim] = false;
        const ComponentMask mask_velocity(velocity_mask_vec);

        // We interpolate first the inlet velocity condition alone, then the wall
        // condition alone, so that the latter "win" over the former where the two
        // boundaries touch.
        inlet_velocity.set_time(time);
        boundary_functions[id_inlet] = &inlet_velocity;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 mask_velocity);

        boundary_functions.clear();
        Functions::ZeroFunction<dim> zero_function(dim + 1);

        boundary_functions[id_walls] = &zero_function;
        boundary_functions[id_obstacle] = &zero_function;

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 mask_velocity);

        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_owned, system_rhs, false);
    }
}

template <unsigned int dim>
void NavierStokes<dim>::assemble_time_step(const double &time)
{
    pcout << "===============================================" << std::endl;
    pcout << "Assembling the system" << std::endl;

    const unsigned int dofs_per_cell = fe->dofs_per_cell;
    const unsigned int n_q = quadrature->size();

    FEValues<dim> fe_values(*fe,
                            *quadrature,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    FullMatrix<double> cell_convection_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

    system_matrix.add(-1., convection_matrix);

    convection_matrix = 0.0;
    system_rhs = 0.0;

    FEValuesExtractors::Vector velocity(0);
    FEValuesExtractors::Scalar pressure(dim);

    // Current velocity
    std::vector<Tensor<1, dim>> current_velocity_values(n_q);
    // Current velocity gradient
    std::vector<Tensor<2, dim>> current_velocity_gradients(n_q);
    // Current velocity divergence
    std::vector<double> current_velocity_divergence(n_q);

    // Previous velocity
    std::vector<Tensor<1, dim>> previous_velocity_values(n_q);
    // Previous velocity gradient
    std::vector<Tensor<2, dim>> previous_velocity_gradients(n_q);
    // Previous velocity divergence
    std::vector<double> previous_velocity_divergence(n_q);
    // Previous divergence values
    std::vector<double> previous_divergence_values(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        if (!cell->is_locally_owned())
            continue;

        fe_values.reinit(cell);

        cell_mass_matrix = 0.0;
        cell_convection_matrix = 0.0;
        cell_rhs = 0.0;

        // current
        fe_values[velocity].get_function_values(solution, current_velocity_values);
        fe_values[velocity].get_function_gradients(solution, current_velocity_gradients);
        fe_values[velocity].get_function_divergences(solution, current_velocity_divergence);

        // previous
        fe_values[velocity].get_function_values(previous_solution, previous_velocity_values);
        fe_values[velocity].get_function_divergences(previous_solution, previous_divergence_values);

        for (unsigned int q = 0; q < n_q; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    // Convective term
                    cell_convection_matrix(i, j) += scalar_product(fe_values[velocity].gradient(j, q) * current_velocity_values[q], fe_values[velocity].value(i, q)) * fe_values.JxW(q);
                }
                // Time derivative discretization on the right hand side - Backward Euler
                cell_rhs(i) += scalar_product(current_velocity_values[q], fe_values[velocity].value(i, q)) * fe_values.JxW(q) / time_step_size;
            }
        }

        cell->get_dof_indices(dof_indices);
        convection_matrix.add(dof_indices, cell_convection_matrix);
        system_rhs.add(dof_indices, cell_rhs);
    }

    convection_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
    pressure_mass.compress(VectorOperation::add);
    system_matrix.add(1., convection_matrix);

    // Apply Dirichlet boundary conditions.
    {
        std::map<types::global_dof_index, double> boundary_values;
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;

        std::vector<bool> velocity_mask_vec(dim + 1, true);
        velocity_mask_vec[dim] = false;
        const ComponentMask velocity_mask(velocity_mask_vec);

        // Inlet
        inlet_velocity.set_time(time);
        boundary_functions[id_inlet] = &inlet_velocity;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 velocity_mask);

        // No overlap
        boundary_functions.clear();
        Functions::ZeroFunction<dim> zero_function(dim + 1);

        // Walls and the Obstacle.
        boundary_functions[id_walls] = &zero_function;
        boundary_functions[id_obstacle] = &zero_function;
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 boundary_functions,
                                                 boundary_values,
                                                 velocity_mask);

        MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs, false);
    }
}

template <unsigned int dim>
void NavierStokes<dim>::solve_time_step(const Preconditioner &preconditioner)
{
    pcout << "===============================================" << std::endl;

    SolverControl solver_control(100000, 1e-6, true);

    SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control);

    // Preconditioners
    {
        switch (preconditioner)
        {
        case (Preconditioner::YOSIDA):
            PreconditionYosida prec;
            prec.initialize(system_matrix.block(0, 0), system_matrix.block(1, 0), system_matrix.block(0, 1), mass_matrix.block(0, 0), solution_owned);
            solver.solve(system_matrix, solution_owned, system_rhs, prec);
            break;
        case (Preconditioner::SIMPLE):
            PreconditionSIMPLE prec;
            prec.initialize(system_matrix.block(0, 0), system_matrix.block(1, 0), system_matrix.block(0, 1), solution_owned);
            solver.solve(system_matrix, solution_owned, system_rhs, prec);
            break;
            /*case (Preconditioner::BLOCK_TRIANGULAR):
                PreconditionBlockTriangular prec;
                prec.initialize(system_matrix.block(0, 0), system_matrix.block(1, 0), system_matrix.block(0, 1), mass_matrix.block(0, 0), solution_owned);
                solver.solve(system_matrix, solution_owned, system_rhs, prec);
                break;*/
        }
    }

    pcout << "Result:  " << solver_control.last_step() << " GMRES iterations" << std::endl;
    solution = solution_owned;
}

template <unsigned int dim>
void NavierStokes<dim>::output(const unsigned int &time)
{
    pcout << "===============================================" << std::endl;

    DataOut<dim> data_out;

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(dim,
                       DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> names(dim, "velocity");
    names.push_back("pressure");

    data_out.add_data_vector(dof_handler, solution, names, interpretation);

    std::vector<unsigned int> partition_int(mesh.n_active_cells());
    GridTools::get_subdomain_association(mesh, partition_int);
    const Vector<double> partitioning(partition_int.begin(), partition_int.end());
    data_out.add_data_vector(partitioning, "partitioning");

    data_out.build_patches();

    const std::string output_file_name = "output-Navier-Stokes" + std::to_string(dim) + "d-";
    data_out.write_vtu_with_pvtu_record("./output/",
                                        output_file_name,
                                        time,
                                        MPI_COMM_WORLD,
                                        numbers::invalid_unsigned_int,
                                        1);

    pcout << "Output written to " << output_file_name << std::endl;
    pcout << "===============================================" << std::endl;
}

template <unsigned int dim>
void NavierStokes<dim>::run()
{
    pcout << "===============================================" << std::endl;

    // Apply the initial condition.
    {
        pcout << "Applying the initial condition" << std::endl;

        VectorTools::interpolate(dof_handler, initial_solution_function, solution_owned);
        solution = solution_owned;

        // Output the initial solution.
        output(0);
        pcout << "===============================================" << std::endl;
    }

    unsigned int time_step = 0;
    double time = 0;

    const unsigned int total_steps = static_cast<unsigned int>(std::round(final_time / time_step_size));

    while (time < final_time - 0.5 * time_step_size)
    {

        time += time_step_size;
        ++time_step;
        inlet_velocity.set_time(time);

        pcout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
              << time << ":" << std::flush;

        if (time_step == 1)
            assemble(time);
        else
            assemble_time_step(time);

        solve_time_step(this.preconditioner);

        if (time_step % 10 == 0)
            output(time_step);
    }
};

// Explicit instantiation
template class NavierStokes<2>;
template class NavierStokes<3>;