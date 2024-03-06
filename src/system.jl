create_buffer(backend, grid::CartesianGrid{1}, equation::Equation) = zeros(SVector{number_of_conserved_variables(equation),Float64}, grid.totalcells[1])
create_buffer(backend::CPUBackend, grid::CartesianGrid{1}, equation::Equation) = KernelAbstractions.zeros(backend.backend, SVector{number_of_conserved_variables(equation),Float64}, grid.totalcells[1])
create_buffer(backend::CUDABackend, grid::CartesianGrid{1}, equation::Equation) = CUDA.cu(zeros(SVector{number_of_conserved_variables(equation),Float64}, grid.totalcells[1]))

abstract type System end

struct ConservedSystem{BackendType, ReconstructionType,NumericalFluxType,EquationType,GridType,BufferType} <: System
    backend::BackendType
    reconstruction::ReconstructionType
    numericalflux::NumericalFluxType
    equation::EquationType
    grid::GridType

    left_buffer::BufferType
    right_buffer::BufferType

    ConservedSystem(backend, reconstruction, numericalflux, equation, grid) = new{
        typeof(backend),
        typeof(reconstruction),
        typeof(numericalflux),
        typeof(equation),
        typeof(grid),
        typeof(create_buffer(backend, grid, equation))
    }(backend, reconstruction, numericalflux, equation, grid, create_buffer(backend, grid, equation), create_buffer(backend, grid, equation))
end

create_buffer(backend, grid, cs::ConservedSystem) = create_buffer(backend, grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state)
    reconstruct!(cs.backend, cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, XDIR)
    compute_flux!(cs.backend, cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.grid, cs.equation, XDIR)
end


struct BalanceSystem{ConservedSystemType<:System,SourceTerm} <: System
    conserved_system::ConservedSystemType
    source_term::SourceTerm
end


function add_time_derivative!(output, bs::BalanceSystem, current_state)
    # First add conserved system (so F_{i+1}-F_i)
    add_time_derivative!(output, bs.conserved_system, current_state)

    @assert false
    # FIX FOR LOOP UNDERNEATH
    # Then add source term
    for_each_inner_cell(bs.conserved_system.grid) do ileft, imiddle, iright
        output[imiddle] += bs.source_term(current_state[imiddle])

        nothing
    end
end
create_buffer(backend, grid, bs::BalanceSystem) = create_buffer(backend, grid, bs.conserved_system)




function compute_wavespeed(system::ConservedSystem, grid, state)
    # TODO: Remove @allowscalar here
    CUDA.@allowscalar RealType = typeof(compute_max_abs_eigenvalue(system.equation, XDIR, first(state)))
    maximum_eigenvalue::RealType = nextfloat(typemin(RealType))

    for direction in directions(grid)

        eigenvalue_in_direction = let eq = system.equation, dir = direction
            u -> compute_max_abs_eigenvalue(eq, dir, u)
        end

        maximum_at_direction = maximum(eigenvalue_in_direction.(state))
        maximum_eigenvalue = max(maximum_at_direction, maximum_eigenvalue)
    end
    return maximum_eigenvalue
end

function compute_wavespeed(system::BalanceSystem, grid, state)
    compute_wavespeed(system.conserved_system, grid, state)
end