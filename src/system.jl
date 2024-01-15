
create_buffer(grid::CartesianGrid{1}, equation::Equation) = zeros(SVector{number_of_conserved_variables(equation),Float64}, grid.totalcells[1])

abstract type System end

struct ConservedSystem{ReconstructionType,NumericalFluxType,EquationType,GridType,BufferType} <: System
    reconstruction::ReconstructionType
    numericalflux::NumericalFluxType
    equation::EquationType
    grid::GridType

    left_buffer::BufferType
    right_buffer::BufferType

    ConservedSystem(reconstruction, numericalflux, equation, grid) = new{
        typeof(reconstruction),
        typeof(numericalflux),
        typeof(equation),
        typeof(grid),
        typeof(create_buffer(grid, equation))
    }(reconstruction, numericalflux, equation, grid, create_buffer(grid, equation), create_buffer(grid, equation))
end

create_buffer(grid, cs::ConservedSystem) = create_buffer(grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state)
    reconstruct!(cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, XDIR)
    compute_flux!(cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.grid, cs.equation, XDIR)
end


struct BalanceSystem{ConservedSystemType<:System,SourceTerm} <: System
    conserved_system::ConservedSystemType
    source_term::SourceTerm
end


function add_time_derivative!(output, bs::BalanceSystem, current_state)
    # First add conserved system (so F_{i+1}-F_i)
    add_time_derivative!(output, bs.conserved_system, current_state)

    # Then add source term
    for_each_inner_cell(bs.conserved_system.grid) do ileft, imiddle, iright
        output[imiddle] += bs.source_term(current_state[imiddle])

        nothing
    end
end
create_buffer(grid, bs::BalanceSystem) = create_buffer(grid, bs.conserved_system)




function compute_wavespeed(system::ConservedSystem, grid, state)
    RealType = typeof(compute_max_abs_eigenvalue(system.equation, XDIR, first(state)))
    maximum_eigenvalue::RealType = nextfloat(typemin(RealType))
    for u in state
        for direction in directions(grid)
            eigenvalue_in_direction = compute_max_abs_eigenvalue(system.equation, direction, u)
            maximum_eigenvalue = max(eigenvalue_in_direction, maximum_eigenvalue)
        end        
    end
    return maximum_eigenvalue
end

function compute_wavespeed(system::BalanceSystem, grid, state)
    compute_wavespeed(system.conserved_system, grid, state)
end