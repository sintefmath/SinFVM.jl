create_volume(backend, grid, equation) = Volume(backend, equation, grid)
create_scalar(backend, grid, equation) = convert_to_backend(backend, zeros(size(grid)))

struct ConservedSystem{BackendType, ReconstructionType,NumericalFluxType,EquationType,GridType,BufferType, ScalarBufferType} <: System
    backend::BackendType
    reconstruction::ReconstructionType
    numericalflux::NumericalFluxType
    equation::EquationType
    grid::GridType

    left_buffer::BufferType
    right_buffer::BufferType
    wavespeeds::ScalarBufferType
    current_wavespeed::MVector{1, Float64}
    function ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
	is_compatible(equation, grid)
        left_buffer = create_volume(backend, grid, equation)
        right_buffer = create_volume(backend, grid, equation)
        wavespeeds = create_scalar(backend, grid, equation)
        return new{
        typeof(backend),
        typeof(reconstruction),
        typeof(numericalflux),
        typeof(equation),
        typeof(grid),
        typeof(left_buffer),
        typeof(wavespeeds)
    }(backend, reconstruction, numericalflux, equation, grid, left_buffer, right_buffer, wavespeeds)
    end
end

create_volume(backend, grid, cs::ConservedSystem) = create_volume(backend, grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state)
    reconstruct!(cs.backend, cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, XDIR)
    return compute_flux!(cs.backend, cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.wavespeeds, cs.grid, cs.equation, XDIR)
end


struct BalanceSystem{ConservedSystemType<:System,SourceTerm} <: System
    conserved_system::ConservedSystemType
    source_term::SourceTerm
end


function add_time_derivative!(output, bs::BalanceSystem, current_state)
    # First add conserved system (so F_{i+1}-F_i)
    wavespeed = add_time_derivative!(output, bs.conserved_system, current_state)

    @assert false
    # FIX FOR LOOP UNDERNEATH
    # Then add source term
    for_each_inner_cell(bs.conserved_system.grid) do ileft, imiddle, iright
        output[imiddle] += bs.source_term(current_state[imiddle])

        nothing
    end

    return wavespeed
end
create_volume(backend, grid, bs::BalanceSystem) = create_volume(backend, grid, bs.conserved_system)

function is_compatible(eq::Burgers, grid::Grid)
    @assert true
end

function is_compatible(eq::ShallowWaterEquations1D, grid::CartesianGrid{1})
    @assert grid.totalcells[1]+1 == size(eq.B)[1]
end
