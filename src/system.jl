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

    source_terms::Vector{SourceTerm}
    function ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms=SourceTerm[])
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
    }(backend, reconstruction, numericalflux, equation, grid, left_buffer, right_buffer, wavespeeds, source_terms)
    end
end

function ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_term::SourceTerm) 
    return ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [source_term])
end

create_volume(backend, grid, cs::ConservedSystem) = create_volume(backend, grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state)
    reconstruct!(cs.backend, cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, XDIR)
    wavespeed = compute_flux!(cs.backend, cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.wavespeeds, cs.grid, cs.equation, XDIR)

    for source_term in cs.source_terms
        evaluate_source_term!(source_term, output, current_state, cs)
    end
    return wavespeed
end

function is_compatible(eq::Equation, grid::Grid)
    nothing
end

function is_compatible(eq::ShallowWaterEquations1D, grid::CartesianGrid{1})
    if grid.totalcells[1]+1 != size(eq.B)[1]
        throw(DimensionMismatch("Equation and grid has mismatching dimensions. "*
                                "\nGrid requires B to have dimension $(grid.totalcells .+ 1), "*
                                "but got B with dimension $(size(eq.B))"))
    end
end
