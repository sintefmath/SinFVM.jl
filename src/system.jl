create_volume(backend, grid, equation) = Volume(backend, equation, grid)
create_scalar(backend, grid, equation) = convert_to_backend(backend, zeros(size(grid)))

struct ConservedSystem{BackendType,ReconstructionType,NumericalFluxType,EquationType,GridType,BufferType,ScalarBufferType} <: System
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
        is_compatible(equation, source_terms)
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

function add_time_derivative!(output, cs::ConservedSystem, current_state, t)
    maximum_wavespeed = zeros(dimension(cs.grid))
    for direction in directions(cs.grid)
        reconstruct!(cs.backend, cs.reconstruction, cs.left_buffer, cs.right_buffer, current_state, cs.grid, cs.equation, direction)
        maximum_wavespeed[direction] = compute_flux!(cs.backend, cs.numericalflux, output, cs.left_buffer, cs.right_buffer, cs.wavespeeds, cs.grid, cs.equation, direction)
        for source_term in cs.source_terms
            evaluate_directional_source_term!(source_term, output, current_state, cs, direction)
        end
    end
    for source_term in cs.source_terms
        evaluate_source_term!(source_term, output, current_state, cs, t)
    end
    return maximum_wavespeed
end

function is_compatible(eq::Equation, grid::Grid)
    nothing
end

function is_compatible(eq::AllPracticalSWE, grid::CartesianGrid)
    validate(eq.B, grid)
end
