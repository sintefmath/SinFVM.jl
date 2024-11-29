# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

create_volume(backend, grid, equation) = Volume(backend, equation, grid)
create_scalar(backend, grid, equation) = convert_to_backend(backend, zeros(backend.realtype, size(grid)))

struct ConservedSystem{BackendType,ReconstructionType,NumericalFluxType,EquationType,GridType,BufferType,ScalarBufferType,ImplicitSourceTermType} <: System
    backend::BackendType
    reconstruction::ReconstructionType
    numericalflux::NumericalFluxType
    equation::EquationType
    grid::GridType

    left_buffer::BufferType
    right_buffer::BufferType
    wavespeeds::ScalarBufferType

    source_terms::Vector{SourceTerm}
    implicit_source_term::ImplicitSourceTermType
    function ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms=SourceTerm[], implicit_source_term=nothing)
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
            typeof(wavespeeds),
            typeof(implicit_source_term)
        }(backend, reconstruction, numericalflux, equation, grid, left_buffer, right_buffer, wavespeeds, source_terms, implicit_source_term)
    end
end

function ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_term::SourceTerm)
    return ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [source_term])
end

create_volume(backend, grid, cs::ConservedSystem) = create_volume(backend, grid, cs.equation)

function add_time_derivative!(output, cs::ConservedSystem, current_state, t)
    maximum_wavespeed = zeros(cs.backend.realtype, dimension(cs.grid))
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


implicit_substep!(output, previous_state, system, backend, implicit_source_term::Nothing, equation, dt) = nothing
function implicit_substep!(output, previous_state, system::ConservedSystem, dt)
    implicit_substep!(output, previous_state, system, system.backend, system.implicit_source_term, system.equation, dt)
    return nothing
end
