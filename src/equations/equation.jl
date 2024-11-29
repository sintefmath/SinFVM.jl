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

number_of_conserved_variables(::Type{T}) where {T} = error("This is not an equation type.")
number_of_conserved_variables(::T) where {T<:Equation} = number_of_conserved_variables(T)
number_of_conserved_variables(::Type{T}) where {T<:Equation} = length(conserved_variable_names(T))


include("burgers.jl")
include("swe_1D_pure.jl")
include("swe_1D.jl")
include("swe_2D_pure.jl")
include("swe_2D.jl")

AllSWE = Union{ShallowWaterEquations1D,ShallowWaterEquations1DPure,ShallowWaterEquationsPure, ShallowWaterEquations}
AllPracticalSWE = Union{ShallowWaterEquations1D, ShallowWaterEquations}
AllPureSWE = Union{ShallowWaterEquations1DPure,ShallowWaterEquationsPure}
AllSWE1D = Union{ShallowWaterEquations1D, ShallowWaterEquations1DPure}
AllSWE2D = Union{ShallowWaterEquations, ShallowWaterEquationsPure}

desingularize(::AllPureSWE, h) = h # TODO: Do we want to something more here?

function desingularize(eq::AllPracticalSWE, h)
    # The different desingularizations are taken from 
    # Brodtkorb and Holm (2021), Coastal ocean forecasting on the GPU using a two-dimensional finite-volume scheme.  
    # Tellus A: Dynamic Meteorology and Oceanography,  73(1), p.1876341.DOI: https://doi.org/10.1080/16000870.2021.1876341
    # and the equation numbers refere to that paper

    # Eq (23):
    # h_star = (sqrt(h^4 + max(h^4, eq.desingularizing_kappa^4)))/(sqrt(2)*h)

    # Eq (24):
    # h_star = (h^2 + eq.desingularizing_kappa^2)/h

    # Eq (25):
    # h_star = (h^2 + max(h^2, eq.desingularizing_kappa^2))/(2*h)

    # Eq (26):
    h_star = copysign(1, h)*max(abs(h), min(h^2/(2*eq.desingularizing_kappa) + eq.desingularizing_kappa/2.0, eq.desingularizing_kappa))
    # h_star = sign(h)*max(abs(h), min(h^2/(2*eq.desingularizing_kappa) + eq.desingularizing_kappa/2.0, eq.desingularizing_kappa))
    # if h < 0.0
    #     h_star = 0.5*eq.desingularizing_kappa
    # end
    return h_star
end

function desingularize(eq, h, momentum)
    h_star = desingularize(eq, h)
    return momentum/h_star
end


function is_compatible(eq::Equation, source_terms::Vector)
    nothing
end

function is_compatible(eq::AllPracticalSWE, source_terms::Vector)
    if is_zero(eq.B) 
        return nothing 
    end
    for source_term in source_terms
        if typeof(source_term) == SourceTermBottom
            return nothing
        end
    end
    throw(ArgumentError("Found non-zero bottom topography in equation, but no corresponding source term. Did you forget to add a source term to your system?"))
end

B_cell(::AllPureSWE, index) = 0.0
B_cell(eq::AllPracticalSWE, index) = B_cell(eq.B, index)
