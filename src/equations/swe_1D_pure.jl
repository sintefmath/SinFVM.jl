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

struct ShallowWaterEquations1DPure{T} <: Equation
    ρ::T
    g::T
    ShallowWaterEquations1DPure(ρ=1.0, g=9.81) = new{typeof(g)}(ρ, g)
end
Adapt.@adapt_structure ShallowWaterEquations1DPure

function (eq::ShallowWaterEquations1DPure)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
    ]
end

function compute_eigenvalues(eq::ShallowWaterEquations1DPure, ::XDIRT, h, hu)
    g = eq.g
    u = hu/h
    return @SVector [u + sqrt(g * h), u - sqrt(g * h)]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1DPure, ::XDIRT, h, hu)
    # TODO: Use compute_eigenvalues
    g = eq.g
    u = hu / h
    return max(abs(u + sqrt(g * h)), abs(u - sqrt(g * h)))
end
conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations1DPure} = (:h, :hu)
