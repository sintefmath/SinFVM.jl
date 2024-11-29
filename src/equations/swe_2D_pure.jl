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


struct ShallowWaterEquationsPure{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquationsPure() = ShallowWaterEquationsPure(1.0, 9.81)

function (eq::ShallowWaterEquationsPure)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquationsPure)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv * hv / h + 0.5 * ρ * g * h^2,
    ]
end

conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquationsPure} = (:h, :hu, :hv)

function compute_eigenvalues(eq::ShallowWaterEquationsPure, ::XDIRT, h, hu, hv)
    g = eq.g
    u = hu / h
    return @SVector [u + sqrt(g * h), u - sqrt(g * h), u]
end



function compute_eigenvalues(eq::ShallowWaterEquationsPure, ::YDIRT, h, hu, hv)
    g = eq.g
    v = hv / h
    return @SVector [v + sqrt(g * h), v - sqrt(g * h), v]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquationsPure, direction, h, hu, hv)
    return maximum(compute_eigenvalues(eq, direction, h, hu, hv))
end
