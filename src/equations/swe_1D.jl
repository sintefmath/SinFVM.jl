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

struct ShallowWaterEquations1D{T, S} <: Equation
    B::S
    ρ::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    ShallowWaterEquations1D(B::BottomType=ConstantBottomTopography(); ρ=1.0, g=9.81, depth_cutoff=10^-5, desingularizing_kappa=10^-5) where {BottomType <: AbstractBottomTopography} = new{typeof(g), typeof(B)}(B, ρ, g, depth_cutoff, desingularizing_kappa)
end

function Adapt.adapt_structure(
    to,
    swe::ShallowWaterEquations1D{T, S}
) where {T, S}
    B = Adapt.adapt_structure(to, swe.B)
    ρ = Adapt.adapt_structure(to, swe.ρ)
    g = Adapt.adapt_structure(to, swe.g)
    depth_cutoff = Adapt.adapt_structure(to, swe.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, swe.desingularizing_kappa)
    
    ShallowWaterEquations1D(B; ρ=ρ, g=g, depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa)
end


function (eq::ShallowWaterEquations1D)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [
        ρ * h * u,
        ρ * h * u * u + 0.5 * ρ * g * h^2,
    ]
end

function compute_eigenvalues(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [u + sqrt(g * h), u - sqrt(g * h)]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    # TODO: Use compute_eigenvalues
    g = eq.g
    u = desingularize(eq, h, hu)
    return max(abs(u + sqrt(g * h)), abs(u - sqrt(g * h)))
end
conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations1D} = (:h, :hu)
