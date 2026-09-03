# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


struct ShallowWaterEquations{T, S} <: Equation
    B::S
    ρ::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    ShallowWaterEquations(B::BottomType=ConstantBottomTopography(); ρ=1.0, g=9.81, depth_cutoff=10^-5, desingularizing_kappa=10^-5) where {BottomType <: AbstractBottomTopography} = new{typeof(g), typeof(B)}(B, ρ, g, depth_cutoff, desingularizing_kappa)
end
function Adapt.adapt_structure(
    to,
    swe::ShallowWaterEquations{T, S}
) where {T, S}
    B = Adapt.adapt_structure(to, swe.B)
    ρ = Adapt.adapt_structure(to, swe.ρ)
    g = Adapt.adapt_structure(to, swe.g)
    depth_cutoff = Adapt.adapt_structure(to, swe.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, swe.desingularizing_kappa)
    
    ShallowWaterEquations(B; ρ=ρ, g=g, depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa)
end

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    v = desingularize(eq, h, hv)
    return @SVector [
        ρ * h * u,
        ρ * h * u * u + ρ * g * h^2 / 2,
        ρ * h * u * v 
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    v = desingularize(eq, h, hv)
    return @SVector [
        ρ * h * v,
        ρ * h * u * v,
        ρ * h * v * v + ρ * g * h^2 / 2,
    ]
end

conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations} = (:h, :hu, :hv)

function compute_eigenvalues(eq::ShallowWaterEquations, ::XDIRT, h, hu, hv)
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [u + sqrt(g * h), u - sqrt(g * h), u]
end



function compute_eigenvalues(eq::ShallowWaterEquations, ::YDIRT, h, hu, hv)
    g = eq.g
    v = desingularize(eq, h, hv)
    return @SVector [v + sqrt(g * h), v - sqrt(g * h), v]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations, direction, h, hu, hv)
    return maximum(abs.(compute_eigenvalues(eq, direction, h, hu, hv)))
end
