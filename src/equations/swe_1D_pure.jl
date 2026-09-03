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
        ρ * hu * hu / h + ρ * g * h^2 / 2,
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
