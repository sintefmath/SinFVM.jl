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
        ρ * hu * hu / h + ρ * g * h^2 / 2,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquationsPure)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv * hv / h + ρ * g * h^2 / 2,
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
