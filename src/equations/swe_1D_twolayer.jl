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

struct TwoLayerShallowWaterEquations1D{T, S} <: Equation
    B::S
    ρ1::T
    ρ2::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    function TwoLayerShallowWaterEquations1D(
        B::BottomType = ConstantBottomTopography();
        ρ1 = 1.0,
        ρ2 = 1.02,
        g = 9.81,
        depth_cutoff = 1e-5,
        desingularizing_kappa = 1e-5,
    ) where {BottomType <: AbstractBottomTopography}
        new{typeof(g), typeof(B)}(
            B, ρ1, ρ2, g, depth_cutoff, desingularizing_kappa
        )
    end
end


function Adapt.adapt_structure(to, eq::TwoLayerShallowWaterEquations1D{T, S}) where {T, S}
    B = Adapt.adapt_structure(to, eq.B)
    ρ1 = Adapt.adapt_structure(to, eq.ρ1)
    ρ2 = Adapt.adapt_structure(to, eq.ρ2)
    g = Adapt.adapt_structure(to, eq.g)
    depth_cutoff = Adapt.adapt_structure(to, eq.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, eq.desingularizing_kappa)

    TwoLayerShallowWaterEquations1D(B; ρ1 = ρ1, ρ2 = ρ2, g = g, depth_cutoff = depth_cutoff, desingularizing_kappa = desingularizing_kappa)
end

function (eq::TwoLayerShallowWaterEquations1D)(::XDIRT, h1, q1, h2, q2)
    g  = eq.g
    u1 = desingularize(eq, h1, q1)
    u2 = desingularize(eq, h2, q2)

    return @SVector [
        eq.ρ1 * q1, 
        eq.ρ1 * (q1*u1 + 0.5*g*h1^2),
        eq.ρ2 * q2,                              
        eq.ρ2 * (q2*u2 + 0.5*g*h2^2),
        ]
end


#Helper function to compute eigenvalue bounds using Lagrange method
function lagrange_bounds(c1, c2, c3, c4)
    c = (c1, c2, c3, c4)
    Sc = Float64[]
    Sd = Float64[]
    for j in 1:4
        cj = c[j]
        dj = (-1)^j * c[j]
        if cj < 0
            push!(Sc, abs(cj)^(1/j))
        end
        if dj < 0
            push!(Sd, -abs(dj)^(1/j))
        end
    end
    if length(Sc) == 0
        λmax = 0.0
    elseif length(Sc) == 1
        λmax = Sc[1]
    else
        sort!(Sc, rev=true)
        λmax = Sc[1] + Sc[2]
    end
    if length(Sd) == 0
        λmin = 0.0
    elseif length(Sd) == 1
        λmin = Sd[1]
    else
        sort!(Sd)
        λmin = Sd[1] + Sd[2]
    end
    return λmin, λmax
end


# See Kurganov and Petrova (2009) "Central-Upwind Schemes for Two-Layer Shallow Water Equations" eq. (2.18) - (2.24)
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations1D,::XDIRT, h1, q1, h2, q2)
    g  = eq.g
    ρ1 = eq.ρ1
    ρ2 = eq.ρ2
    r  = ρ1 / ρ2
    H = h1 + h2
    u1 = desingularize(eq, h1, q1)
    u2 = desingularize(eq, h2, q2)

    # Check if we can use the eigenvalues (2.20) from Kurganov and Petrova (2009)
    if (u2 - u1)^2 < (1 - r)*g*H
        Um = (h1*u1 + h2*u2)/H
        Uc = (h1*u2 + h2*u1)/H

        c_ext = sqrt(g*H)
        c_int = sqrt((1 - r)*g*(h1*h2/H) * (1- (u2 - u1)^2/((1 - r)*g*H)))

        return @SVector [Um + c_ext, Um - c_ext, Uc + c_int, Uc - c_int]
    else
        # Use Lagrange method to compute bounds on eigenvalues
        c1 = -2*(u1 + u2)
        c2 = (u1 + u2)^2 + 2*u1*u2 - g*H
        c3 = -2*u1*u2*(u1 + u2) + 2*g*(u1*h2 + u2*h1)
        c4 = u1^2*u2^2 - g*(u1^2*h2 + u2^2*h1) + g^2*(1 - r)*h1*h2

        λmin, λmax = lagrange_bounds(c1, c2, c3, c4)
        return @SVector [λmin, λmax] 
    end
end


function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations1D, ::XDIRT, h1, q1, h2, q2)
    λ = compute_eigenvalues(eq, XDIRT(), h1, q1, h2, q2)
    return maximum(abs, λ)
end

conserved_variable_names(::Type{T}) where {T<:TwoLayerShallowWaterEquations1D} = (:h1, :q1, :h2, :q2)