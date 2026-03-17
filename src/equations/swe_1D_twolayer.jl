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
        ρ1 = 0.98,
        ρ2 = 1.00,
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

function (eq::TwoLayerShallowWaterEquations1D)(::XDIRT, h1, q1, w, q2, Bface)
    g  = eq.g
    r  = eq.ρ1 / eq.ρ2
    h2 = w - Bface
    u1 = desingularize(eq, h1, q1)
    u2 = desingularize(eq, h2, q2)
    

    return @SVector [
        q1,
        (q1*u1 + g*(h1 + w)*h1),
        q2,
        (q2*u2 + 0.5*g*w^2 - 0.5*g*r*h1^2 - g*Bface*(r*h1 + w)),
    ]
end


# Helper function to compute eigenvalue bounds using Lagrange method
function lagrange_bounds(c1, c2, c3, c4)
    T = promote_type(typeof(c1), typeof(c2), typeof(c3), typeof(c4))
    c = (c1, c2, c3, c4)

    Sc = T[]
    Sd = T[]

    for j in 1:4
        cj = c[j]
        dj = isodd(j) ? -cj : cj   # dj = (-1)^j * cj

        if cj < 0
            push!(Sc, abs(cj)^(one(T)/j))
        end
        if dj < 0
            push!(Sd, -abs(dj)^(one(T)/j))
        end
    end

    λmax = isempty(Sc) ? zero(T) :
           length(Sc) == 1 ? Sc[1] :
           (sort!(Sc, rev=true); Sc[1] + Sc[2])

    λmin = isempty(Sd) ? zero(T) :
           length(Sd) == 1 ? Sd[1] :
           (sort!(Sd); Sd[1] + Sd[2])

    return λmin, λmax
end

# See Kurganov and Petrova (2009) "Central-Upwind Schemes for Two-Layer Shallow Water Equations" eq. (2.18) - (2.24)
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations1D, direction::Direction, h1, q1, h2, q2)
    g  = eq.g
    ρ1 = eq.ρ1
    ρ2 = eq.ρ2
    r  = ρ1 / ρ2
    H = h1 + h2

    # In 1D this is just q/h, but direction is now available for 2D reuse
    u1 = desingularize(eq, h1, q1)
    u2 = desingularize(eq, h2, q2)

    # Kurganov & Petrova eigenvalues
    if (u2 - u1)^2 < (1 - r)*g*H
        Um = (h1*u1 + h2*u2)/H
        Uc = (h1*u2 + h2*u1)/H

        c_ext = sqrt(g*H)
        c_int = sqrt((1 - r)*g*(h1*h2/H) *
                     (1 - (u2 - u1)^2/((1 - r)*g*H)))

        return @SVector [Um + c_ext, Um - c_ext, Uc + c_int, Uc - c_int]
    else
        c1 = -2*(u1 + u2)
        c2 = (u1 + u2)^2 + 2*u1*u2 - g*H
        c3 = -2*u1*u2*(u1 + u2) + 2*g*(u1*h2 + u2*h1)
        c4 = u1^2*u2^2 - g*(u1^2*h2 + u2^2*h1) + g^2*(1 - r)*h1*h2

        λmin, λmax = lagrange_bounds(c1, c2, c3, c4)
        return @SVector [λmax, λmin, λmax, λmin]
    end
end


function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations1D, direction::Direction, h1, q1, h2, q2)
    λ = compute_eigenvalues(eq, direction, h1, q1, h2, q2)
    return maximum(abs, λ)
end


conserved_variable_names(::Type{T}) where {T<:TwoLayerShallowWaterEquations1D} = (:h1, :q1, :w, :q2)


# Enforce hyperbolicity by adding friction source term if needed for all two-layer equations, following Section 4 in Castro et al. (2010) "Numerical Treatment of the Loss of Hyperbolicity of the Two-Layer Shallow-Water System"
#
function enforce_hyperbolicity!(backend, U, grid::Grid, eq::TwoLayerShallowWaterEquations1D, dt)
    ρ1 = eq.ρ1; ρ2 = eq.ρ2; r  = ρ1 / ρ2
    g  = eq.g; gp = g * (ρ2 - ρ1) / ρ2
    @fvmloop for_each_cell(backend, grid) do imiddle
        V = U[imiddle]; h1 = V[1]; q1 = V[2]; w  = V[3]; q2 = V[4]
        B  = B_cell(eq.B, imiddle)
        h2 = w - B
        if h1 > eq.depth_cutoff && h2 > eq.depth_cutoff
            u1m = desingularize(eq, h1, q1)
            u2m = desingularize(eq, h2, q2)

            crit   = gp * (h1 + h2)
            shear  = abs(u1m - u2m)

            if shear^2 > crit
                # Castro et al. practical coefficient
                c = inv(dt) * max(shear / sqrt(crit) - 1, 0)
                # Scaled coefficient used in the momentum correction
                ctilde = c * h1 * h2 / (h2 + r * h1)

                #Update velocities with friction correction
                denom = 1 + dt * ctilde * (1 / h1 + r / h2)
                Δu    = (u1m - u2m) / denom
                u1 = u1m - dt * ctilde / h1 * Δu
                u2 = u2m + dt * r * ctilde / h2 * Δu

                U[imiddle] = typeof(V)(h1, h1 * u1, w, h2 * u2)
            end
        end
    end

    return nothing
end

##############################################################################
######## Trying new eigenvalue computation and friction correction ###########
""" See: M.J. Castro Díaz et al. “Discussion on different numerical treatments on the loss of
hyperbolicity for the two-layer shallow water system”. In: Advances in Water Re-
sources 182 (2023), p. 104587. issn: 0309-1708. url: https://www.sciencedirect.
com/science/article/pii/S030917082300221X."""
# -----------------------------------------------------------------------------
# New approximate eigenvalues from Theorem 1, using conserved variables
# -----------------------------------------------------------------------------

function compute_eigenvalues(eq::TwoLayerShallowWaterEquations1D,
                             direction::Direction, h1, q1, h2, q2)
    T = promote_type(typeof(h1), typeof(q1), typeof(h2), typeof(q2), typeof(eq.g))
    g = T(eq.g); r = T(eq.ρ1 / eq.ρ2)
    u1 = desingularize(eq, h1, q1)
    u2 = desingularize(eq, h2, q2)
    H = h1 + h2
    if H <= zero(T)
        return @SVector zeros(T, 4)
    end

    Δu = u1 - u2
    # Eq. (9d)
    γc_sq = (g * H / 2)^2 + (r - one(T)) * g^2 * h1 * h2 + (2 * g * h1 * h2 / H) * Δu^2
    γc = sqrt(max(zero(T), γc_sq))

    # Eqs. (9b)-(9c)
    base = g * H / 2 + (h1 * h2 / H^2) * Δu^2
    c_ext_sq = base + γc
    c_int_sq = base - γc
    c_ext = sqrt(max(zero(T), c_ext_sq))
    c_int = sqrt(max(zero(T), c_int_sq))

    # Eq. (9a)
    Uext = (u1 * h1 + u2 * h2) / H
    Uint = (u1 * h2 + u2 * h1) / H

    return @SVector [Uext + c_ext, Uext - c_ext, Uint + c_int, Uint - c_int]
end


function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations1D,
                                    direction::Direction, h1, q1, h2, q2)
    λ = compute_eigenvalues(eq, direction, h1, q1, h2, q2)
    return maximum(abs, λ)
end


# -----------------------------------------------------------------------------
# New hyperbolicity bounds from eq. (16), using depths only
# -----------------------------------------------------------------------------

@inline function enforce_hyperbolicity!(eq::TwoLayerShallowWaterEquations1D, h1, h2)
    T = promote_type(typeof(h1), typeof(h2), typeof(eq.g))
    g = T(eq.g)
    r = T(eq.ρ1 / eq.ρ2)

    H = h1 + h2
    if h1 <= zero(T) || h2 <= zero(T) || H <= zero(T)
        return zero(T), zero(T)
    end

    disc = 1 - 4 * (1 - r) * (h1 * h2) / H^2
    disc = max(zero(T), disc)
    sdisc = sqrt(disc)

    pref = g * H^3 / (2 * h1 * h2)

    # lower boundary from eq. (16b)
    FL_sq = pref * (1 - sdisc)

    # right boundary from eq. (16c)
    FR_sq = 2 * g * H * (1 + sqrt(max(zero(T), r)))

    FL = sqrt(max(zero(T), FL_sq))
    FR = sqrt(max(zero(T), FR_sq))

    return FL, FR
end


# -----------------------------------------------------------------------------
# New friction treatment (NFT), using conserved variables only
# -----------------------------------------------------------------------------

function enforce_hyperbolicity!(backend, U, grid::Grid, eq::TwoLayerShallowWaterEquations1D, dt)
    ρ1 = eq.ρ1; ρ2 = eq.ρ2; r = ρ1 / ρ2
    g  = eq.g
    @fvmloop for_each_cell(backend, grid) do imiddle
        V = U[imiddle]; h1 = V[1]; q1 = V[2]; w = V[3]; q2 = V[4]
        B = B_cell(eq.B, imiddle)
        h2 = w - B

        if h1 > eq.depth_cutoff && h2 > eq.depth_cutoff
            u1m = desingularize(eq, h1, q1)
            u2m = desingularize(eq, h2, q2)

            shear = abs(u1m - u2m)
            H = h1 + h2

            disc  = max(0.0, 1 - 4 * (1 - r) * (h1 * h2) / H^2)
            FL    = sqrt(max(0.0, g * H^3 / (2 * h1 * h2) * (1 - sqrt(disc))))
            FR    = sqrt(max(0.0, 2 * g * H * (1 + sqrt(r))))
            if FL < shear < FR
                ctilde = (h1 * h2) / (dt * (h2 + r * h1)) * max(shear / FL - 1, 0.0)

                #Same correction idea with new values
                denom = 1 + dt * ctilde * (1 / h1 + r / h2)
                Δu    = (u1m - u2m) / denom
                u1    = u1m - dt * ctilde / h1 * Δu
                u2    = u2m + dt * r * ctilde / h2 * Δu

                U[imiddle] = typeof(V)(h1, h1 * u1, w, h2 * u2)
            end
        end
    end

    return nothing
end