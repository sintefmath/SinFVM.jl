using StaticArrays
using Adapt
using SinFVM

# ============================================================
# Two-layer SWE in 2D (w-storage)
# Conserved variables:
#   U = (h1, q1, p1, w, q2, p2)
# where:
#   q_i = h_i u_i,  p_i = h_i v_i
#   w = h2 + B   (equilibrium/storage variable)
# Physical h2 at faces: h2 = w - Bface
# ============================================================

struct TwoLayerShallowWaterEquations2D{T, S} <: Equation
    B::S
    ρ1::T
    ρ2::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    eigenvalue_method::Symbol
    hyperbolicity_correction::Bool
    function TwoLayerShallowWaterEquations2D(
        B::BottomType = ConstantBottomTopography();
        ρ1 = 1.0,
        ρ2 = 1.02,
        g = 9.81,
        depth_cutoff = 1e-5,
        desingularizing_kappa = 1e-5,
        eigenvalue_method = :old, #:old or :new
        hyperbolicity_correction = true
    ) where {BottomType <: AbstractBottomTopography}
        new{typeof(g), typeof(B)}(B, ρ1, ρ2, g, depth_cutoff, desingularizing_kappa, eigenvalue_method, hyperbolicity_correction)
    end
end

function Adapt.adapt_structure(to, eq::TwoLayerShallowWaterEquations2D{T,S}) where {T,S}
    B  = Adapt.adapt_structure(to, eq.B)
    ρ1 = Adapt.adapt_structure(to, eq.ρ1)
    ρ2 = Adapt.adapt_structure(to, eq.ρ2)
    g  = Adapt.adapt_structure(to, eq.g)
    depth_cutoff = Adapt.adapt_structure(to, eq.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, eq.desingularizing_kappa)
    eigenvalue_method = Adapt.adapt_structure(to, eq.eigenvalue_method)
    hyperbolicity_correction = Adapt.adapt_structure(to, eq.hyperbolicity_correction)
    TwoLayerShallowWaterEquations2D(B; ρ1=ρ1, ρ2=ρ2, g=g,
        depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa, eigenvalue_method=eigenvalue_method, hyperbolicity_correction=hyperbolicity_correction)
end


# F(U,B)
function (eq::TwoLayerShallowWaterEquations2D)(::XDIRT, h1, q1, p1, w, q2, p2, Bface)
    g = eq.g
    r = eq.ρ1 / eq.ρ2
    h2 = w - Bface

    u1 = desingularize(eq, h1, q1); v1 = desingularize(eq, h1, p1)
    u2 = desingularize(eq, h2, q2); v2 = desingularize(eq, h2, p2)

    return @SVector[
        # layer 1
        q1,
        q1*u1 + g*h1*(h1 + w),
        q1*v1,

        # layer 2
        q2,
        q2*u2 + 0.5*g*w^2 - 0.5*g*r*h1^2 - g*Bface*(r*h1 + w),
        q2*v2
    ]
end

# ----------------------------
# Flux in y-direction: G(U,B)
# ----------------------------
function (eq::TwoLayerShallowWaterEquations2D)(::YDIRT, h1, q1, p1, w, q2, p2, Bface)
    g = eq.g
    r = eq.ρ1 / eq.ρ2
    h2 = w - Bface

    v1 = desingularize(eq, h1, p1); v2 = desingularize(eq, h2, p2)

    return @SVector[
        # layer 1
        p1,
        q1*v1,
        p1*v1 + g*h1*(h1 + w),

        # layer 2
        p2,
        q2*v2,
        p2*v2 + 0.5*g*w^2 - 0.5*g*r*h1^2 - g*Bface*(r*h1 + w)
    ]
end

# ============================================================
# Eigenvalues:
# Use the 1D directional eigenvalue approximations in each coordinate
# direction. The CentralUpwind flux passes the momentum in the active direction, so m1 and m2 denote the directional momenta for layers 1 and 2.
# ============================================================

# Old directional eigenvalue approximation
# See Kurganov and Petrova (2009), eqs. (2.18)–(2.24)
function compute_eigenvalues_old(eq::TwoLayerShallowWaterEquations2D, direction::Direction, h1, m1, h2, m2)
    g  = eq.g
    r  = eq.ρ1 / eq.ρ2
    H  = h1 + h2

    u1 = desingularize(eq, h1, m1)
    u2 = desingularize(eq, h2, m2)

    if (u2 - u1)^2 < (1 - r) * g * H
        Um = (h1 * u1 + h2 * u2) / H
        Uc = (h1 * u2 + h2 * u1) / H

        c_ext = sqrt(g * H)
        c_int = sqrt((1 - r) * g * (h1 * h2 / H) *
                     (1 - (u2 - u1)^2 / ((1 - r) * g * H)))

        return @SVector [Um + c_ext, Um - c_ext, Uc + c_int, Uc - c_int]
    else
        c1 = -2 * (u1 + u2)
        c2 = (u1 + u2)^2 + 2 * u1 * u2 - g * H
        c3 = -2 * u1 * u2 * (u1 + u2) + 2 * g * (u1 * h2 + u2 * h1)
        c4 = u1^2 * u2^2 - g * (u1^2 * h2 + u2^2 * h1) + g^2 * (1 - r) * h1 * h2

        λmin, λmax = lagrange_bounds(c1, c2, c3, c4)
        return @SVector [λmax, λmin, λmax, λmin]
    end
end


# New directional eigenvalue approximation
function compute_eigenvalues_new(eq::TwoLayerShallowWaterEquations2D, direction::Direction, h1, m1, h2, m2)
    T = promote_type(typeof(h1), typeof(m1), typeof(h2), typeof(m2), typeof(eq.g))
    g = T(eq.g)
    r = T(eq.ρ1 / eq.ρ2)
    u1 = desingularize(eq, h1, m1)
    u2 = desingularize(eq, h2, m2)

    H = h1 + h2
    if H <= zero(T)
        return @SVector [zero(T), zero(T), zero(T), zero(T)]
    end

    Δu = u1 - u2

    γc_sq = (g * H / 2)^2 + (r - one(T)) * g^2 * h1 * h2 + (2 * g * h1 * h2 / H) * Δu^2
    γc = sqrt(max(zero(T), γc_sq))

    base = g * H / 2 + (h1 * h2 / H^2) * Δu^2

    c_ext = sqrt(max(zero(T), base + γc))
    c_int = sqrt(max(zero(T), base - γc))

    Uext = (u1 * h1 + u2 * h2) / H
    Uint = (u1 * h2 + u2 * h1) / H

    return @SVector [Uext + c_ext, Uext - c_ext, Uint + c_int, Uint - c_int]
end


function compute_eigenvalues(eq::TwoLayerShallowWaterEquations2D,
                             direction::Direction, h1, m1, h2, m2)
    if eq.eigenvalue_method == :old
        return compute_eigenvalues_old(eq, direction, h1, m1, h2, m2)
    elseif eq.eigenvalue_method == :new
        return compute_eigenvalues_new(eq, direction, h1, m1, h2, m2)
    else
        error("Unknown eigenvalue method $(eq.eigenvalue_method). Use :old or :new.")
    end
end


function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations2D,
                                    direction::Direction, h1, m1, h2, m2)
    λ = compute_eigenvalues(eq, direction, h1, m1, h2, m2)
    return maximum(abs, λ)
end

# Hyperbolicity bounds for 2D
# Reuse the same formulas as in 1D, since they depend only on h1, h2, ρ1/ρ2, g
#The hyperbolicity condition should be valid in all possible directions in \mathbb{R}^2, so we use the shear^2 =  (u_1-u_2)^2 + (v_1-v_2)^2 in the bounds computation.
function enforce_hyperbolicity!(backend, U, grid::Grid, eq::TwoLayerShallowWaterEquations2D, dt)
    ρ1 = eq.ρ1
    ρ2 = eq.ρ2
    r  = ρ1 / ρ2
    @fvmloop for_each_cell(backend, grid) do imiddle
        V  = U[imiddle]; h1 = V[1]; q1 = V[2]; p1 = V[3]; w  = V[4]; q2 = V[5]; p2 = V[6]

        B  = B_cell(eq.B, imiddle)
        h2 = w - B

        if h1 > eq.depth_cutoff && h2 > eq.depth_cutoff
            u1 = desingularize(eq, h1, q1); v1 = desingularize(eq, h1, p1)
            u2 = desingularize(eq, h2, q2); v2 = desingularize(eq, h2, p2)

            Δu = u1 - u2
            Δv = v1 - v2
            shear = sqrt(Δu^2 + Δv^2)

            active = false
            FL = 0.0
            if eq.eigenvalue_method == :old
                FL, FR = hyperbolicity_bounds_old(eq, h1, h2)
                active = shear > FL
            elseif eq.eigenvalue_method == :new
                FL, FR = hyperbolicity_bounds_new(eq, h1, h2)
                active = FL < shear < FR
            else
                error("Unknown eigenvalue method $(eq.eigenvalue_method). Use :old or :new.")
            end

            if active
                ctilde = (h1 * h2) / (dt * (h2 + r * h1)) * max(shear / FL - 1, 0.0)

                # Semi-implicit update of the full relative velocity vector
                denom  = 1 + dt * ctilde * (1 / h1 + r / h2)
                Δu_new = Δu / denom
                Δv_new = Δv / denom

                u1_new = u1 - dt * (ctilde / h1) * Δu_new
                v1_new = v1 - dt * (ctilde / h1) * Δv_new
                u2_new = u2 + dt * (r * ctilde / h2) * Δu_new
                v2_new = v2 + dt * (r * ctilde / h2) * Δv_new

                U[imiddle] = typeof(V)(h1, h1 * u1_new, h1 * v1_new, w, h2 * u2_new, h2 * v2_new)
            end
        end
    end

    return nothing
end

conserved_variable_names(::Type{T}) where {T<:TwoLayerShallowWaterEquations2D} = (:h1, :q1, :p1, :w, :q2, :p2)
