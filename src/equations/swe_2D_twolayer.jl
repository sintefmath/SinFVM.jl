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
    function TwoLayerShallowWaterEquations2D(
        B::BottomType = ConstantBottomTopography();
        ρ1 = 1.0,
        ρ2 = 1.02,
        g = 9.81,
        depth_cutoff = 1e-5,
        desingularizing_kappa = 1e-5,
    ) where {BottomType <: AbstractBottomTopography}
        new{typeof(g), typeof(B)}(B, ρ1, ρ2, g, depth_cutoff, desingularizing_kappa)
    end
end

function Adapt.adapt_structure(to, eq::TwoLayerShallowWaterEquations2D{T,S}) where {T,S}
    B  = Adapt.adapt_structure(to, eq.B)
    ρ1 = Adapt.adapt_structure(to, eq.ρ1)
    ρ2 = Adapt.adapt_structure(to, eq.ρ2)
    g  = Adapt.adapt_structure(to, eq.g)
    depth_cutoff = Adapt.adapt_structure(to, eq.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, eq.desingularizing_kappa)
    TwoLayerShallowWaterEquations2D(B; ρ1=ρ1, ρ2=ρ2, g=g,
        depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa)
end


conserved_variable_names(::Type{T}) where {T<:TwoLayerShallowWaterEquations2D} = (:h1, :q1, :p1, :w, :q2, :p2)

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

    u1 = desingularize(eq, h1, q1); v1 = desingularize(eq, h1, p1)
    u2 = desingularize(eq, h2, q2); v2 = desingularize(eq, h2, p2)

    return @SVector[
        # layer 1
        p1,
        p1*u1,
        p1*v1 + g*h1*(h1 + w),

        # layer 2
        p2,
        p2*u2,
        p2*v2 + 0.5*g*w^2 - 0.5*g*r*h1^2 - g*Bface*(r*h1 + w)
    ]
end

# ============================================================
# Eigenvalues
# Keep using h2 as in the 1D routine since the eigenvalues only depend on the local state and Bface
# ============================================================

# core: XDIR uses q1,q2 as momenta
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations2D, ::XDIRT, h1, m1, h2, m2)
    eq1d = TwoLayerShallowWaterEquations1D(eq.B; ρ1=eq.ρ1, ρ2=eq.ρ2, g=eq.g,
                                          depth_cutoff=eq.depth_cutoff,
                                          desingularizing_kappa=eq.desingularizing_kappa)
    return compute_eigenvalues(eq1d, XDIRT(), h1, m1, h2, m2)
end

# core: YDIR uses p1,p2 as momenta in the 1D routine
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations2D, ::YDIRT, h1, m1, h2, m2)
    eq1d = TwoLayerShallowWaterEquations1D(eq.B; ρ1=eq.ρ1, ρ2=eq.ρ2, g=eq.g,
                                          depth_cutoff=eq.depth_cutoff,
                                          desingularizing_kappa=eq.desingularizing_kappa)
    return compute_eigenvalues(eq1d, XDIRT(), h1, m1, h2, m2)
end


function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations2D, dir, h1, m1, h2, m2)
    λ = compute_eigenvalues(eq, dir, h1, m1, h2, m2)
    return maximum(abs, λ)
end
