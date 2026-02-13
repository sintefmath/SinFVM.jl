using StaticArrays
using Adapt
using SinFVM

# ============================================================
# Two-layer SWE in 2D
# Conserved variables:
#   U = (h1, q1, p1, h2, q2, p2)
# where q_i = h_i u_i,  p_i = h_i v_i
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

conserved_variable_names(::Type{T}) where {T<:TwoLayerShallowWaterEquations2D} = (:h1, :q1, :p1, :h2, :q2, :p2)

# x-direction (F(U,B))
function (eq::TwoLayerShallowWaterEquations2D)(::XDIRT, h1, q1, p1, h2, q2, p2, Bface)
    g  = eq.g
    ρ1 = eq.ρ1
    ρ2 = eq.ρ2
    r = ρ1/ρ2

    u1 = desingularize(eq, h1, q1); v1 = desingularize(eq, h1, p1)
    u2 = desingularize(eq, h2, q2); v2 = desingularize(eq, h2, p2)

    return @SVector[
        # layer 1
        q1,
        q1*u1 + g*h1*(h1 + h2 + Bface),
        q1*v1,

        # layer 2
        q2,
        q2*u2 + 0.5*g*(h2 +Bface)^2 -0.5*g*r*(h1)^2 - g*Bface*(r*h1 + h2 + Bface), 
        q2*v2
    ]
end

# y-direction G(U,B)
function (eq::TwoLayerShallowWaterEquations2D)(::YDIRT, h1, q1, p1, h2, q2, p2, Bface)
    g  = eq.g
    ρ1 = eq.ρ1
    ρ2 = eq.ρ2
    r = ρ1/ρ2

    u1 = desingularize(eq, h1, q1); v1 = desingularize(eq, h1, p1)
    u2 = desingularize(eq, h2, q2); v2 = desingularize(eq, h2, p2)

    return @SVector[
        # layer 1
        p1,
        p1*u1,
        p1*v1 + g*h1*(h1 + h2 + Bface),

        # layer 2
        p2,
        p2*u2,
        p2*v2 + 0.5*g*(h2 +Bface)^2 - 0.5*g*r*(h1)^2 - g*Bface*(r*h1 + h2 + Bface)
    ]
end


#Make 1D version of equation to compute eigenvalues in each direction using the same code as in the 1D case
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations2D, ::XDIRT, h1, q1, p1, h2, q2, p2)
    return compute_eigenvalues(TwoLayerShallowWaterEquations1D(eq.B; ρ1=eq.ρ1, ρ2=eq.ρ2, g=eq.g,depth_cutoff=eq.depth_cutoff, desingularizing_kappa=eq.desingularizing_kappa),
                               XDIRT(), h1, q1, h2, q2)
end

#Need to pass XDIRT rutine in 1D to compute the eigenvalues in the y-direction by passing the y-components of the conserved variables instead of the x-components
function compute_eigenvalues(eq::TwoLayerShallowWaterEquations2D, ::YDIRT, h1, q1, p1, h2, q2, p2)
    return compute_eigenvalues(TwoLayerShallowWaterEquations1D(eq.B; ρ1=eq.ρ1, ρ2=eq.ρ2, g=eq.g, depth_cutoff=eq.depth_cutoff, desingularizing_kappa=eq.desingularizing_kappa),
                               XDIRT(), h1, p1, h2, p2)
end

function compute_max_abs_eigenvalue(eq::TwoLayerShallowWaterEquations2D, dir, h1, q1, p1, h2, q2, p2)
    λ = compute_eigenvalues(eq, dir, h1, q1, p1, h2, q2, p2)
    return maximum(abs, λ)
end
