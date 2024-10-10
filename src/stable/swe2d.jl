
struct ShallowWaterEquationsStable{T, S} <: Equation
    B::S
    ρ::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    ShallowWaterEquationsStable(B::BottomType=ConstantBottomTopography(); ρ=1.0, g=9.81, depth_cutoff=10^-5, desingularizing_kappa=10^-5) where {BottomType <: AbstractBottomTopography} = new{typeof(g), typeof(B)}(B, ρ, g, depth_cutoff, desingularizing_kappa)
end
function Adapt.adapt_structure(
    to,
    swe::ShallowWaterEquationsStable{T, S}
) where {T, S}
    B = Adapt.adapt_structure(to, swe.B)
    ρ = Adapt.adapt_structure(to, swe.ρ)
    g = Adapt.adapt_structure(to, swe.g)
    depth_cutoff = Adapt.adapt_structure(to, swe.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, swe.desingularizing_kappa)
    
    ShallowWaterEquationsStable(B; ρ=ρ, g=g, depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa)
end

function (eq::ShallowWaterEquationsStable)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    v = desingularize(eq, h, hv)
    return @SVector [
        ρ * h * u,
        ρ * h * u * u + 0.5 * ρ * g * h^2,
        ρ * h * u * v 
    ]
end

function (eq::ShallowWaterEquationsStable)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    v = desingularize(eq, h, hv)
    return @SVector [
        ρ * h * v,
        ρ * h * u * v,
        ρ * h * v * v + 0.5 * ρ * g * h^2,
    ]
end

conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquationsStable} = (:h, :hu, :hv)

function compute_eigenvalues(eq::ShallowWaterEquationsStable, ::XDIRT, h, hu, hv)
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [u + sqrt(g * h), u - sqrt(g * h), u]
end



function compute_eigenvalues(eq::ShallowWaterEquationsStable, ::YDIRT, h, hu, hv)
    g = eq.g
    v = desingularize(eq, h, hv)
    return @SVector [v + sqrt(g * h), v - sqrt(g * h), v]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquationsStable, direction, h, hu, hv)
    return maximum(abs.(compute_eigenvalues(eq, direction, h, hu, hv)))
end