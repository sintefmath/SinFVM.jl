
struct ShallowWaterEquations{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquations() = ShallowWaterEquations(1.0, 9.81)

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv * hv / h + 0.5 * ρ * g * h^2,
    ]
end

conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations} = (:h, :hu, :hv)

function compute_eigenvalues(eq::ShallowWaterEquations, ::XDIRT, h, hu, hv)
    g = eq.g
    u = hu / h
    return @SVector [u + sqrt(g * h), u - sqrt(g * h), u]
end



function compute_eigenvalues(eq::ShallowWaterEquations, ::YDIRT, h, hu, hv)
    g = eq.g
    v = hv / h
    return @SVector [v + sqrt(g * h), v - sqrt(g * h), v]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1DPure, direction, h, hu, hv)
    return maximum(compute_eigenvalues(eq, direction, h, hu, hv))
end