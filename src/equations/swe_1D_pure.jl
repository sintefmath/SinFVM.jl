struct ShallowWaterEquations1DPure{T} <: Equation
    ρ::T
    g::T
    ShallowWaterEquations1D(ρ=1.0, g=9.81) = new{typeof(g)}(ρ, g)
end
Adapt.@adapt_structure ShallowWaterEquations1DPure

function (eq::ShallowWaterEquations1DPure)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
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
