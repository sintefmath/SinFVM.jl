abstract type Equation end

struct ShallowWaterEquations1D{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquations1D() = ShallowWaterEquations1D(1.0, 9.81)

function (eq::ShallowWaterEquations1D)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
    ]
end

function compute_eigenvalues(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    g = eq.g
    u = hu / h
    return @SVector [u + sqrt(g*h), u - sqrt(g*h)]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    # TODO: Use compute_eigenvalues
    g = eq.g
    u = hu / h
    return max(abs(u + sqrt(g*h)), abs(u - sqrt(g*h)))
end
number_of_conserved_variables(::ShallowWaterEquations1D) = 2

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

struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = @SVector [0.5 * u .^ 2]

compute_max_abs_eigenvalue(::Burgers, ::XDIRT, u) = abs(first(u))
number_of_conserved_variables(::Burgers) = 1
