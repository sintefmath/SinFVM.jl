abstract type Equation end

struct ShallowWaterEquations{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquations() = ShallowWaterEquations(1.0, 9.81)

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  @SVector [
        ρ * hu,
        ρ * hu  * hu / h + 0.5 * ρ * g * h^2 ,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  @SVector [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv  * hv / h + 0.5 * ρ * g * h^2 ,
    ]
end

struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = @SVector [0.5 * u.^2]

compute_max_eigenvalue(::Burgers, ::XDIRT, u) = abs(u)
number_of_conserved_variables(::Burgers) = 1
