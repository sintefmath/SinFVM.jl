module Rumpetroll

direction(integer) = Val{integer}

const XDIRT = Val{1}
const YDIRT = Val{2}
const ZDIRT = Val{3}

const XDIR = XDIRT()
const YDIR = YDIRT()
const ZDIR = ZDIRT()

using StaticArrays

struct ShallowWaterEquations{T}
    ρ::T
    g::T
end

ShallowWaterEquations() = ShallowWaterEquations(1.0, 9.81)

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  [
        ρ * hu,
        ρ * hu  * hu / h + 0.5 * ρ * g * h^2 ,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return  [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv  * hv / h + 0.5 * ρ * g * h^2 ,
    ]
end
export XDIR, YDIR, ZDIR, ShallowWaterEquations
end

equation = Rumpetroll.ShallowWaterEquations()

@show equation(Rumpetroll.XDIR, 1.0, 2.0, 3.0)
@show equation(Rumpetroll.YDIR, 1.0, 2.0, 3.0)

using Symbolics

@variables h hu hv

Gres = equation(Rumpetroll.XDIR, h, hu, hv)

@show Symbolics.jacobian(Gres, [h, hu, hv])

using AbstractAlgebra

eigenvalues = det(Gres)