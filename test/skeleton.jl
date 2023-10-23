module Rumpetroll

direction(integer) = Val{integer}

const XDIRT = Val{1}
const YDIRT = Val{2}
const ZDIRT = Val{3}

const XDIR = XDIRT()
const YDIR = YDIRT()
const ZDIR = ZDIRT()

using StaticArrays


struct CartesianGrid{dimension}
    ghostcells::SVector{dimension, Int64}
    totalcells::SVector{dimension, Int64}
end

function for_each_inner_cell(f, g::CartesianGrid{1}, include_ghostcells=0)
    for i in (g.ghostcells[1]-include_ghostcells+1):(g.totalcells-2*g.ghostcells[1]+include_ghostcells + 1)
        f(i-1, i, i+1)
    end
end
abstract type Equation end
abstract type NumericalFlux end

struct Rusanov{EquationType <: Equation} <: NumericalFlux 
    eq::Equation
end

function (rus::Rusanov)(left, right)
    flux_left = rus.eq(XDIR(), left...)
    flux_right = rus.eq(XDIR(), right...)

    eigenvalue_left = compute_eigenvalue(rus.eq, XDIR(), left...)
    eigenvalue_right = compute_eigenvalue(rus.eq, XDIR(), right...)

    eigenvalue_max = max(eigenvalue_left, eigenvalue_right)

    F = 0.5 .* (flux_left .+ flux_right) .- 0.5 * eigenvalue_max .* (right .- left);

    return F
end


abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end

function reconstruct!(::NoReconstruction, output_left, output_right, input_conserved, grid, equation::Equation)
    for_each_inner_cell(grid) do ileft, imiddle, iright
        output_left[imiddle] = input_conserved[imiddle]
        output_right[imiddle] = input_conserved[imiddle]
    end
end

function compute_flux!(F::NumericalFlux, output, left, right, grid, equation::Equation)
    for_each_inner_cell(grid) do ileft, imiddle, iright
        output[imiddle] = F(left[iright], right[imiddle]) - F(left[imiddle], right[ileft])
    end
end



struct ShallowWaterEquations{T} <: Equation
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





eq = ShallowWaterEquations()

conserved = [h, hu, hv]
eq(XDIR(), conserved...)
eq(XDIR(), 0.5, 0.2, 0.2)

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

