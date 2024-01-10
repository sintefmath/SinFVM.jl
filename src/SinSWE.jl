module SinSWE

direction(integer) = Val{integer}

const XDIRT = Val{1}
const YDIRT = Val{2}
const ZDIRT = Val{3}

const XDIR = XDIRT()
const YDIR = YDIRT()
const ZDIR = ZDIRT()

using StaticArrays
using Parameters

include("grid.jl")
include("equation.jl")
include("reconstruction.jl")
include("numericalflux.jl")
include("system.jl")
include("timestepper.jl")
include("simulator.jl")
export XDIR, YDIR, ZDIR, ShallowWaterEquations, Burgers, CartesianGrid
end