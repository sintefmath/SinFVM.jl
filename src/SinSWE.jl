module SinSWE
using Logging
direction(integer) = Val{integer}

const XDIRT = Val{1}
const YDIRT = Val{2}
const ZDIRT = Val{3}

const XDIR = XDIRT()
const YDIR = YDIRT()
const ZDIR = ZDIRT()

Base.to_index(::XDIRT) = 1
Base.to_index(::YDIRT) = 2
Base.to_index(::ZDIRT) = 3

const Direction = Union{XDIRT, YDIRT, ZDIRT}

using StaticArrays
using Parameters

include("abstract_types.jl")
include("meta/staticvectors.jl")
include("grid.jl")
include("backends/kernel_abstractions_cuda.jl")

include("meta/loops.jl")
include("backends/kernel_abstractions.jl")
include("backends/buffer.jl")
include("bottom_topography.jl")
include("equations/equation.jl")
include("volume/volume.jl")
include("reconstruction.jl")
include("numericalflux.jl")
include("system.jl")
include("timestepper.jl")
include("simulator.jl")

include("source_terms.jl")
include("bc.jl")
export XDIR, YDIR, ZDIR, ShallowWaterEquations, Burgers, CartesianGrid, make_cpu_backend, make_cuda_backend, Volume, get_available_backends
end
