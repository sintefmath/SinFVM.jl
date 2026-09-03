# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

module VolumeFluxes
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

include("artifacts.jl")
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
include("reconstruction/reconstruction.jl")
include("numericalflux/numericalflux.jl")
include("system.jl")
include("timestepper/timestepper.jl")
include("simulator.jl")

include("sourceterms/source_terms.jl")
include("friction.jl")
include("bc.jl")
include("callbacks.jl")
export XDIR, YDIR, ZDIR, Burgers, CartesianGrid, make_cpu_backend, make_cuda_backend, Volume, get_available_backends, IntervalWriter, Simulator, RungeKutta2, ForwardEulerStepper, ShallowWaterEquations, ShallowWaterEquations1D, ShallowWaterEquations1DPure, ShallowWaterEquationsPure, CentralUpwind, Rusanov, Godunov, LinearReconstruction, NoReconstruction, ConservedSystem, simulate_to_time
end
