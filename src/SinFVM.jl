# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

module SinFVM
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
