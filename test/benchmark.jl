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

using Plots
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinFVM
using Parameters


@with_kw mutable struct CountTimesteps
    current_timestep::Int64 = 0
end

function (counter::CountTimesteps)(t, sim)
    counter.current_timestep += 1
end
function run_simulation(nx; backend=make_cuda_backend())
    u0 = x -> sin.(2π * x)
    grid = SinFVM.CartesianGrid(nx)

    equation = SinFVM.Burgers()
    reconstruction = SinFVM.NoReconstruction()
    numericalflux = SinFVM.Godunov(equation)
    conserved_system = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinFVM.ForwardEulerStepper()

    x = SinFVM.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

    SinFVM.set_current_state!(simulator, initial)

    t = 0.0

    T = 1.0

    time_scale_threshold = 16 * 1024
    if nx > time_scale_threshold
        T = (Float64(time_scale_threshold) / Float64(nx)) * T
    end
    # plot(x, first.(SinFVM.current_interior_state(simulator)))
    if nx <= 64 * 1024
        println("Running SinFVM twice")
        @time SinFVM.simulate_to_time(simulator, T)
    end
    SinFVM.set_current_state!(simulator, initial)
    timestep_counter = CountTimesteps()
    result_sinswe = @timed SinFVM.simulate_to_time(simulator, T, callback=timestep_counter)
    plot!(x, first.(collect(SinFVM.current_interior_state(simulator))))


    number_of_x_cells = nx

    if nx <= 64 * 1024
        println("Running bare bones twice")
        @time xcorrect, ucorrect, _ = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    end

    result_barebones = @timed xcorrect, ucorrect, timesteps_barebones = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())

    return (result_sinswe, timestep_counter.current_timestep, result_barebones, timesteps_barebones)
end

function benchmark(outname, backend)
    open(outname, "w") do io
        write(io, "resolution,time_swe,bytes_swe,gctime_swe,timesteps_swe,time_bb,bytes_bb,gctime_bb,timesteps_bb
")
    end
    resolutions = 2 .^ (2:28)

    for resolution in resolutions
        println("resolution = $resolution")
        open(outname, "a") do io
            result_sinswe, timesteps_swe, result_barebones, timesteps_barebones = run_simulation(resolution, backend=backend)
            write(io, "$(resolution),$(result_sinswe.time),$(result_sinswe.bytes),$(result_sinswe.gctime),$(timesteps_swe),$(result_barebones.time),$(result_barebones.bytes),$(result_barebones.gctime),$(timesteps_barebones)
")
        end
    end
end
benchmark("results_cuda.txt", make_cuda_backend())
benchmark("results_cpu.txt", make_cpu_backend())
