using Plots
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinSWE
using Parameters


@with_kw mutable struct CountTimesteps
    current_timestep::Int64 = 0
end

function (counter::CountTimesteps)(t, sim)
    counter.current_timestep += 1
end
function run_simulation(nx; backend=make_cuda_backend())
    u0 = x -> sin.(2π * x)
    grid = SinSWE.CartesianGrid(nx)

    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Godunov(equation)
    conserved_system = SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    SinSWE.set_current_state!(simulator, initial)

    t = 0.0

    T = 1.0

    time_scale_threshold = 16 * 1024
    if nx > time_scale_threshold
        T = (Float64(time_scale_threshold) / Float64(nx)) * T
    end
    # plot(x, first.(SinSWE.current_interior_state(simulator)))
    if nx <= 64 * 1024
        println("Running SinSWE twice")
        @time SinSWE.simulate_to_time(simulator, T)
    end
    SinSWE.set_current_state!(simulator, initial)
    timestep_counter = CountTimesteps()
    result_sinswe = @timed SinSWE.simulate_to_time(simulator, T, callback=timestep_counter)
    plot!(x, first.(collect(SinSWE.current_interior_state(simulator))))


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
        write(io, "resolution,time_swe,bytes_swe,gctime_swe,timesteps_swe,time_bb,bytes_bb,gctime_bb,timesteps_bb\n")
    end
    resolutions = 2 .^ (2:28)

    for resolution in resolutions
        println("resolution = $resolution")
        open(outname, "a") do io
            result_sinswe, timesteps_swe, result_barebones, timesteps_barebones = run_simulation(resolution, backend=backend)
            write(io, "$(resolution),$(result_sinswe.time),$(result_sinswe.bytes),$(result_sinswe.gctime),$(timesteps_swe),$(result_barebones.time),$(result_barebones.bytes),$(result_barebones.gctime),$(timesteps_barebones)\n")
        end
    end
end
benchmark("results_cuda.txt", make_cuda_backend())
benchmark("results_cpu.txt", make_cpu_backend())
