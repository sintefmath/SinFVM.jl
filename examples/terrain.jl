import DelimitedFiles
using NPZ

using SinSWE
import Meshes
using Parameters
using Printf
using StaticArrays
using CairoMakie
import CUDA

include("example_tools.jl")


for backend in [SinSWE.make_cpu_backend(), SinSWE.make_cuda_backend()]

    terrain = loadgrid("examples/data/bay.txt")
    upper_corner = Float64.(size(terrain))
    coarsen_times = 2
    terrain_original = terrain
    terrain = coarsen(terrain, coarsen_times)
    mkpath("figs/bay/")

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        ax2 = Axis(f[2, 1])

        heatmap!(ax1, terrain_original, title="original")
        heatmap!(ax2, terrain, title="Coarsened")
        save("figs/bay/terrain_comparison.png", f, px_per_unit=2)

    end

    grid_size = size(terrain) .- (5, 5)
    grid = SinSWE.CartesianGrid(grid_size...; gc=2, boundary=SinSWE.NeumannBC(), extent=[0 upper_corner[1]; 0 upper_corner[2]])
    infiltration = SinSWE.HortonInfiltration(grid, backend)
    #infiltration = SinSWE.ConstantInfiltration(15 / (1000.0) / 3600.0)
    bottom = SinSWE.BottomTopography2D(terrain, backend, grid)
    bottom_source = SinSWE.SourceTermBottom()
    equation = SinSWE.ShallowWaterEquations(bottom; depth_cutoff=8e-2)
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)
    constant_rain = SinSWE.ConstantRain(15 / (1000.0))
    friction = SinSWE.ImplicitFriction()

    with_theme(theme_latexfonts()) do
        f = Figure(xlabel="Time", ylabel="Infiltration")
        ax = Axis(f[1, 1])
        t = LinRange(0, 60 * 60 * 24.0, 10000)
        CUDA.@allowscalar infiltrationf(t) = SinSWE.compute_infiltration(infiltration, t, CartesianIndex(30, 30))
        CUDA.@allowscalar lines!(ax, t ./ 60 ./ 60, infiltrationf.(t))

        save("figs/bay/infiltration.png", f, px_per_unit=2)
    end

    conserved_system =
        SinSWE.ConservedSystem(backend,
            reconstruction,
            numericalflux,
            equation,
            grid,
            [
                infiltration,
                constant_rain,
                bottom_source
            ],
            friction)
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    u0 = x -> @SVector[0.0, 0.0, 0.0]
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)

    SinSWE.set_current_state!(simulator, initial)
    SinSWE.current_state(simulator).h[1:end, 1:end] = bottom_per_cell(bottom)
    T = 24 * 60 * 60.0
    callback_to_simulator = IntervalWriter(step=10., writer=(t, s) -> callback(terrain, SinSWE.name(backend), t, s))

    total_water_writer = TotalWaterVolume(bottom_topography=bottom)
    total_water_writer(0.0, simulator)
    total_water_writer_interval_writer = IntervalWriter(step=10., writer=total_water_writer)

    SinSWE.simulate_to_time(simulator, T; maximum_timestep=1.0, callback=MultipleCallbacks([callback_to_simulator, total_water_writer_interval_writer]))
end
