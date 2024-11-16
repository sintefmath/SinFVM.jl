# # Modelling of a terrain with a bay
# In this example, we will setup a simple model of a terrain with a bay. We will use the Horton infiltration model to simulate the infiltration of water into the soil. The terrain is represented by a grid, and we will use the shallow water equations to simulate the flow of water over the terrain. We will use the `SinFVM` package to setup and solve the model. The model will be solved using the `ForwardEuler` time-stepping method. We will use the `CairoMakie` package to visualize the results of the simulation.

import DelimitedFiles
using NPZ

using SinFVM
import Meshes
using Parameters
using Printf
using StaticArrays
using CairoMakie
import CUDA

include("example_tools.jl")

for backend in get_available_backends()
    dataset_base = joinpath(datapath_testdata(), "data", "small")
    terrain = loadgrid(joinpath(dataset_base, "bay.txt"))
    upper_corner = Float64.(size(terrain))
    coarsen_times = 2
    terrain_original = terrain
    terrain = coarsen(terrain, coarsen_times)
    mkpath("figs/bay/")

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        ax2 = Axis(f[2, 1])

        heatmap!(ax1, terrain_original, label="original")
        heatmap!(ax2, terrain, label="Coarsened")
        save("figs/bay/terrain_comparison.png", f, px_per_unit=2)

    end

    grid_size = size(terrain) .- (5, 5)
    grid = SinFVM.CartesianGrid(grid_size...; gc=2, boundary=SinFVM.NeumannBC(), extent=[0 upper_corner[1]; 0 upper_corner[2]])
    infiltration = SinFVM.HortonInfiltration(grid, backend)
    #infiltration = SinFVM.ConstantInfiltration(15 / (1000.0) / 3600.0)
    bottom = SinFVM.BottomTopography2D(terrain, backend, grid)
    bottom_source = SinFVM.SourceTermBottom()
    equation = SinFVM.ShallowWaterEquations(bottom; depth_cutoff=8e-2)
    reconstruction = SinFVM.LinearReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)
    constant_rain = SinFVM.ConstantRain(15 / (1000.0))
    friction = SinFVM.ImplicitFriction()

    with_theme(theme_latexfonts()) do
        f = Figure(xlabel="Time", ylabel="Infiltration")
        ax = Axis(f[1, 1])
        t = LinRange(0, 60 * 60 * 24.0, 10000)
        CUDA.@allowscalar infiltrationf(t) = SinFVM.compute_infiltration(infiltration, t, CartesianIndex(30, 30))
        CUDA.@allowscalar lines!(ax, t ./ 60 ./ 60, infiltrationf.(t))

        save("figs/bay/infiltration.png", f, px_per_unit=2)
    end

    conserved_system =
        SinFVM.ConservedSystem(backend,
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
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

    u0 = x -> @SVector[0.0, 0.0, 0.0]
    x = SinFVM.cell_centers(grid)
    initial = u0.(x)

    SinFVM.set_current_state!(simulator, initial)
    SinFVM.current_state(simulator).h[1:end, 1:end] = bottom_per_cell(bottom)
    T = 1# 24 * 60 * 60.0
    callback_to_simulator = IntervalWriter(step=10., writer=(t, s) -> callback(terrain, SinFVM.name(backend), t, s))

    total_water_writer = TotalWaterVolume(bottom_topography=bottom)
    total_water_writer(0.0, simulator)
    total_water_writer_interval_writer = IntervalWriter(step=10., writer=total_water_writer)

    SinFVM.simulate_to_time(simulator, T; maximum_timestep=1.0, callback=[callback_to_simulator, total_water_writer_interval_writer])
end
