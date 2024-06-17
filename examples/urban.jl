import CairoMakie, Images # for visualization and loading of textures
import ColorSchemes
import ArchGDAL # for loading of topographical grids from files in geotiff format
using CairoMakie
using SinSWE
import CUDA
using StaticArrays
include("example_tools.jl")
datapath_testdata() = "/home/kjetil/projects/swamp/SinSWE.jl/"
# ## Prepare data

# The package with SWIM testdata is provided as a Julia artifact, which can be
# accessed using the function `datapath_testdata`.  We subsequently load a
# digital surface model (including buildings and vegetation) and a digtial
# terrain model (without buildings or vegetation) of the area of study
# [^1]. The data is converted into simple Julia arrays with height values
# stored as Float64.
kuba_datapath = joinpath(datapath_testdata(), "data", "kuba")
geoarray_dsm = ArchGDAL.readraster(joinpath(kuba_datapath, "dom1", "data", "dom1.tif"))
geoarray_dtm = ArchGDAL.readraster(joinpath(kuba_datapath, "dtm1", "data", "dtm1.tif"))

grid_dtm = permutedims(geoarray_dtm[:, :, 1]) .|> Float64
grid_dsm = permutedims(geoarray_dsm[:, :, 1]) .|> Float64
typeof(grid_dtm), size(grid_dtm)

# In addition to the elevation data, we also load a set of textures and masks that can
# be used to visualize the model, as well as indicate the locations of buildings,
# permeable areas, rivers and sinks (e.g. manholes).
mapimg = Images.load(joinpath(kuba_datapath, "textures", "kuba.png"))
photoimg = Images.load(joinpath(kuba_datapath, "textures", "kuba_photo.png"))

building_mask = Images.load(joinpath(kuba_datapath, "textures", "building_mask.png"))
pavement_mask = Images.load(joinpath(kuba_datapath, "textures", "pavement_mask.png"))
river_mask = Images.load(joinpath(kuba_datapath, "textures", "river_mask.png"))
permeable_mask = Images.load(joinpath(kuba_datapath, "textures", "permeable_mask.png"))
sink_mask = Images.load(joinpath(kuba_datapath, "textures", "sink_mask.png"))
typeof(sink_mask), size(sink_mask)

# The textures are not all of the same resolution.  We resize them all to equal
# resolution; twice the topographical grid resolution in both directions.
mapimg = Images.imresize(mapimg, size(grid_dtm))# .* 2)
photoimg = Images.imresize(photoimg, size(mapimg))

building_mask = Images.imresize(building_mask, size(mapimg))
pavement_mask = Images.imresize(pavement_mask, size(mapimg))
river_mask = Images.imresize(river_mask, size(mapimg))
permeable_mask = Images.imresize(permeable_mask, size(mapimg))
sink_mask = Images.imresize(sink_mask, size(mapimg))
size(sink_mask)
with_theme(theme_latexfonts()) do
    f = Figure()
    ax1 = Axis(f[1, 1])
    with_buildings = grid_dtm .+ (Float64.(building_mask) .!= 0.0) .* (grid_dsm .- grid_dtm)
    only_buildings = (Float64.(building_mask) .!= 0.0)
    only_buildings_mask = only_buildings .!= 0.0
    factor = Float64.(permeable_mask) .* (1.0 .- (Float64.(building_mask) .!= 0.0))
    hm = heatmap!(ax1, only_buildings_mask .- factor)
    Colorbar(f[1, 2], hm)
    display(f)
end

with_theme(theme_latexfonts()) do
    f = Figure()
    ax1 = Axis(f[1, 1])
    hm = heatmap!(ax1, Float64.(permeable_mask) .* (1.0 .- Float64.(building_mask)))
    Colorbar(f[1, 2], hm)
    display(f)
end


with_theme(theme_latexfonts()) do
    f = Figure()
    ax1 = Axis(f[1, 1])
    hm = heatmap!(ax1, Float64.(grid_dsm .- grid_dtm))
    Colorbar(f[1, 2], hm)
    display(f)
end


include("example_tools.jl")


for backend in [SinSWE.make_cpu_backend(), SinSWE.make_cuda_backend()]
    with_buildings = grid_dtm .+ (Float64.(building_mask) .!= 0.0) .* (grid_dsm .- grid_dtm)
    terrain = with_buildings
    upper_corner = Float64.(size(terrain))
    coarsen_times = 2
    terrain_original = terrain
    #terrain = coarsen(terrain, coarsen_times)
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

    factor_cpu = bottom_per_cell(Float64.(permeable_mask) .* (1.0 .- (Float64.(building_mask) .!= 0.0)))
    factor = SinSWE.convert_to_backend(backend, factor_cpu)

    infiltration = SinSWE.HortonInfiltration(grid, backend; factor=factor)
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
    callback_to_simulator = IntervalWriter(step=10.0, writer=(t, s) -> callback(terrain, SinSWE.name(backend), t, s))

    total_water_writer = TotalWaterVolume(bottom_topography=bottom)
    total_water_writer(0.0, simulator)
    total_water_writer_interval_writer = IntervalWriter(step=10.0, writer=total_water_writer)

    SinSWE.simulate_to_time(simulator, T; maximum_timestep=60.0, callback=MultipleCallbacks([callback_to_simulator, total_water_writer_interval_writer]))
end