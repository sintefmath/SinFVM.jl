import DelimitedFiles
using NPZ

using SinSWE
import Meshes
using Parameters
using Printf
using StaticArrays
using CairoMakie


function loadgrid(filename::String; delim=',')
    return DelimitedFiles.readdlm(filename, delim)
end


function coarsen(data, times)
    for c in 1:times
        old_size = size(data)
        newsize = floor.(Int64, old_size./2)
        data_new = zeros(newsize)

        for i in CartesianIndices(data_new)
            for j in max(1, 2*(i[1]-1)):min(old_size[1], 2*(i[1]-1)+1)
                for k in max(1, 2*(i[2]-1)):min(old_size[2], 2*(i[2]-1)+1)
                    data_new[i] += data[CartesianIndex((j, k))] / 4
                end
            end
        end
        data = data_new
    end
    return data
end

@with_kw mutable struct IntervalWriter{WriterType}
    current_t::Float64 = 0.0
    step::Float64
    writer::WriterType
end


function (writer::IntervalWriter)(t, simulator)
    dt = SinSWE.current_timestep(simulator)
    if t + dt >= writer.current_t
        writer.writer(t, simulator)
        writer.current_t += writer.step
    end
end
mkpath("figs")
mkpath("data")

function callback(bottom, basename, t, simulator)

    state = SinSWE.current_interior_state(simulator)
    tstr = @sprintf "%0.3f" t
    # @show collect(B)
    w = collect(state.h)
    h = w .- bottom[3:end-3, 3:end-3]
    hu = collect(state.hu)
    hv = collect(state.hv)
    mkpath("figs/bay/$(basename)")

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        hm = heatmap!(ax1,  collect(h), title="Time t=$(tstr)", colorrange = (0.0, 3.0))
        Colorbar(f[:, end+1], hm)
        save("figs/bay/$(basename)/state_$(tstr).png", f, px_per_unit = 2) 
    end
    
    tstr_print = @sprintf "%d" floor(Int64, t)

    mkpath("data/bay/$(basename)")
    npzwrite("data/bay/$(basename)/w_$(tstr_print).npz", h)
    npzwrite("data/bay/$(basename)/hu_$(tstr_print).npz", hu)
    npzwrite("data/bay/$(basename)/hv_$(tstr_print).npz", hv)
end

for backend in [SinSWE.make_cuda_backend(), SinSWE.make_cpu_backend()]

    terrain = loadgrid("examples/data/bay.txt")
    upper_corner = Float64.(size(terrain))
    coarsen_times  = 2
    terrain_original = terrain
    terrain = coarsen(terrain, coarsen_times)
    mkpath("figs/bay/")
    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        ax2 = Axis(f[2, 1])
    
        heatmap!(ax1, terrain_original, title="original")
        heatmap!(ax2, terrain, title="Coarsened")
        save("figs/bay/terrain_comparison.png", f, px_per_unit = 2) 

    end
    
    grid_size = size(terrain) .- (5, 5)
    grid = SinSWE.CartesianGrid(grid_size...; gc=2, extent=[0 upper_corner[1]; 0 upper_corner[2]])
    infiltration = SinSWE.HortonInfiltration(grid, backend)
    bottom = SinSWE.BottomTopography2D(terrain, backend, grid)
    bottom_source = SinSWE.SourceTermBottom()
    equation = SinSWE.ShallowWaterEquations(bottom)
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)
    constant_rain = SinSWE.ConstantRain(0.01)
    friction = SinSWE.ImplicitFriction()

    conserved_system =
        SinSWE.ConservedSystem(backend, 
        reconstruction,
         numericalflux, 
         equation,
          grid,
           [infiltration, constant_rain, bottom_source],
           friction)
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    u0 = x -> @SVector[0.0, 0.0, 0.0]
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    
    SinSWE.set_current_state!(simulator, initial)
    SinSWE.current_state(simulator).h[1:end, 1:end] = terrain[1:end-1, 1:end-1]
    T = 24*60*60.0
    callback_to_simulator = IntervalWriter(step=60, writer = (t, s) -> callback(terrain, SinSWE.name(backend), t, s))
    SinSWE.simulate_to_time(simulator, T, callback=callback_to_simulator)
end