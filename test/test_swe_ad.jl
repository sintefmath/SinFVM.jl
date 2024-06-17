using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test
import CUDA
using ForwardDiff
import KernelAbstractions
using SinSWE
import ForwardDiff

function run_swe_2d_ad_simulation(height_and_position)
    # Here we say that we want to have one derivative (height_of_wall)
    ADType = eltype(height_and_position)

    backend = SinSWE.KernelAbstractionBackend(KernelAbstractions.get_backend(ones(3)); realtype=ADType)

    backend_name = SinSWE.name(backend)
    nx = 256
    ny = 32
    grid = SinSWE.CartesianGrid(nx, ny; gc=2)
    dx = SinSWE.compute_dx(grid)
    dy = SinSWE.compute_dy(grid)
    width_of_wall = 4
    length_of_wall = 40

    # Important AD stuff happens here:
    # First allocate the array:
    bottom_topography_array = zeros(ADType, size(grid) .+ (1, 1))

    # Then we set the general slope with no derivative wrt to h, x or y.
    for I in eachindex(bottom_topography_array)
        bottom_topography_array[I] = 1.0 .- dx * I[1]
    end

    height = height_and_position[1]
    xposition_of_wall = height_and_position[2]
    yposition_of_wall = height_and_position[3]
    @show xposition_of_wall / dx
    iposition_of_wall = ceil(Int64, xposition_of_wall / dx)
    jposition_of_wall = ceil(Int64, yposition_of_wall / dy)
    # Now we place the wall:
    for j in 1:width_of_wall
        for i in 1:length_of_wall
            bottom_topography_array[i+iposition_of_wall, j+jposition_of_wall] += height
        end
    end

    bottom_topography = SinSWE.BottomTopography2D(bottom_topography_array, backend, grid)
    bottom_source = SinSWE.SourceTermBottom()
    equation = SinSWE.ShallowWaterEquations(bottom_topography)
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_source])
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)
    T = 0.05

    # Two ways for setting initial conditions:
    # 1) Directly
    x = SinSWE.cell_centers(grid)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    initial = u0.(x)
    SinSWE.set_current_state!(simulator, initial)

    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles = ["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j-1], ylabel=L"y", xlabel=L"x", title="$(titles[j])\n$(names[i])") for i in 1:3] for j in 1:2]


    # IMPORTANT: To get the value, we need to do ForwardDiff.value
    current_simulator_state = ForwardDiff.value.(collect(SinSWE.current_state(simulator)))
    @test !any(isnan.(current_simulator_state))

    initial_state = SinSWE.current_interior_state(simulator)
    hm = heatmap!(axes[1][1], ForwardDiff.value.(collect(initial_state.h)))
    Colorbar(f[1, 2], hm)
    hm = heatmap!(axes[1][2], ForwardDiff.value.(collect(initial_state.hu)))
    Colorbar(f[2, 2], hm)
    hm = heatmap!(axes[1][3], ForwardDiff.value.(collect(initial_state.hv)))
    Colorbar(f[3, 2], hm)

    t = 0.0
    @time SinSWE.simulate_to_time(simulator, T)
    @test SinSWE.current_time(simulator) == T

    result = SinSWE.current_interior_state(simulator)
    h = ForwardDiff.value.(collect(result.h))
    hu = ForwardDiff.value.(collect(result.hu))
    hv = ForwardDiff.value.(collect(result.hv))

    hm = heatmap!(axes[2][1], h)
    if !any(isnan.(h))
        Colorbar(f[1, 4], hm)
    end
    hm = heatmap!(axes[2][2], hu)
    if !any(isnan.(hu))
        Colorbar(f[2, 4], hm)
    end

    hm = heatmap!(axes[2][3], hv)
    if !any(isnan.(hv))
        Colorbar(f[3, 4], hm)
    end
    display(f)

    # Now we can get the derivative of the height of the water wrt to the height of the building as

    derivative = map(x -> x.partials[1], collect(result.h))
    f = Figure(size=(1600, 1200), fontsize=24)
    ax = Axis(f[1, 1])
    hm = heatmap!(ax, derivative)
    Colorbar(f[1, 2], hm)
    display(f)

    return sum(collect(result.h))
end

@show ForwardDiff.gradient(run_swe_2d_ad_simulation, [2.0, 0.2, 0.2])