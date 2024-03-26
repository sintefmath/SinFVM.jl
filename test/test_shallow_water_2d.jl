using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test

module Correct
include("fasit.jl")
end
using SinSWE
function run_swe_2d_pure_simulation(backend)

    backend_name = SinSWE.name(backend)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    nx = 64
    ny = 64
    grid = SinSWE.CartesianGrid(nx, ny; gc=1)
    
    equation = SinSWE.ShallowWaterEquations()

    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    T = 0.05

    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles=["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j - 1], ylabel=L"y", xlabel=L"x", title="$(titles[j])\n$(names[i])") for i in 1:3 ] for j in 1:2]
    
    
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)
    SinSWE.set_current_state!(simulator, initial)
    current_simulator_state = collect(SinSWE.current_state(simulator))
    @test !any(isnan.(current_simulator_state))
    
    initial_state = SinSWE.current_interior_state(simulator)
    hm = heatmap!(axes[1][1], collect(initial_state.h))
    Colorbar(f[1, 2], hm)
    hm = heatmap!(axes[1][2], collect(initial_state.hu))
    Colorbar(f[2, 2], hm)
    hm = heatmap!(axes[1][3], collect(initial_state.hv))
    Colorbar(f[3, 2], hm)

    t = 0.0
    @time SinSWE.simulate_to_time(simulator, T)
    
    result = SinSWE.current_interior_state(simulator)

    hm = heatmap!(axes[2][1], collect(result.h))
    if !any(isnan.(collect(result.h)))
        Colorbar(f[1, 4], hm)
    end
    hm = heatmap!(axes[2][2], collect(result.hu))
    if !any(isnan.(collect(result.hu)))
        Colorbar(f[2, 4], hm)
    end
    
    hm = heatmap!(axes[2][3], collect(result.hv))
    if !any(isnan.(collect(result.hv)))
        Colorbar(f[3, 4], hm)
    end
    display(f)

end

for backend in get_available_backends()
    run_swe_2d_pure_simulation(backend)
end
