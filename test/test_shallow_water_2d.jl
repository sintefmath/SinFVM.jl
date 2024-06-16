using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test
import CUDA

module Correct
include("fasit.jl")
end
using SinSWE
function run_swe_2d_pure_simulation(backend)

    backend_name = SinSWE.name(backend)
    nx = 256
    ny = 32
    grid = SinSWE.CartesianGrid(nx, ny; gc=2)
    
    equation = SinSWE.ShallowWaterEquationsPure()
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)
    T = 0.05
    
    # Two ways for setting initial conditions:
    # 1) Directly
    x = SinSWE.cell_centers(grid)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    initial = u0.(x)
    SinSWE.set_current_state!(simulator, initial)
    
    # 2) Via volumes:
    @show size(x)
    @show size(grid)
    init_volume = SinSWE.Volume(backend, equation, grid)
    CUDA.@allowscalar SinSWE.InteriorVolume(init_volume)[1:end, 1:end] = [SVector{3, Float64}(exp.(-(norm(xi .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0) for xi in x]
    SinSWE.set_current_state!(simulator, init_volume)


    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles=["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j - 1], ylabel=L"y", xlabel=L"x", title="$(titles[j])\n$(names[i])") for i in 1:3 ] for j in 1:2]
    
    
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
    @test SinSWE.current_time(simulator) == T

    result = SinSWE.current_interior_state(simulator)
    h = collect(result.h)
    hu = collect(result.hu)
    hv = collect(result.hv)

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

    # Test symmetry (field[x, y])
    tolerance = 10^-13
    xleft = Int(floor(nx/3))
    xright = nx - xleft + 1
    ylower = Int(floor(ny/3))
    yupper = ny - ylower + 1
    # @show xleft, xright

    @test maximum(h[xleft,:] - h[xright,:]) ≈ 0 atol=tolerance
    @test maximum(h[:, ylower] - h[:, yupper]) ≈ 0 atol=tolerance
    @test maximum(hu[xleft,:] + hu[xright,:]) ≈ 0 atol=tolerance
    @test maximum(hu[:, ylower] - hu[:, yupper]) ≈ 0 atol=tolerance
    @test maximum(hv[xleft,:] - hv[xright,:]) ≈ 0 atol=tolerance
    @test maximum(hv[:, ylower] + hv[:, yupper]) ≈ 0 atol=tolerance
end

for backend in get_available_backends()
    run_swe_2d_pure_simulation(backend)
end
