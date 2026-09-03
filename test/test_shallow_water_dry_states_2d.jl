# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test

module Correct
include("fasit.jl")
end
using VolumeFluxes
isdefined(Main, :maybe_display) || include("testing_utils.jl")
function run_swe_2d_pure_simulation(backend)

    backend_name = VolumeFluxes.name(backend)
    nx = 64
    ny = 64
    grid = VolumeFluxes.CartesianGrid(nx, ny; gc=2)
    
    equation = VolumeFluxes.ShallowWaterEquations()
    reconstruction = VolumeFluxes.LinearReconstruction()
    numericalflux = VolumeFluxes.CentralUpwind(equation)

    conserved_system =
        VolumeFluxes.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = VolumeFluxes.RungeKutta2()
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid)
    
    # Two ways for setting initial conditions:
    # 1) Directly
    x = VolumeFluxes.cell_centers(grid)
    #u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    u0 = x -> @SVector[1 - sign((norm(x .- 0.5)^2 - 0.01)), 0.0, 0.0]
    initial = u0.(x)
    VolumeFluxes.set_current_state!(simulator, initial)
    
    # 2) Via volumes:
    # init_volume = VolumeFluxes.Volume(backend, equation, grid)
    # CUDA.@allowscalar VolumeFluxes.InteriorVolume(init_volume)[:, :] = [SVector{3, Float64}(exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0) for x in x]
    # VolumeFluxes.set_current_state!(simulator, init_volume)

    T = 0.05

    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles=["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j - 1], ylabel=L"y", xlabel=L"x", title="$(titles[j])
$(names[i])") for i in 1:3 ] for j in 1:2]
    
    
    current_simulator_state = collect(VolumeFluxes.current_state(simulator))
    @test !any(isnan.(current_simulator_state))
    
    initial_state = VolumeFluxes.current_interior_state(simulator)
    hm = heatmap!(axes[1][1], collect(initial_state.h))
    Colorbar(f[1, 2], hm)
    hm = heatmap!(axes[1][2], collect(initial_state.hu))
    Colorbar(f[2, 2], hm)
    hm = heatmap!(axes[1][3], collect(initial_state.hv))
    Colorbar(f[3, 2], hm)
    maybe_display(f)
    @time VolumeFluxes.simulate_to_time(simulator, T)
    @test VolumeFluxes.current_time(simulator) == T
    
    result = VolumeFluxes.current_interior_state(simulator)
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
    maybe_display(f)

    # Test symmetry (field[x, y])
    tolerance = 10^-12
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

# run_swe_2d_pure_simulation(VolumeFluxes.make_cuda_backend())

for backend in get_available_backends()
    run_swe_2d_pure_simulation(backend)
end
