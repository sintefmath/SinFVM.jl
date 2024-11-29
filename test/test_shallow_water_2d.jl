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

using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test
import CUDA
module Correct
include("fasit.jl")
end
using SinFVM
function run_swe_2d_pure_simulation(backend)

    backend_name = SinFVM.name(backend)
    nx = 256
    ny = 32
    grid = SinFVM.CartesianGrid(nx, ny; gc=2)
    
    equation = SinFVM.ShallowWaterEquationsPure()
    reconstruction = SinFVM.LinearReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)

    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)
    T = 0.05
    
    # Two ways for setting initial conditions:
    # 1) Directly
    x = SinFVM.cell_centers(grid)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    initial = u0.(x)
   
    SinFVM.set_current_state!(simulator, initial)
    
    # 2) Via volumes:
    init_volume = SinFVM.Volume(backend, equation, grid)
    CUDA.@allowscalar SinFVM.InteriorVolume(init_volume)[1:end, 1:end] = [SVector{3, Float64}(exp.(-(norm(xi .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0) for xi in x]
    SinFVM.set_current_state!(simulator, init_volume)


    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles=["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j - 1], ylabel=L"y", xlabel=L"x", title="$(titles[j])
$(names[i])") for i in 1:3 ] for j in 1:2]
    
    
    current_simulator_state = collect(SinFVM.current_state(simulator))
    @test !any(isnan.(current_simulator_state))
    
    initial_state = SinFVM.current_interior_state(simulator)
    hm = heatmap!(axes[1][1], collect(initial_state.h))
    Colorbar(f[1, 2], hm)
    hm = heatmap!(axes[1][2], collect(initial_state.hu))
    Colorbar(f[2, 2], hm)
    hm = heatmap!(axes[1][3], collect(initial_state.hv))
    Colorbar(f[3, 2], hm)

    t = 0.0
    @time SinFVM.simulate_to_time(simulator, T)
    @test SinFVM.current_time(simulator) == T

    result = SinFVM.current_interior_state(simulator)
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
