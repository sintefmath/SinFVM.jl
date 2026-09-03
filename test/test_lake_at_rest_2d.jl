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

using VolumeFluxes
isdefined(Main, :test_backends) || include("testing_utils.jl")
using Test
using StaticArrays
using CairoMakie
using LinearAlgebra


function test_lake_at_rest(backend, grid, B_data, w0, t=0.001; plot=false)

    B = VolumeFluxes.BottomTopography2D(B_data, backend, grid)
    eq = VolumeFluxes.ShallowWaterEquations(B)
    rec = VolumeFluxes.LinearReconstruction(2)
    flux = VolumeFluxes.CentralUpwind(eq)
    bst = VolumeFluxes.SourceTermBottom()
    conserved_system = VolumeFluxes.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = VolumeFluxes.BalanceSystem(conserved_system, bst)
    
    
    timestepper = VolumeFluxes.ForwardEulerStepper()
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = VolumeFluxes.cell_centers(grid)
    xf = VolumeFluxes.cell_faces(grid)
    u0 = x -> @SVector[w0, 0.0, 0.0]
    initial = u0.(x)

    VolumeFluxes.set_current_state!(simulator, initial)
    
    VolumeFluxes.simulate_to_time(simulator, t)

    
    # initial_state = VolumeFluxes.current_interior_state(simulator)
    # lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
    # lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")

    state = VolumeFluxes.current_interior_state(simulator)
    w =  collect(state.h)
    hu = collect(state.hu)
    hv = collect(state.hv)
        
    if plot
        infostring = "lake at rest 
t=$(t) nx=$(nx)
$(typeof(rec)) and $(typeof(flux))"
        f = Figure(size=(800, 800), fontsize=24)
        ax_B = Axis(
            f[1, 1],
            title=infostring*"
terrain",
        )
        hm = heatmap!(ax_B, collect(B.B[3:end-2, 3:end-2]))
        Colorbar(f[1, 2], hm)
        ax_w = Axis(
            f[1, 3],
            title="w",
        )
        hm = heatmap!(ax_w, w)
        Colorbar(f[1, 4], hm)
        ax_hu = Axis(
            f[2, 1],
            title="hu (max = $(round(maximum(abs.(hu)); digits=18)))",
        )
        hm = heatmap!(ax_hu, hu)
        Colorbar(f[2, 2], hm)
        ax_hv = Axis(
            f[2, 3],
            title="hv (max = $(round(maximum(abs.(hv)); digits=18)))",
        )
        hm = heatmap!(ax_hv, hv)
        Colorbar(f[2, 4], hm)
    
        # axislegend(ax_h)
        # axislegend(ax_u)
        maybe_display(f)
    end
    @test maximum(abs.(hu)) ≈ 0.0 atol=test_atol(backend, 10^-14)
    @test maximum(abs.(w[1] - w0)) ≈ 0.0 atol=test_atol(backend, 10^-14)
end


@testset "$(backend_label(backend))" for backend in test_backends()
    nx = 64
    ny = 64
    grid = VolumeFluxes.CartesianGrid(nx, ny; gc=2, boundary=VolumeFluxes.WallBC(), extent=[0.0  10.0; 0.0 10.0], )
    x0 = 5.0
    y0 = 5.0
    B = [x[1] < x0 && x[2] < y0 ? 0.45 : 0.55 for x in VolumeFluxes.cell_faces(grid, interior=false)]

    nx_bumpy = 124
    ny_bumpy = 124
    grid_bumpy = VolumeFluxes.CartesianGrid(nx_bumpy, ny_bumpy; gc=2, boundary=VolumeFluxes.WallBC(), extent=[-2*pi  2*pi; -2*pi 2*pi], )
    B_bumpy = [(cos(x[1] + x[2])-0.5 - 1.5*(norm(x) < 1.0)) for x in VolumeFluxes.cell_faces(grid_bumpy, interior=false)]

   @testset "lake_at_rest_$(backend_label(backend))" begin

        test_lake_at_rest(backend, grid, B, 0.7, 0.01, plot=false)
        test_lake_at_rest(backend, grid_bumpy, B_bumpy, 0.7, 0.01, plot=false)
   end
end

# bst = VolumeFluxes.SourceTermBottom()
# rain = VolumeFluxes.SourceTermRain(1.0)
# infl = VolumeFluxes.SourceTermInfiltration(-1.0)

# v_st::Vector{VolumeFluxes.SourceTerm} = [bst, rain, infl]
# @show(v_st)
#@show maximum(B)  
