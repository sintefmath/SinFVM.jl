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

using SinFVM
using Test
using StaticArrays
using CairoMakie

function test_lake_at_rest(backend, grid, B_data, w0, t=0.001; plot=true)

    B = SinFVM.BottomTopography1D(B_data, backend, grid)
    eq = SinFVM.ShallowWaterEquations1D(B)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = SinFVM.BalanceSystem(conserved_system, bst)
    
    
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    u0 = x -> @SVector[w0, 0.0]
    initial = u0.(x)

    SinFVM.set_current_state!(simulator, initial)
    
    SinFVM.simulate_to_time(simulator, t)

    
    # initial_state = SinFVM.current_interior_state(simulator)
    # lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
    # lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")

    state = SinFVM.current_interior_state(simulator)
    w =  collect(state.h)
    hu = collect(state.hu)
        
    if plot
        f = Figure(size=(800, 800), fontsize=24)
        infostring = "lake at rest 
t=$(t) nx=$(nx)
$(typeof(rec)) and $(typeof(flux))"
        ax_h = Axis(
            f[1, 1],
            title="water and terrain"*infostring,
            ylabel="y",
            xlabel="x",
        )

        lines!(ax_h, x, w, color="blue", label='w')
        lines!(ax_h, xf, collect(B.B[3:end-2]), label='B', color="red")

        ax_u = Axis(
            f[2, 1],
            title="hu",
            ylabel="hu",
            xlabel="x",
        )

        lines!(ax_u, x, hu, label="hu")

        # axislegend(ax_h)
        # axislegend(ax_u)
        display(f)
    end
    @test maximum(abs.(hu)) ≈ 0.0 atol=10^-14
    @test maximum(abs.(w[1] - w0)) ≈ 0.0 atol=10^-14
end





for backend in SinFVM.get_available_backends()
    nx = 64
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0  10.0], )
    x0 = 5.0
    B = [x < x0 ? 0.45 : 0.55 for x in SinFVM.cell_faces(grid, interior=false)]
    
    nx_bumpy = 1024
    grid_bumpy = SinFVM.CartesianGrid(nx_bumpy; gc=2, boundary=SinFVM.PeriodicBC(), extent=[-2*pi  2*pi], )
    B_bumpy = [(cos(x)-0.5 - 1.5*(abs(x) < 1.0)) for x in SinFVM.cell_faces(grid_bumpy, interior=false)]
    
    @testset "lake_at_rest_$(SinFVM.name(backend))" begin

        # test_lake_at_rest(grid, B, 0.7, plot=false)
        test_lake_at_rest(backend, grid, B, 0.7, 0.01, plot=false)

        test_lake_at_rest(backend, grid_bumpy, B_bumpy, 0.7, 0.01, plot=false)
    end
end

# bst = SinFVM.SourceTermBottom()
# rain = SinFVM.SourceTermRain(1.0)
# infl = SinFVM.SourceTermInfiltration(-1.0)

# v_st::Vector{SinFVM.SourceTerm} = [bst, rain, infl]
# @show(v_st)
#@show maximum(B)
