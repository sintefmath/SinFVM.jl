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

function tsunami(;T=10, dt=1, w0_height=1.0, bump=false)

    nx = 1024
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0  1000.0], )
    function terrain(x)
        b =  25/1000*(x - 800)
        if bump
            b += 8*exp(-(x - 500)^2/100)
        end
        return b
    end
    
    B_data = Float64[terrain(x) for x in SinFVM.cell_faces(grid, interior=false)]

    backend = make_cpu_backend()
    B = SinFVM.BottomTopography1D(B_data, backend, grid)
    eq = SinFVM.ShallowWaterEquations1D(B)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = SinFVM.BalanceSystem(conserved_system, bst)
    
    
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    u0 = x -> @SVector[w0_height*(x < 100), 0.0]
    initial = u0.(x)

    SinFVM.set_current_state!(simulator, initial)
    all_t = []
    all_h = []
    all_hu = []
    t = 0.0
    while t < T
        t += dt
        SinFVM.simulate_to_time(simulator, t)
        
        state = SinFVM.current_interior_state(simulator)
        push!(all_h, collect(state.h))
        push!(all_hu, collect(state.hu))
        push!(all_t, t)
    end

    index = Observable(1)
    h = @lift(all_h[$index])
    hu = @lift(all_hu[$index])
    t = @lift(all_t[$index])

    f = Figure(size=(800, 800), fontsize=24)
    infostring = @lift("tsunami, t=$(all_t[$index]) nx=$(nx)") #
$(split(typeof(rec),".")[2]) and $(split(typeof(flux), ".")[2])"
        ax_h = Axis(
            f[1, 1],
            title=infostring,
            ylabel="h",
            xlabel="x",
        )

        lines!(ax_h, x, h, color="blue", label='w')
        lines!(ax_h, xf, collect(B.B[3:end-2]), label='B', color="red")

        ax_u = Axis(
            f[2, 1],
            title="hu",
            ylabel="hu",
            xlabel="x",
        )
        ylims!(ax_u, -3, 8)

        lines!(ax_u, x, hu, label="hu")

        record(f, "tsunami.mp4", 1:length(all_t); framerate=10) do i 
            index[] =i                
        end
        #display(f)
end

tsunami(T=200; bump=true)
println("done")
nothing
