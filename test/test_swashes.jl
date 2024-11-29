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

using StaticArrays
using CairoMakie
using Test
import CUDA

include("swashes.jl")

function compare_swashes(sw::Swashes41x, nx, t)
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=getExtent(sw), )
    backend = make_cpu_backend()
    eq = SinFVM.ShallowWaterEquations1D(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    rec = SinFVM.LinearReconstruction(1.2)
    flux = SinFVM.CentralUpwind(eq)
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid)
    # TODO: Second order timestepper
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    initial = get_initial_conditions(sw, grid, eq, backend)
    SinFVM.set_current_state!(simulator, initial)
 
    SinFVM.simulate_to_time(simulator, t)
    #SinFVM.simulate_to_time(simulator, 0.000001)

    f = Figure(size=(1600, 600), fontsize=24)
    x = SinFVM.cell_centers(grid)
    infostring = "swashes test case $(sw.id) $(sw.name) t=$(t) nx=$(nx) $(typeof(eq)) $(typeof(rec)) $(typeof(flux))"
    ax_h = Axis(
        f[1, 1],
        title="h ",
        ylabel="h",
        xlabel="x",
    )

    ax_u = Axis(
        f[1, 2],
        title="u ",
        ylabel="u",
        xlabel="x",
    )
    supertitle = Label(f[0, :], infostring)



    ref_sol = SinFVM.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    lines!(ax_h, x, collect(ref_sol.h), label="swashes")
    lines!(ax_u, x, collect(ref_sol.hu), label="swashes")

    @show SinFVM.current_timestep(simulator)
    our_sol = SinFVM.current_interior_state(simulator)
    hu = collect(our_sol.hu)
    h  = collect(our_sol.h) 
    u = SinFVM.desingularize.(Ref(eq), h, hu)
    lines!(ax_h, x, h, label="swamp")
    lines!(ax_u, x, u, label="swamp")

    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
    # plot_simulator_state(simulator)
end

function plot_simulator_state(simulator, verbose=false)
    x_cells = SinFVM.cell_centers(simulator.grid)
    x_faces = SinFVM.cell_faces(simulator.grid)
    b_cells = SinFVM.collect_topography_cells(simulator.system.equation.B, simulator.grid)
    b_faces = SinFVM.collect_topography_intersections(simulator.system.equation.B, simulator.grid)
    if verbose @show size(b_cells) end
    if verbose @show size(b_faces) end
    if verbose @show size(x_cells) end
    if verbose @show size(x_faces) end

    prev_state = SinFVM.InteriorVolume(simulator.substep_outputs[2])
    w = collect(prev_state.h)
    if verbose @show size(w) end
    h = w .- b_cells
    hu = collect(prev_state.hu)
    if verbose @show size(hu) end

    interior_right = SinFVM.InteriorVolume(simulator.system.right_buffer)
    interior_left  = SinFVM.InteriorVolume(simulator.system.left_buffer)
    # h_right = collect(simulator.system.right_buffer.h)
    h_right = collect(interior_right.h)
    w_right = h_right .+ b_faces[2:end]
    hu_right = collect(interior_right.hu)
    h_left = collect(interior_left.h)
    w_left = h_left .+ b_faces[1:end-1]
    hu_left = collect(interior_left.hu)

    f = Figure(size=(1600, 600), fontsize=24)
    ax_h = Axis(f[1, 1], title="h" , ylabel="h", xlabel="x")
    ax_w = Axis(f[1, 2], title="w",  ylabel="z", xlabel="x")
    ax_hu = Axis(f[1, 3], title="hu", ylabel="h", xlabel="x")
    supertitle = Label(f[0, :], "Simulator state")

    scatter!(ax_h, x_faces[1:end-1], h_left, label="h left")
    scatter!(ax_h, x_faces[2:end], h_right, label="h right")
    scatter!(ax_h, x_cells, h, label="h")

    scatter!(ax_w, x_faces[1:end-1], w_left, label="w left")
    scatter!(ax_w, x_faces[2:end], w_right, label="w right")
    scatter!(ax_w, x_cells, w, label="w")
    lines!(ax_w, x_cells, b_cells, label="B cells")
    lines!(ax_w, x_faces, b_faces, label="B faces")
    
    scatter!(ax_hu, x_faces[1:end-1], hu_left, label="hu left")
    scatter!(ax_hu, x_faces[2:end], hu_right, label="hu right")
    scatter!(ax_hu, x_cells, hu, label="hu")

    for ax in [ax_h, ax_w, ax_hu]
        axislegend(ax)
    end
    display(f)

end

function compare_swashes(sw::Swashes421, nx, t)

    dx = 4.0/nx
    xl_start = Int(floor(0.45/dx))
    xl_end = Int(floor(0.55/dx))
    xr_start = Int(floor(2.45/dx))
    xr_end = Int(floor(2.55/dx))

    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=getExtent(sw), )
    backend = make_cpu_backend()
    topography = get_bottom_topography(sw, grid, backend)
    eq = SinFVM.ShallowWaterEquations1D(topography; depth_cutoff=10^-3, desingularizing_kappa=10^-4)
    rec = SinFVM.LinearReconstruction(1.3)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, bst)

    # TODO: Second order timestepper
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    initial = get_initial_conditions(sw, grid, eq, backend)
    SinFVM.set_current_state!(simulator, initial)
 

    SinFVM.simulate_to_time(simulator, t)
    #SinFVM.simulate_to_time(simulator, 0.000001)
    @show SinFVM.current_timestep(simulator)

    plot_simulator_state(simulator)
    # return nothing

    f = Figure(size=(1600, 1200), fontsize=24)
    x = SinFVM.cell_centers(grid)
    x_faces = SinFVM.cell_faces(grid)
    topo_faces = SinFVM.collect_topography_intersections(topography, grid)
    topo_cells = SinFVM.collect_topography_cells(topography, grid)
    infostring = "swashes test case $(sw.id)
 $(sw.name)
t=$(t) nx=$(nx)
$(typeof(eq))
$(typeof(rec)) 
 $(typeof(flux))"
    ax_h = Axis(f[1, 1], title="h", ylabel="h", xlabel="x")
    ax_hu = Axis(f[1, 2], title="hu", ylabel="hu", xlabel="x")
    ax_hl = Axis(f[2, 1], title="h", ylabel="h", xlabel="x")
    ax_hr = Axis(f[2, 2], title="h", ylabel="h", xlabel="x")

    supertitle = Label(f[0, :], infostring)

    ref_sol = SinFVM.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    lines!(ax_h, x_faces, topo_faces, label="B(x)", color="black")
    lines!(ax_h, x, collect(ref_sol.h), label="swashes")
    lines!(ax_hu, x, collect(ref_sol.hu), label="swashes")

    
    our_sol = SinFVM.current_interior_state(simulator)
    hu = collect(our_sol.hu)
    w  = collect(our_sol.h)
    h = w - topo_cells
    u = SinFVM.desingularize.(Ref(eq), h, hu)
    lines!(ax_h, x, w, label="swamp")
    lines!(ax_hu, x, hu, label="swamp")

    lines!(ax_hl, x_faces[xl_start:xl_end+1], topo_faces[xl_start:xl_end+1], label="B(x)", color="black")
    lines!(ax_hl, x[xl_start:xl_end+1], collect(ref_sol.h)[xl_start:xl_end+1], label="swashes")
    lines!(ax_hl, x[xl_start:xl_end+1], w[xl_start:xl_end+1], label="swamp")
    lines!(ax_hr, x_faces[xr_start:xr_end+1], topo_faces[xr_start:xr_end+1], label="B(x)", color="black")
    lines!(ax_hr, x[xr_start:xr_end+1], collect(ref_sol.h)[xr_start:xr_end+1], label="swashes")
    lines!(ax_hr, x[xr_start:xr_end+1], w[xr_start:xr_end+1], label="swamp")

    axislegend(ax_h)
    axislegend(ax_hr)
    axislegend(ax_hl)
    axislegend(ax_hu)
    display(f)
end
    
function compare_swashes_in2d(sw::Swashes41x, nx, t; 
                              do_plot=true, do_test=true, 
                              backend=make_cpu_backend(), timestepper = SinFVM.ForwardEulerStepper())
    extent = getExtent(sw)
    ny = 3
    grid_x = SinFVM.CartesianGrid(nx, ny; gc=2, boundary=SinFVM.WallBC(), extent=[extent[1] extent[2]; 0.0 5], )
    grid_y = SinFVM.CartesianGrid(ny, nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 5; extent[1] extent[2]], )
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=getExtent(sw), )
    eq_2D = SinFVM.ShallowWaterEquations(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    eq_1D = SinFVM.ShallowWaterEquations1D(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    rec = SinFVM.LinearReconstruction(1.2)
    flux_2D = SinFVM.CentralUpwind(eq_2D)
    flux_1D = SinFVM.CentralUpwind(eq_1D)
    
    conserved_system_x  = SinFVM.ConservedSystem(backend, rec, flux_2D, eq_2D, grid_x)
    conserved_system_y  = SinFVM.ConservedSystem(backend, rec, flux_2D, eq_2D, grid_y)
    conserved_system_1D = SinFVM.ConservedSystem(backend, rec, flux_1D, eq_1D, grid)
    # TODO: Second order timestepper
    simulator_x  = SinFVM.Simulator(backend, conserved_system_x,  timestepper, grid_x, cfl=0.1)
    simulator_y  = SinFVM.Simulator(backend, conserved_system_y,  timestepper, grid_y, cfl=0.1)
    simulator_1D = SinFVM.Simulator(backend, conserved_system_1D, timestepper, grid,   cfl=0.1)
    
    initial_x =  get_initial_conditions(sw, grid_x, eq_2D, backend; dim=2, dir=1)
    initial_y =  get_initial_conditions(sw, grid_y, eq_2D, backend; dim=2, dir=2)
    initial_1D = get_initial_conditions(sw, grid, eq_1D, backend; dim=1, dir=1)

    SinFVM.set_current_state!(simulator_x, initial_x)
    SinFVM.set_current_state!(simulator_y, initial_y)
    SinFVM.set_current_state!(simulator_1D, initial_1D)
   
    println("2D x-direction,  grid: [512, 3]")
    @time SinFVM.simulate_to_time(simulator_x, t, show_progress=true)
    println("2D y-direction,  grid: [3, 512]")
    @time SinFVM.simulate_to_time(simulator_y, t, show_progress=true)
    println("1D med 512 celler")
    @time SinFVM.simulate_to_time(simulator_1D, t, show_progress=true)
    #SinFVM.simulate_to_time(simulator, 0.000001)

    
    ref_sol = SinFVM.InteriorVolume(get_reference_solution(sw, grid, t, eq_1D, backend))
    
    our_sol_1D = SinFVM.current_interior_state(simulator_1D)
    h_1D = collect(our_sol_1D.h)
    hu_1D = collect(our_sol_1D.hu)
    u_1D = SinFVM.desingularize.(Ref(eq_1D), h_1D, hu_1D)

    our_sol_x = SinFVM.current_interior_state(simulator_x)
    h_x  = collect(our_sol_x.h)[:, 2]
    hu_x = collect(our_sol_x.hu)[:, 2]
    u_x = SinFVM.desingularize.(Ref(eq_2D), h_x, hu_x)
    hv_x = collect(our_sol_x.hv)
    
    our_sol_y = SinFVM.current_interior_state(simulator_y)
    hv_y = collect(our_sol_y.hv)[2, :]
    h_y  = collect(our_sol_y.h)[2, :]
    v_y = SinFVM.desingularize.(Ref(eq_2D), h_y, hv_y)
    hu_y = collect(our_sol_y.hu)
    
    if do_plot
        f = Figure(size=(1600, 600), fontsize=24)
        x = SinFVM.cell_centers(grid)
        infostring = "2D swashes test case $(sw.id)
 $(sw.name)
t=$(t) nx=$(nx)
$typeof(eq)
$(typeof(rec)) 
 $(typeof(flux_2D))"
        ax_h = Axis(
            f[1, 1],
            title="h ",
            ylabel="h",
            xlabel="x",
        )
    
        ax_u = Axis(
            f[1, 2],
            title="u ",
            ylabel="u",
            xlabel="x",
        )
        _ = Label(f[0, :], infostring)
    
        lines!(ax_h, x, collect(ref_sol.h), label="swashes")
        lines!(ax_u, x, collect(ref_sol.hu), label="swashes")

        lines!(ax_h, x, h_1D, label="swamp 1D")
        lines!(ax_u, x, u_1D, label="swamp 1D")

        lines!(ax_h, x, h_x, label="swamp x")
        lines!(ax_u, x, u_x, label="swamp x")
       
        lines!(ax_h, x, h_y, label="swamp y")
        lines!(ax_u, x, v_y, label="swamp y")
       
        axislegend(ax_h)
        axislegend(ax_u)
        display(f)
    end
    
    if do_test
        @test hv_x == zero(hv_x) 
        @test hu_y == zero(hu_y)

        @test v_y == u_x
        @test h_y == h_x
        @test u_1D == u_x
        @test h_1D == h_x
        
        @test simulator_x.current_timestep[1] == simulator_1D.current_timestep[1]
        @test simulator_x.current_timestep[1] == simulator_y.current_timestep[1]
    end
    @show simulator_x.current_timestep[1] 
    @show simulator_y.current_timestep[1] 
    @show simulator_1D.current_timestep[1]
    return nothing
end

# plot_ref_solution(swashes411, nx, 0:2:10)
# plot_ref_solution(swashes412, nx, 0:2:10)
# plot_ref_solution(swashes421, nx, 0:1:5)

# compare_swashes(swashes411, nx, 8.0)
# compare_swashes(swashes412, nx, 6.0)
# compare_swashes(swashes421, nx, swashes421.period*1)
# println("

#")

# compare_swashes(swashes412, nx, 2.0)
# compare_swashes_in2d(swashes412, nx, 4.0; do_plot=true, do_test=true, timestepper=SinFVM.RungeKutta2())


for backend in SinFVM.get_available_backends()
    swashes411 = Swashes411()
    swashes412 = Swashes412()
    swashes421 = Swashes421(offset=0.3)

    nx = 512
    compare_swashes_in2d(swashes411, nx, 6.0; do_plot=false, backend=backend)
    compare_swashes_in2d(swashes412, nx, 4.0; do_plot=false, backend=backend)
end
