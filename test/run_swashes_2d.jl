
using SinSWE

using StaticArrays
using CairoMakie
using Test
import CUDA

include("swashes.jl")

function compare_swashes(sw::Swashes2D, t_periods, nx, ny=nx; cross_sec=0.5)
    t = sw.period * t_periods
    grid = SinSWE.CartesianGrid(nx, ny; gc=2, boundary=SinSWE.WallBC(), extent=getExtent(sw), )
    backend = make_cuda_backend()
    topography = get_bottom_topography(sw, grid, backend)
    eq = SinSWE.ShallowWaterEquations(topography; depth_cutoff=10^-4, desingularizing_kappa=10^-4)
    rec = SinSWE.LinearReconstruction(1.2)
    flux = SinSWE.CentralUpwind(eq)
    src_terms = [SinSWE.SourceTermBottom()]
    conserved_system = SinSWE.ConservedSystem(backend, rec, flux, eq, grid, src_terms)
    # TODO: Second order timestepper
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    initial = get_initial_conditions(sw, grid, eq, backend)
    SinSWE.set_current_state!(simulator, initial)

    @time SinSWE.simulate_to_time(simulator, t)
    plot_crossection(sw, simulator, t, cross_sec)
    # plot_surfaces(sw, simulator, t)
end

function plot_crossection(sw::Swashes2D, simulator, t, cross_sec)
    grid = simulator.system.grid
    eq = simulator.system.equation
    rec = simulator.system.reconstruction
    flux = simulator.system.numericalflux
    backend = simulator.backend
    
    (nx, ny) = SinSWE.interior_size(grid)
    x_index = Int(floor(cross_sec*nx))
    y_index = Int(floor(cross_sec*ny))

    f = Figure(size=(1600, 1200), fontsize=24)
    x = SinSWE.cell_centers(grid, XDIR)
    y = SinSWE.cell_centers(grid, YDIR)

    topo_cells =SinSWE.collect_topography_cells(eq.B, grid)


    infostring = "swashes test case $(sw.id)\n $(sw.name)\nt=$(t) (nx, ny)=$((nx, ny))\n$(typeof(eq))\n$(typeof(rec)) \n $(typeof(flux))"
    ax_h_x = Axis(f[1, 1], title="water level (y=$(y_index))", ylabel="z", xlabel="x")
    ax_hu_x = Axis(f[1, 2], title="hu (y=$(y_index))", ylabel="hu", xlabel="x",)
    ax_hv_x = Axis(f[1, 3], title="hv (y=$(y_index))", ylabel="hu", xlabel="x",)
    ax_h_y = Axis(f[2, 1], title="water level (x=$(x_index))", ylabel="z", xlabel="y")
    ax_hu_y = Axis(f[2, 2], title="hu (x=$(x_index))", ylabel="hu", xlabel="y",)
    ax_hv_y = Axis(f[2, 3], title="hv (x=$(x_index))", ylabel="hu", xlabel="y",)
    supertitle = Label(f[0, :], infostring)

    ref_sol = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    lines!(ax_h_x, x, topo_cells[:, y_index], color="black", label="topography")
    lines!(ax_h_x,  x, collect(ref_sol.h)[:, y_index], label="swashes")
    lines!(ax_hu_x, x, collect(ref_sol.hu)[:, y_index], label="swashes")
    lines!(ax_hv_x, x, collect(ref_sol.hv)[:, y_index], label="swashes")

    lines!(ax_h_y, x, topo_cells[x_index, :], color="black", label="topography")
    lines!(ax_h_y,  y, collect(ref_sol.h)[x_index, :], label="swashes")
    lines!(ax_hu_y, y, collect(ref_sol.hu)[x_index, :], label="swashes")
    lines!(ax_hv_y, y, collect(ref_sol.hv)[x_index, :], label="swashes")

    our_sol = SinSWE.current_interior_state(simulator)
    lines!(ax_h_x,  x, collect(our_sol.h)[:, y_index], label="SWAMP")
    lines!(ax_hu_x, x, collect(our_sol.hu)[:, y_index], label="SWAMP")
    lines!(ax_hv_x, x, collect(our_sol.hv)[:, y_index], label="SWAMP")

    lines!(ax_h_y,  y, collect(our_sol.h)[x_index, :], label="SWAMP")
    lines!(ax_hu_y, y, collect(our_sol.hu)[x_index, :], label="SWAMP")
    lines!(ax_hv_y, y, collect(our_sol.hv)[x_index, :], label="SWAMP")
    
    for ax in [ax_h_x ax_hu_x ax_hv_x ax_h_y  ax_hu_y ax_hv_y]
        axislegend(ax)
    end
    display(f)
    # plot_simulator_state(simulator)
end


function plot_surfaces(sw::Swashes2D, simulator, t)
    grid = simulator.system.grid
    eq = simulator.system.equation
    rec = simulator.system.reconstruction
    flux = simulator.system.numericalflux
    backend = simulator.backend
    
    (nx, ny) = SinSWE.interior_size(grid)
    x_index = Int(floor(nx/2))
    y_index = Int(floor(ny/2))

    f = Figure(size=(1600, 600), fontsize=24)
    x = SinSWE.cell_centers(grid, XDIR)
    y = SinSWE.cell_centers(grid, YDIR)

    topo_cells =SinSWE.collect_topography_cells(eq.B, grid)


    infostring = "swashes test case $(sw.id)\n $(sw.name)\nt=$(t) (nx, ny)=$((nx, ny))\n$(typeof(eq))\n$(typeof(rec)) \n $(typeof(flux))"
    ax_h = Axis3(f[1, 1], title="water level", ylabel="z", xlabel="x", elevation=0.2*π, azimuth=0.1*π)
    ax_hu = Axis3(f[1, 2], title="hu", ylabel="hu", xlabel="x", azimuth=0.1*π)
    ax_hv = Axis3(f[1, 3], title="hv", ylabel="hu", xlabel="x", azimuth=0.1*π)
    supertitle = Label(f[0, :], infostring)

    # ref_sol = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    surface!(ax_h, x, y, topo_cells.+0.01, colormap="algae", label="topography", alpha=0.6)
    # surface!(ax_h,  x, y, collect(ref_sol.h), label="swashes")
    # surface!(ax_hu, x, y, collect(ref_sol.hu), label="swashes")
    # surface!(ax_hv, x, y, collect(ref_sol.hv), label="swashes")

  
    our_sol = SinSWE.current_interior_state(simulator)
    surface!(ax_h, x, y, collect(our_sol.h),  colormap="devon", label="SWAMP", alpha=0.6)
    surface!(ax_hu, x, y, collect(our_sol.hu), label="SWAMP")
    surface!(ax_hv, x, y, collect(our_sol.hv), label="SWAMP")

    # for ax in [ax_h ax_hu ax_hv]
    #     axislegend(ax)
    # end
    display(f)
    # plot_simulator_state(simulator)
end


swashes422a = Swashes422a()
swashes422b = Swashes422b()

nx = 128

# compare_swashes(swashes422a, 0.4, nx; cross_sec=0.3)
compare_swashes(swashes422b, 0.05, nx, cross_sec=0.55)
