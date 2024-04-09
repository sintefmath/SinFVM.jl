using SinSWE

using StaticArrays
using CairoMakie
using Test
import CUDA

# Implements analytic solutions from:
# O. Delestre et al., SWASHES: a compilation of shallow water analytic 
# for hydraulic and environmental studies. 
# International Journal for Numerical Methods in Fluids, 72(3):269–300, 2013.
# DOI: https://doi.org/10.1002/fld.3741


abstract type Swashes end
abstract type Swashes1D <:Swashes end
abstract type Swashes41x <: Swashes1D end

# Other useful tests:
# struct Swashes421 <: Swashes1D end
# abstract struct Swashes2D <: Swashes end
# struct Swashes422 <: Swashes2D end


struct Swashes411 <: Swashes41x
    x0::Float64
    L::Float64
    g::Float64
    hr::Float64
    hl::Float64
    cm::Float64
    id::String
    name::String
    Swashes411(;x0=5.0, L=10.0, g=9.81, 
                hr=0.001, hl=0.005, cm=0.1578324867,
                id="4.1.1", name="Dam break on wet domain without friction"
                ) =  new(x0, L, g, hr, hl, cm, id, name)
end
struct Swashes412 <: Swashes41x
    x0::Float64
    L::Float64
    g::Float64
    hr::Float64
    hl::Float64
    cm::Float64
    id::String
    name::String
    Swashes412(;x0=5.0, L=10.0, g=9.81, 
                hr=0.0, hl=0.005, cm=0.0,
                id="4.1.2", name="Dam break on dry domain without friction"
                ) =  new(x0, L, g, hr, hl, cm, id, name)
end

function getExtent(sw::Swashes1D)
    return [0.0  sw.L]
end

function get_reference_solution(sw::Swashes41x, grid::CartesianGrid, t, eq::SinSWE.AllPracticalSWE, backend; dir=1, dim=1)
    xA = sw.x0 - t*sqrt(sw.g*sw.hl)
    xB = sw.x0 + t*(2*sqrt(sw.g*sw.hl) - 3*sw.cm)
    xC = sw.x0 + t*(2*sw.cm^2 *(sqrt(sw.g*sw.hl) - sw.cm))/(sw.cm^2 - sw.g*sw.hr)
    function tmp_h_rarefaction(t, x)
        h =(4.0/(9.0*sw.g))* (sqrt(sw.g*sw.hl) - (x - sw.x0)/(2*t))^2
        u = (2.0/3.0)*((x-sw.x0)/t + sqrt(sw.g*sw.hl))
        return h, u
    end   
    rarefaction_h(x) = (4.0/(9.0*sw.g))* (sqrt(sw.g*sw.hl) - (x - sw.x0)/(2*t))^2
    rarefaction_u(x) = (2.0/3.0)*((x-sw.x0)/t + sqrt(sw.g*sw.hl))
    function get_h(::Swashes411, x)
        x < xA && return sw.hl
        x < xB && return rarefaction_h(x)
        x < xC && return sw.cm^2/sw.g
        return sw.hr
    end
    function get_u(::Swashes411, x)
        x < xA && return 0.0
        x < xB && return rarefaction_u(x)
        x < xC && return 2*(sqrt(sw.g*sw.hl) - sw.cm)
        return 0.0
    end
    function get_h(::Swashes412, x)
        x < xA && return sw.hl
        x < xB && return rarefaction_h(x)
        return sw.hr
    end
    function get_u(::Swashes412, x)
        x < xA && return 0.0
        x < xB && return rarefaction_u(x)
        return 0.0
    end

    all_x = SinSWE.cell_centers(grid)
    if (dir == 2) @assert dim == 2 end
    
    ref_state = SinSWE.Volume(backend, eq, grid)
    if dir == 1 && dim == 1
        CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:] = [SVector{2, Float64}(get_h(sw, x), get_u(sw, x)) for x in all_x]
        return ref_state
    elseif dir == 1 && dim == 2
        u0 = x ->  @SVector[get_h(sw, x[1]), get_u(sw, x[1]), 0.0]
        return u0.(all_x)    
        # tmp =  [SVector{3, Float64}(get_h(sw, x[1]), get_u(sw, x[1]), 0.0) for x in all_x]
        # CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:, :] = tmp
        # return ref_state
    elseif dir == 2 && dim == 2
        u0 = x ->  @SVector[get_h(sw, x[2]), 0.0, get_u(sw, x[2])]
        return u0.(all_x)    
    end
end


function get_initial_conditions(sw::Swashes, grid, eq::SinSWE.AllPracticalSWE, backend; dir=1, dim=1)
    return get_reference_solution(sw, grid, 0.0, eq, backend; dir=dir, dim=dim)
end


swashes411 = Swashes411()
@show swashes411
swashes412 = Swashes412()
@show swashes412


function plot_ref_solution(sw::Swashes1D, grid, T)
    f = Figure(size=(1600, 600), fontsize=24)
    x = SinSWE.cell_centers(grid)
    ax_h = Axis(
        f[1, 1],
        title="h in swashes test case $(sw.id)\n $(sw.name) cells.\nT=$(T)",
        ylabel="h",
        xlabel="x",
    )

    ax_u = Axis(
        f[1, 2],
        title="u in swashes test case $(sw.id)\n $(sw.name) cells.\nT=$(T)",
        ylabel="u",
        xlabel="x",
    )
    for t in T
        ref_state = get_reference_solution(sw, grid, t)
        lines!(ax_h, x, ref_state.h, label="t=$(t)")
        lines!(ax_u, x, ref_state.hu, label="t=$(t)")
    end
    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
end



function compare_swashes(sw::Swashes41x, nx, t)
    grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=getExtent(sw), )
    backend = make_cpu_backend()
    eq = SinSWE.ShallowWaterEquations1D(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    rec = SinSWE.LinearReconstruction(1.2)
    flux = SinSWE.CentralUpwind(eq)
    conserved_system = SinSWE.ConservedSystem(backend, rec, flux, eq, grid)
    # TODO: Second order timestepper
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    initial = get_initial_conditions(sw, grid, eq, backend)
    SinSWE.set_current_state!(simulator, initial)
 
    SinSWE.simulate_to_time(simulator, t)
    #SinSWE.simulate_to_time(simulator, 0.000001)

    f = Figure(size=(1600, 600), fontsize=24)
    x = SinSWE.cell_centers(grid)
    infostring = "swashes test case $(sw.id)\n $(sw.name)\nt=$(t) nx=$(nx)\n$typeof(eq)\n$(typeof(rec)) \n $(typeof(flux))"
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



    ref_sol = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    lines!(ax_h, x, collect(ref_sol.h), label="swashes")
    lines!(ax_u, x, collect(ref_sol.hu), label="swashes")

    
    our_sol = SinSWE.current_interior_state(simulator)
    hu = collect(our_sol.hu)
    h  = collect(our_sol.h)
    u = SinSWE.desingularize.(Ref(eq), h, hu)
    lines!(ax_h, x, h, label="swamp")
    lines!(ax_u, x, u, label="swamp")

    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
end
    
function compare_swashes_in2d(sw::Swashes41x, nx, t; 
                              do_plot=true, do_test=true, 
                              backend=make_cpu_backend(), timestepper = SinSWE.ForwardEulerStepper())
    extent = getExtent(sw)
    ny = 3
    grid_x = SinSWE.CartesianGrid(nx, ny; gc=2, boundary=SinSWE.WallBC(), extent=[extent[1] extent[2]; 0.0 5], )
    grid_y = SinSWE.CartesianGrid(ny, nx; gc=2, boundary=SinSWE.WallBC(), extent=[0.0 5; extent[1] extent[2]], )
    grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=getExtent(sw), )
    eq_2D = SinSWE.ShallowWaterEquations(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    eq_1D = SinSWE.ShallowWaterEquations1D(;depth_cutoff=10^-7, desingularizing_kappa=10^-7)
    rec = SinSWE.LinearReconstruction(1.2)
    flux_2D = SinSWE.CentralUpwind(eq_2D)
    flux_1D = SinSWE.CentralUpwind(eq_1D)
    
    conserved_system_x  = SinSWE.ConservedSystem(backend, rec, flux_2D, eq_2D, grid_x)
    conserved_system_y  = SinSWE.ConservedSystem(backend, rec, flux_2D, eq_2D, grid_y)
    conserved_system_1D = SinSWE.ConservedSystem(backend, rec, flux_1D, eq_1D, grid)
    # TODO: Second order timestepper
    simulator_x  = SinSWE.Simulator(backend, conserved_system_x,  timestepper, grid_x, cfl=0.1)
    simulator_y  = SinSWE.Simulator(backend, conserved_system_y,  timestepper, grid_y, cfl=0.1)
    simulator_1D = SinSWE.Simulator(backend, conserved_system_1D, timestepper, grid,   cfl=0.1)
    
    initial_x =  get_initial_conditions(sw, grid_x, eq_2D, backend; dim=2, dir=1)
    initial_y =  get_initial_conditions(sw, grid_y, eq_2D, backend; dim=2, dir=2)
    initial_1D = get_initial_conditions(sw, grid, eq_1D, backend; dim=1, dir=1)

    SinSWE.set_current_state!(simulator_x, initial_x)
    SinSWE.set_current_state!(simulator_y, initial_y)
    SinSWE.set_current_state!(simulator_1D, initial_1D)
   
    println("2D x-direction,  grid: [512, 3]")
    @time SinSWE.simulate_to_time(simulator_x, t, show_progress=false)
    println("2D y-direction,  grid: [3, 512]")
    @time SinSWE.simulate_to_time(simulator_y, t, show_progress=false)
    println("1D med 512 celler")
    @time SinSWE.simulate_to_time(simulator_1D, t, show_progress=false)
    #SinSWE.simulate_to_time(simulator, 0.000001)

    
    ref_sol = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t, eq_1D, backend))
    
    our_sol_1D = SinSWE.current_interior_state(simulator_1D)
    h_1D = collect(our_sol_1D.h)
    hu_1D = collect(our_sol_1D.hu)
    u_1D = SinSWE.desingularize.(Ref(eq_1D), h_1D, hu_1D)

    our_sol_x = SinSWE.current_interior_state(simulator_x)
    h_x  = collect(our_sol_x.h)[:, 2]
    hu_x = collect(our_sol_x.hu)[:, 2]
    u_x = SinSWE.desingularize.(Ref(eq_2D), h_x, hu_x)
    hv_x = collect(our_sol_x.hv)
    
    our_sol_y = SinSWE.current_interior_state(simulator_y)
    hv_y = collect(our_sol_y.hv)[2, :]
    h_y  = collect(our_sol_y.h)[2, :]
    v_y = SinSWE.desingularize.(Ref(eq_2D), h_y, hv_y)
    hu_y = collect(our_sol_y.hu)
    
    if do_plot
        f = Figure(size=(1600, 600), fontsize=24)
        x = SinSWE.cell_centers(grid)
        infostring = "2D swashes test case $(sw.id)\n $(sw.name)\nt=$(t) nx=$(nx)\n$typeof(eq)\n$(typeof(rec)) \n $(typeof(flux_2D))"
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
    return nothing
end

# plot_ref_solution(swashes411, grid, 0:2:10)
# plot_ref_solution(swashes412, grid, 0:2:10)

#compare_swashes(swashes411, 512, 8.0)
# compare_swashes(swashes412, 512, 6.0)
println("\n\n")

# compare_swashes(swashes412, 512, 2.0)
#compare_swashes_in2d(swashes412, 512, 4.0; do_plot=true, do_test=true)


for backend in SinSWE.get_available_backends()
    compare_swashes_in2d(swashes411, 512, 6.0; do_plot=false, backend=backend)
    compare_swashes_in2d(swashes412, 512, 4.0; do_plot=false, backend=backend)
end

