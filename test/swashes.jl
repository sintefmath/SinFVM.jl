using SinSWE

using StaticArrays
using CairoMakie

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

function get_reference_solution(sw::Swashes41x, grid::CartesianGrid{1}, t, eq::SinSWE.ShallowWaterEquations1D, backend)
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
    #@show all_x
    #@show type(all_x)
    # h = zeros(MVector{SinSWE.inner_cells(grid, SinSWE.XDIR)})
    # u = zeros(MVector{SinSWE.inner_cells(grid, SinSWE.XDIR)})

    # for (i, x) in pairs(all_x)
    #     h[i] = get_h(sw, x)
    #     u[i] = get_u(sw, x)
    # end        
    
    ref_state = SinSWE.Volume(backend, eq, grid)
    SinSWE.InteriorVolume(ref_state)[:] = [SVector{2, Float64}(get_h(sw, x), get_u(sw, x)) for x in all_x]
    return ref_state
end


function get_initial_conditions(sw::Swashes, grid, eq::SinSWE.ShallowWaterEquations1D, backend)
    return get_reference_solution(sw, grid, 0.0, eq, backend)
end


swashes411 = Swashes411()
@show swashes411
swashes412 = Swashes412()
@show swashes412

nx = 512
grid = SinSWE.CartesianGrid(nx; gc=2, extent=getExtent(swashes411))

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
    infostring = "swashes test case $(sw.id)\n $(sw.name)\nt=$(t) nx=$(nx)\n$(typeof(rec)) and $(typeof(flux))"
    ax_h = Axis(
        f[1, 1],
        title="h "*infostring,
        ylabel="h",
        xlabel="x",
    )

    ax_u = Axis(
        f[1, 2],
        title="u "*infostring,
        ylabel="u",
        xlabel="x",
    )

    ref_sol = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t, eq, backend))
    lines!(ax_h, x, collect(ref_sol.h), label="swashes")
    lines!(ax_u, x, collect(ref_sol.hu), label="swashes")

    
    our_sol = SinSWE.current_interior_state(simulator)
    hu = collect(our_sol.hu)
    h  = collect(our_sol.h)
    u = hu./h
    lines!(ax_h, x, h, label="swamp")
    lines!(ax_u, x, u, label="swamp")

    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
end
    

# plot_ref_solution(swashes411, grid, 0:2:10)
# plot_ref_solution(swashes412, grid, 0:2:10)

compare_swashes(swashes411, 512, 8.0)
compare_swashes(swashes412, 512, 6.0)

