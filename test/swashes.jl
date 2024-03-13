using SinSWE

using StaticArrays
using CairoMakie

# Implements analytic solutions from:
# O. Delestre et al., SWASHES: a compilation of shallow water analytic 
# for hydraulic and environmental studies. 
# International Journal for Numerical Methods in Fluids, 72(3):269–300, 2013.
# DOI: https://doi.org/10.1002/fld.3741


abstract type Swashes end
abstract type Swashes1D end
abstract type Swashes41x <: Swashes1D end

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

function get_reference_solution(sw::Swashes41x, grid::CartesianGrid{1}, t)
    xA = sw.x0 - t*sqrt(sw.g*sw.hl)
    # xA = sw.x0 - t*sqrt(sw.g*sw.hl)
    xB = sw.x0 + t*(2*sqrt(sw.g*sw.hl) - 3*sw.cm)
    # xB = sw.x0 + 2*(t*sqrt(sw.g*sw.hl))
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
    h = zeros(MVector{SinSWE.inner_cells(grid, SinSWE.XDIR)})
    u = zeros(MVector{SinSWE.inner_cells(grid, SinSWE.XDIR)})
    for i = eachindex(all_x)
        h[i] = get_h(sw, all_x[i])
        u[i] = get_u(sw, all_x[i])
    end
    return h, u
end


function get_initial_conditions(sw::Swashes, grid)
    return get_reference_solution(sw, grid, 0.0)
end


swashes411 = Swashes411()
@show swashes411
swashes412 = Swashes412()
@show swashes412

nx = 512
grid = SinSWE.CartesianGrid(nx; gc=2, extent=getExtent(swashes411))

function plot_solution(sw::Swashes1D, grid, T)
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
        h, u = get_reference_solution(sw, grid, t)
        lines!(ax_h, x, h, label="t=$(t)")
        lines!(ax_u, x, u, label="t=$(t)")
    end
    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
end

#  @show get_reference_solution(swashes, grid, 4)

# a = zeros(MVector{5})
# for i = eachindex(a)
#     a[i] = i^2
# end
# @show a


plot_solution(swashes411, grid, 0:2:10)
plot_solution(swashes412, grid, 0:2:10)
