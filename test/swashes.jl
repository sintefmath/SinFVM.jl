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
abstract type Swashes2D <: Swashes end
abstract type Swashes422x <: Swashes2D end


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

struct Swashes413 <: Swashes1D
x0::Float64
L::Float64
g::Float64
hl::Float64
C::Float64
id::String
name::String
Swashes413(;x0=5.0, L=10.0, g=9.81, hl=0.005, C=0.0,
            id="4.1.3", name="Dressler's dam break with friction"
            ) =  new(x0, L, g, hl, C, id, name)
end

struct Swashes421 <: Swashes1D
    a::Float64
    L::Float64
    g::Float64
    h0::Float64
    period::Float64
    offset::Float64
    id::String
    name::String
    Swashes421(;a=1.0, L=4.0, g=9.81, h0=0.5, period=2.00606, offset=0.0,
                id="4.2.1", name="Planar surface in a parabola without friction"
    ) = new(a, L, g, h0, period, offset, id, name)
end


struct Swashes422a <: Swashes422x
    a::Float64
    L::Float64
    g::Float64
    h0::Float64
    r0::Float64
    ω::Float64
    period::Float64
    A::Float64
    offset::Float64
    id::String
    name::String
    function Swashes422a(;a=1.0, L=4.0, g=9.81, h0=0.1, r0=0.8, offset=0.0,
                id="4.2.2.a", name="2D radially-symmetrical paraboloid")
        ω = sqrt(8*g*h0)/a
        period = 3*2*pi/ω
        A = (a^2 - r0^2)/(a^2 + r0^2)
        return new(a, L, g, h0, r0, ω, period, A, offset, id, name)
    end
end

struct Swashes422b <: Swashes422x
    a::Float64
    L::Float64
    g::Float64
    h0::Float64
    η::Float64
    ω::Float64
    period::Float64
    offset::Float64
    id::String
    name::String
    function Swashes422b(;a=1.0, L=4.0, g=9.81, h0=0.1, η=0.5, offset=0.0,
                id="4.2.2.b", name="2D planar surface in a paraboloid")
        omega = sqrt(2*g*h0)/a
        period = 3*2*π/omega
        return new(a, L, g, h0, η, omega, period, offset, id, name)
    end
end

getExtent(sw::Swashes1D) = [0.0  sw.L]
getExtent(sw::Swashes2D) = [0.0  sw.L; 0.0 sw.L]

function get_reference_solution(sw::Swashes41x, grid::CartesianGrid, t, eq::SinSWE.AllPracticalSWE=SinSWE.ShallowWaterEquations1D(), backend=SinSWE.make_cpu_backend(); dir=1, dim=1)
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



function get_reference_solution(sw::Swashes413, grid::CartesianGrid, t, eq::SinSWE.AllPracticalSWE=SinSWE.ShallowWaterEquations1D(), backend=SinSWE.make_cpu_backend(); dir=1, dim=1)
    
    function alpha_1(x) 
        p = 2 - ((x - sw.x0)/t*sqrt(sw.g*sw.hl))
        return (6/(5*p)) - 2/3 + (4*sqrt(3)/135)*sqrt(p)^3
    end
    function alpha_2(x) 
        p = 2 - ((x - sw.x0)/t*sqrt(sw.g*sw.hl))
        return 12/p - 8/3 + (8*sqrt(3)/189)*sqrt(p)^3 - 108/(7*p^2)
    end
    function h_co(x)
        return ((2*sqrt(sw.g*sw.hl)/3)  - ((x-sw.x0)/(3*t)) + g^2*alpha_1(x)*t/sw.C^2 )^2/sw.g    
    end
    function u_co(x)
        return (2*sqrt(sw.g*sw.hl)/3)  - (2*(x-sw.x0)/(3*t)) + g^2*alpha_2(x)*t/sw.C^2
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




function get_reference_solution(sw::Swashes421, grid::CartesianGrid, t, eq::SinSWE.AllPracticalSWE=SinSWE.ShallowWaterEquations1D(), backend=SinSWE.make_cpu_backend(); momentum=true, dir=1, dim=1)
    B = sqrt(2.0*sw.g*sw.h0)/(2.0*sw.a)
    x1 = -0.5*cos(2*B*t) - sw.a + sw.L/2.0
    x2 = -0.5*cos(2*B*t) + sw.a + sw.L/2.0
    dx = SinSWE.compute_dx(grid)
    function get_h(x)
        if x < x1 || x > x2
            return 0.0
        end
        term1 = (1.0/sw.a)*(x - sw.L/2.0)
        term2 = (1.0/(2.0*sw.a))*cos(2.0*B*t)
        return -sw.h0*((term1 + term2)^2 - 1)
    end
    function get_w(x)
        B_left  = b_value(sw, x - 0.5*dx)
        B_right = b_value(sw, x + 0.5*dx)
        B_center = 0.5*(B_left + B_right)
        # return get_h(x) + B_center
        w = max(get_h(x) + B_center, B_center)
        return w
    end
    function get_u(x)
        if x < x1 || x > x2
            return 0.0
        end
        return B*sin(2.0*B*t)
    end
    function get_hu(x)
        return get_u(x)*get_h(x)
    end

    all_x = SinSWE.cell_centers(grid)  
    ref_state = SinSWE.Volume(backend, eq, grid)
    if momentum
        CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:] = [SVector{2, Float64}(get_w(x), get_hu(x)) for x in all_x]
    else
        CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:] = [SVector{2, Float64}(get_w(x), get_u(x)) for x in all_x]
    end
    return ref_state
end


function get_reference_solution(sw::Swashes422a, grid::CartesianGrid, t, eq::ShallowWaterEquations=SinSWE.ShallowWaterEquations2D(), backend=SinSWE.make_cpu_backend())
    h_term1 = (sqrt(1 - sw.A^2)/(1 - sw.A*cos(sw.ω*t))) - 1.0
    h_term2 = ((1 - sw.A^2)/(1 - sw.A*cos(sw.ω*t))^2) - 1.0
    velocity_term = (1.0/(1 - sw.A*cos(sw.ω*t)))*(0.5*sw.ω*sw.A*sin(sw.ω*t))
    dx = SinSWE.compute_dx(grid)
    dy = SinSWE.compute_dx(grid)
    function get_B_cell(x, y)
        return 0.25*(b_value(sw, sw_r(sw, x - 0.5*dx, y - 0.5*dy)) +  
                     b_value(sw, sw_r(sw, x - 0.5*dx, y + 0.5*dy)) +
                     b_value(sw, sw_r(sw, x + 0.5*dx, y - 0.5*dy)) +  
                     b_value(sw, sw_r(sw, x + 0.5*dx, y + 0.5*dy)))
    end
    function get_w(x, y)
        r = sw_r(sw, x, y)
        w = sw.h0*(h_term1 - (r^2/sw.a^2)*h_term2)
        return max(w, get_B_cell(x, y))
    end
    function get_h(x, y)
        return get_w(x, y) - get_B_cell(x, y)        
    end
    function get_hu(x, y)
        return velocity_term*(x - sw.L/2.0)*get_h(x, y)
    end
    function get_hv(x, y)
        return velocity_term*(y - sw.L/2.0)*get_h(x, y)
    end
    all_x = SinSWE.cell_centers(grid)
    ref_state = SinSWE.Volume(backend, eq, grid)
    CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:] = [SVector{3, Float64}(get_w(x[1], x[2]), get_hu(x[1], x[2]), get_hv(x[1], x[2])) for x in all_x]
    return ref_state   
end

function get_reference_solution(sw::Swashes422b, grid::CartesianGrid, t, eq::ShallowWaterEquations=SinSWE.ShallowWaterEquations2D(), backend=SinSWE.make_cpu_backend())
    cos_term = cos(sw.ω*t)
    sin_term = sin(sw.ω*t)
    u = -sw.η*sw.ω*sin_term
    v =  sw.η*sw.ω*cos_term
    dx = SinSWE.compute_dx(grid)
    dy = SinSWE.compute_dx(grid)
    function get_B_cell(x, y)
        return 0.25*(b_value(sw, sw_r(sw, x - 0.5*dx, y - 0.5*dy)) +  
                     b_value(sw, sw_r(sw, x - 0.5*dx, y + 0.5*dy)) +
                     b_value(sw, sw_r(sw, x + 0.5*dx, y - 0.5*dy)) +  
                     b_value(sw, sw_r(sw, x + 0.5*dx, y + 0.5*dy)))
    end
    function get_w(x, y)
        w = (sw.η*sw.h0/sw.a^2)*(2*(x - sw.L/2.0)*cos_term + 2*(y - sw.L/2.0)*sin_term - sw.η)
        return max(w, get_B_cell(x, y))
    end
    function get_h(x, y)
        return get_w(x, y) - get_B_cell(x, y)        
    end
    function get_hu(x, y)
        return u*get_h(x, y)
    end
    function get_hv(x, y)
        return v*get_h(x, y)
    end
    all_x = SinSWE.cell_centers(grid)
    ref_state = SinSWE.Volume(backend, eq, grid)
    CUDA.@allowscalar SinSWE.InteriorVolume(ref_state)[:] = [SVector{3, Float64}(get_w(x[1], x[2]), get_hu(x[1], x[2]), get_hv(x[1], x[2])) for x in all_x]
    return ref_state   
end


sw_r(sw::Swashes422x, x, y) = sqrt((x - sw.L/2)^2 + (y - sw.L/2)^2)

function get_bottom_topography(::Swashes, ::CartesianGrid, backend)
    return SinSWE.ConstantBottomTopography()
end

b_value(sw::Swashes421, x) = sw.h0*((1/sw.a^2)*(x - sw.L/2.0)^2 - 1.0) + sw.offset
function get_bottom_topography(sw::Swashes421, grid::CartesianGrid, backend)
    # b = x -> sw.h0*((1/sw.a^2)*(x - sw.L/2.0)^2 - 1.0) + sw.offset
    B_data = [b_value(sw, x) for x in SinSWE.cell_faces(grid, interior=false)]
    return SinSWE.BottomTopography1D(B_data, backend, grid)
end

b_value(sw::Swashes422x, r) = -sw.h0*(1 - (r^2/sw.a^2))
function get_bottom_topography(sw::Swashes422x, grid::CartesianGrid{2}, backend)
    # b = (r) -> -sw.h0*(1 - (r^2/sw.a^2))
    B_data = [b_value(sw, sw_r(sw, x[1], x[2])) for x in SinSWE.cell_faces(grid, interior=false)]
    return SinSWE.BottomTopography2D(B_data, backend, grid)
end   


function get_initial_conditions(sw::Swashes1D, grid, eq::SinSWE.AllPracticalSWE, backend; dir=1, dim=1)
    return get_reference_solution(sw, grid, 0.0, eq, backend; dir=dir, dim=dim)
end
function get_initial_conditions(sw::Swashes2D, grid, eq::SinSWE.AllPracticalSWE, backend)
    return get_reference_solution(sw, grid, 0.0, eq, backend)
end

function plot_ref_solution(sw::Swashes1D, nx, T)
    f = Figure(size=(1600, 600), fontsize=24)
    grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=getExtent(sw), )
    x = SinSWE.cell_centers(grid, interior=false)
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

function plot_ref_solution(sw::Swashes421, nx, T)
    f = Figure(size=(1600, 600), fontsize=24)
    grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=getExtent(sw), )
    backend = SinSWE.make_cpu_backend()
    topography = get_bottom_topography(sw, grid, backend)
    x = SinSWE.cell_centers(grid)
    x_faces = SinSWE.cell_faces(grid)
    bottom_faces = SinSWE.collect_topography_intersections(topography, grid)
    @show size(x)
    @show size(x_faces)
    @show typeof(grid) <: SinSWE.CartesianGrid{2}
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
    lines!(ax_h, x_faces, bottom_faces, label="B(x)", color="black")
    for t in T
        ref_state = SinSWE.InteriorVolume(get_reference_solution(sw, grid, t))
        lines!(ax_h, x, ref_state.h, label="t=$(t)")
        lines!(ax_u, x, ref_state.hu, label="t=$(t)")
    end
    axislegend(ax_h)
    axislegend(ax_u)
    display(f)
end

# plot_ref_solution(Swashes421(), 64, 0:0.5:2)
