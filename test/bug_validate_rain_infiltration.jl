using StaticArrays
using LinearAlgebra
using Test
using CairoMakie
import CUDA

using SinFVM

# Validation of friction, rain and infiltration
# Using the results from the synthetic test cases from
# Fernandez-Pato, Caviedes-Voullieme, Garcia-Navarro (2016) 
# Rainfall/runoff simulation with 2D full shallow water equations: Sensitivity analysis and calibration of infiltration parameters.
# Journal of Hydrology, 536, 496-513. https://doi.org/10.1016/j.jhydrol.2016.03.021

abstract type ValidationTopography end
struct Topography1 <: ValidationTopography
    Lx
    Ly
    Topography1() = new(2000, 20)
end
struct Topography1Debug <: ValidationTopography
    Lx
    Ly
    Lx_start
    Lx_end
    Topography1Debug() = new(2000, 20, 1990, 2015)
end
struct Topography2 <: ValidationTopography
    Lx
    Ly
    x0::Float64
    Topography2(x0=100) = new(2000, 20, x0)
end
struct Topography3 <: ValidationTopography 
    Lx
    Ly
    x0::Float64
    Topography3(x0=200) = new(200, 20, x0) 
end

function make_grid(topo::ValidationTopography; dx=1.0, dy=5.0)
    Lx = topo.Lx*2
    Ly = topo.Ly
    Nx = Int32(Lx/dx)
    Ny = Int32(Ly/dy)
    return SinFVM.CartesianGrid(Nx, Ny;  boundary=SinFVM.WallBC(), gc=2, extent=[0 Lx; 0 Ly])
end

function make_grid(topo::Topography1Debug; dx=1.0, dy=5.0)
    Ly = topo.Ly
    Nx = Int32((topo.Lx_end-topo.Lx_start)/dx)
    Ny = Int32(Ly/dy)
    return SinFVM.CartesianGrid(Nx, Ny;  boundary=SinFVM.WallBC(), gc=2, extent=[topo.Lx_start topo.Lx_end; 0 Ly])
end

function make_topography(::Topography1, coord)
    x = coord[1]
    if x < 2000
        return 10 - (10.0/2000.0)*abs(x)
    end
    if x > 4000
        x = 4000 - (x - 4000)
        return -(10/2000.0)*(x - 2000)
    end
    return -(10/2000.0)*(x-2000)
end
function make_topography(topo::Topography1Debug, coord)
    x = coord[1]
    # if x < 2000
    #     return 10 - (10.0/2000.0)*abs(x)
    # end
    if x < topo.Lx_start
        x = topo.Lx_start + (topo.Lx_start - x)
    end
    if x > topo.Lx_end
        x = topo.Lx_end - (x - topo.Lx_end)
        # return -(10/2000.0)*(x - 2000)
    end
    return -(10/2000.0)*(x-2000)
end


function make_topography(topo::Topography2, coord)
    B = make_topography(Topography1, coord)
    x0 = topo.x0
    x = coord[1]
    if x >= x0 && x <= x0 + 5
        B = B - (x - x0)
    elseif x >= x0 + 5 && x <= x0 + 15
        B = B - 5
    elseif x >= x0 + 15 && x <= x0 + 20
        B = B - (x0 + 20 - x)
    end
    return B
end

function make_topography(topo::Topography3, coord)
    x = coord[1]
    x0 = topo.x0
    if x <= x0
        return 21.0 - 1.0*sin(π*abs(x)/10.0) - 0.005*abs(x) - 20.0
    else
        b0 = 21.0 - 1.0*sin(π*x0/10.0) - 0.005*x0
        if x > x0*2
            x = x0*2 - (x - x0*2)
        end
        return b0 - 0.02*(x - x0) - 20.0
    end
end

function make_topography(topo::ValidationTopography, grid::SinFVM.Grid, backend)
    B_data = [make_topography(topo, coord) for coord in SinFVM.cell_faces(grid, interior=false)]
    return SinFVM.BottomTopography2D(B_data, backend, grid)
end

function make_initial_conditions(grid, bottom_topography, init_shock; add_to_zero=0.0)
    function u_zero(x, B)
        return @SVector[B + add_to_zero, 0.0, 0.0]
    end
    function u_shock(x, B)
        w0 = B
        if x[1] < 500
            w0 = max(10.0, B)
        end
        return @SVector[w0, 0.0, 0.0]
    end
    if init_shock
        initial = [u_shock(x, B) for (x, B) in zip(SinFVM.cell_centers(grid; interior=true), 
                                                   SinFVM.collect_topography_cells(bottom_topography, grid, interior=true))]
    else
        initial = [u_zero(x, B) for (x, B) in zip(SinFVM.cell_centers(grid; interior=true), 
                                                   SinFVM.collect_topography_cells(bottom_topography, grid, interior=true))]
    end
    return initial
end

function make_initial_water(grid, bottom_topography; add_to_zero=0.0, stop_adding_at=2000)

    function u_water(x, B)
        w0 = B
        if x[1] <= stop_adding_at
            w0 = B + add_to_zero
        end
        return @SVector[w0, 0.0, 0.0]
    end
    initial = [u_water(x, B) for (x, B) in zip(SinFVM.cell_centers(grid; interior=true), 
                                            SinFVM.collect_topography_cells(bottom_topography, grid, interior=true))]
    return initial
end

function make_infiltration(topo::ValidationTopography, grid::SinFVM.Grid, backend)
    
    function infiltration_factor(Lx, coord)
        inf_factor = 0.0
        if coord[1] < Lx
            inf_factor= 1.0
        end
        return inf_factor
    end
    factor = [infiltration_factor(topo.Lx, coord) for coord in SinFVM.cell_centers(grid; interior=false)]
    return SinFVM.HortonInfiltration(grid, backend; factor=factor)
end


#############################################
# Rain source terms
#############################################

function zero_rain(t, x, y)

    return 0.0
end

lots_of_rain(t, x, y) = 10000000.0

function rain_fcg_1_1(t, x, y)
    
    # if x < 2000 && t < 250.0*60.0
    #     return 0.000125
    # end
    rain = 0.0
    if x < 2010 && t < 250.0*60.0
        rain = 0.000125*2*100
    end
    # @show rain, x, y
    return rain
end

function rain_fcg_1_2(t, x, y)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.00025
        elseif t < 2*rain_step
            return 0.000125
        elseif t < 3*rain_step
            return 7.4755e-5
        elseif t < 4*rain_step
            return 4.9020e-5
        elseif t < 5*rain_step
            return 7.4755e-5
        elseif t < 6*rain_step
            return 0.0001765
        end
    end
    return 0.0
end

function rain_fcg_1_3(t, x, y)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.0001867
        elseif t < 2*rain_step
            return 9.3341e-5
        elseif t < 3*rain_step
            return 5.6496e-5
        elseif t < 4*rain_step
            return 3.6845e-5
        elseif t < 5*rain_step
            return 5.6496e-5
        elseif t < 6*rain_step
            return 0.0001326
        end
    end
    return 0.0
end

function rain_fcg_1_4(t, x, y)
    
    if x < 2000 && t < 50.0*60.0
        #return 0.000291667
        return 0.000125
    end
    return 0.0
end

function rain_fcg_1_5(t, x, y)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.0001873
        elseif t < 2*
            rain_step
            return 3.6963e-5
        elseif t < 3*rain_step
            return 1.7249e-5
        elseif t < 4*rain_step
            return 0.000000
        elseif t < 5*rain_step
            return 5.5444e-5
        elseif t < 6*rain_step
            return 0.0001331
        end
    end
    return 0.0
end

function rain_fcg_3(t, x, y)
    if x < 200 && t < 125.0*60.0
        # 0.25 mm/s
        return 0.00025
    end
    return 0.0
end

## Run case

function setup_case(backend, topo::ValidationTopography, rain_function; init_shock=false)
    
    grid = make_grid(topo, dx=1, dy=20)
    bottom_topography = make_topography(topo, grid, backend)
    infiltration = make_infiltration(topo, grid, backend)

    equation = SinFVM.ShallowWaterEquations(bottom_topography; depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    reconstruction = SinFVM.LinearReconstruction()
    reconstruction = SinFVM.NoReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_fcg2016)
    rain = SinFVM.FunctionalRain(rain_function, grid)

    # source_terms = [SinFVM.SourceTermBottom(), rain, infiltration]
    source_terms = [SinFVM.SourceTermBottom(), infiltration, rain]
    # source_terms = [SinFVM.SourceTermBottom(), infiltration]
    # source_terms = [SinFVM.SourceTermBottom(), rain]
    # source_terms = [SinFVM.SourceTermBottom()]
    
    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms) #, friction)
    # timestepper = SinFVM.ForwardEulerStepper() #RungeKutta2()
    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)

    # initial = make_initial_conditions(grid, bottom_topography, init_shock) #, add_to_zero=0.0001)
    initial = make_initial_water(grid, bottom_topography, add_to_zero=0.0000, stop_adding_at=0000)
    SinFVM.set_current_state!(simulator, initial)

    init_state =  SinFVM.current_interior_state(simulator)

    B_cells = SinFVM.collect_topography_cells(bottom_topography, grid, interior=true)
    init_volume = sum(collect(init_state.h) - B_cells)
    @show init_volume


    SinFVM.simulate_to_time(simulator, 3 )# 100*60)

    f = Figure(size=(800, 800), fontsize=24)
    ax_h = Axis(f[1, 1], title="hei", xlabel="x", ylabel="z")
    x_cells = SinFVM.cell_centers(grid, XDIR, interior=true)
    x_faces = SinFVM.cell_faces(grid, XDIR, interior=true)
    B_faces = SinFVM.collect_topography_intersections(bottom_topography, grid, interior=true)
    
    state = SinFVM.current_interior_state(simulator)
    w_cells = collect(state.h)

    end_volume = sum(w_cells - B_cells)
    @show end_volume
    mass_loss = init_volume - end_volume 
    @show mass_loss
    

    @show(typeof(x_faces))
    @show(typeof(B_faces))
    @show size(x_faces)
    @show size(B_faces)
    lines!(ax_h, x_faces[:], B_faces[:, 1], label="bottom", color="saddlebrown")
    
    @show typeof(x_cells)
    @show typeof(w_cells)
     lines!(ax_h, x_cells[:], w_cells[:, 1], label="water level", color="deepskyblue1")
    scatter!(ax_h, x_cells[:], w_cells[:, 1], label="water level", color="deepskyblue1")
    # lines!(ax_h, x_cells[:], w_cells[:, 1].-B_cells[:, 1], label="water level", color="deepskyblue1")
    axislegend(ax_h)
    display(f)
    @show simulator.current_timestep

    @show sum(w_cells[:, 1] .- B_cells[:, 1] .> 1)
end



# Status 6. aug
# * Water builds up where the rain and infiltration stops
#   --> change which source terms to include, and where the rain stops for rain_fcg_1_1
# * question: Should dt be used when evaluating source terms such as rain and infiltration? I would guess so, right?
#   --> but if so, why does the tests pass..?



# backend = SinFVM.make_cuda_backend()
backend = SinFVM.make_cpu_backend()
topography = Topography1Debug()
setup_case(backend, topography, rain_fcg_1_1, init_shock=false)

