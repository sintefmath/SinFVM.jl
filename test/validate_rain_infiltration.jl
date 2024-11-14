using StaticArrays
using LinearAlgebra
using Test
using CairoMakie
import CUDA

using Dates
using NPZ

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

function make_grid(topo::ValidationTopography; dx=1.0, dy=10.0)
    Lx = topo.Lx*2
    Ly = topo.Ly
    Nx = Int32(Lx/dx)
    Ny = Int32(Ly/dy)
    return SinFVM.CartesianGrid(Nx, Ny;  boundary=SinFVM.WallBC(), gc=2, extent=[0 Lx; 0 Ly])
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
        if coord[1] < Lx
            return 1.0
        end
        return 0.0
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
    if x < 2000 && t < 250.0*60.0
        return 0.000125
    end

    return 0.0
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
        elseif t < 2*rain_step
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


function rain_fcg_1_5_2(t, x, y)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.0001873
        elseif t < 2*rain_step
            return 3.6963e-5
        elseif t < 4*rain_step
            return 1.7249e-5
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
    
    grid = make_grid(topo, dx=1)
    bottom_topography = make_topography(topo, grid, backend)
    infiltration = make_infiltration(topo, grid, backend)

    depth_cutoff = 10^-4
    equation = SinFVM.ShallowWaterEquations(bottom_topography; depth_cutoff=depth_cutoff, desingularizing_kappa=10^-4)
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
    initial = make_initial_water(grid, bottom_topography) 
    SinFVM.set_current_state!(simulator, initial)

    init_state =  SinFVM.current_interior_state(simulator)

    B_cells = SinFVM.collect_topography_cells(bottom_topography, grid, interior=true)
    init_volume = sum(collect(init_state.h) - B_cells)*SinFVM.compute_cell_size(grid)
    @show init_volume


    SinFVM.simulate_to_time(simulator, 600 )# 100*60)

    f = Figure(size=(800, 800), fontsize=24)
    ax_h = Axis(f[1, 1], title="hei", xlabel="x", ylabel="z")
    x_cells = SinFVM.cell_centers(grid, XDIR, interior=true)
    x_faces = SinFVM.cell_faces(grid, XDIR, interior=true)
    B_faces = SinFVM.collect_topography_intersections(bottom_topography, grid, interior=true)
    
    state = SinFVM.current_interior_state(simulator)
    w_cells = collect(state.h)

    end_volume = sum(w_cells - B_cells)*SinFVM.compute_cell_size(grid)
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
    # lines!(ax_h, x_cells[:], w_cells[:, 1].-B_cells[:, 1], label="water level", color="deepskyblue1")
    axislegend(ax_h)
    display(f)

    @show sum(w_cells[:, 1] .- B_cells[:, 1] .> 1)
end

#### Utility functions for validation 

mutable struct ValidationData
    runoff_volume
    h_at_1000
    rain_rate 
    infiltration_rate
    total_infiltration_rate
    t
    ValidationData() = new(Float64[], Float64[], Float64[], Float64[], Float64[], Float64[])
    function ValidationData(folder::String)
        @assert isdir(folder)
        t = npzread(joinpath(folder, "validation_t.npz"))
        runoff_volume = npzread(joinpath(folder, "validation_runoff_volume.npz"))
        h_at_1000 = npzread(joinpath(folder, "validation_h_at_1000.npz"))
        rain_rate = npzread(joinpath(folder, "validation_rain_rate.npz"))
        infiltration_rate = npzread(joinpath(folder, "validation_infiltration_rate.npz"))
        total_infiltration_rate = npzread(joinpath(folder, "validation_total_infiltration_rate.npz"))
        return new(runoff_volume, h_at_1000, rain_rate, infiltration_rate, total_infiltration_rate, t)
    end
end

function compute_runoff!(vd::ValidationData, ::Topography1, simulator, B_cells, rain, infiltration, grid; boundary_cell=2000, qoi_cell=1000)

    # Check that dx = 1
    @assert size(simulator.grid)[1] == 4004


    state = SinFVM.current_interior_state(simulator)
    t = collect(simulator.t)[1]
    h = collect(state.h) - B_cells
    runoff_volume = sum(h[boundary_cell:end, :])*SinFVM.compute_cell_size(simulator.grid)
    h_at_1000 = h[qoi_cell, 1]
    rain_rate = rain.rain_function(t, qoi_cell, 10)
    #infiltration_rate = SinFVM.compute_infiltration(infiltration, t, CartesianIndex(1002, 6))

    f = infiltration
    infiltration_rate = min(f.fc + (f.f0 - f.fc)*exp(-f.k*t), h_at_1000 + rain_rate)

    total_infiltration_rate = sum(min.(f.fc + (f.f0 - f.fc)*exp(-f.k*t), h[1:2000, :] .+ rain_rate))*SinFVM.compute_cell_size(grid)

    push!(vd.t, t)
    push!(vd.runoff_volume, runoff_volume)
    push!(vd.h_at_1000, h_at_1000)
    push!(vd.rain_rate, rain_rate)
    push!(vd.infiltration_rate, infiltration_rate)
    push!(vd.total_infiltration_rate, total_infiltration_rate)
end

function write_to_file(vd::ValidationData, topo::ValidationTopography, rain_function)
    #@show string(typeof(topo))
    #@show rain_function
    #@show string(rain_function)
    timestamp = now()
    timestamp_string = Dates.format(timestamp, "yymmdd-HHMMSS")
    #@show timestamp_string

    folder = "validation_data"
    isdir(folder) || mkdir(folder)
    folder = joinpath(folder, string(typeof(topo))*"_"*string(rain_function))
    isdir(folder) || mkdir(folder)
    folder = joinpath(folder, timestamp_string)
    isdir(folder) || mkdir(folder)

    npzwrite(joinpath(folder, "validation_t.npz"), vd.t)
    npzwrite(joinpath(folder, "validation_runoff_volume.npz"), vd.runoff_volume)
    npzwrite(joinpath(folder, "validation_h_at_1000.npz"), vd.h_at_1000)
    npzwrite(joinpath(folder, "validation_rain_rate.npz"), vd.rain_rate)
    npzwrite(joinpath(folder, "validation_infiltration_rate.npz"), vd.infiltration_rate)
    npzwrite(joinpath(folder, "validation_total_infiltration_rate.npz"), vd.total_infiltration_rate)
    return folder
end


function visualize(vd, rain_function, topography; folder="")
    f = Figure()

    git_hash = read(`git rev-parse HEAD`, String)[1:end-1]
    hostname = read(`hostname`, String)[1:end-1]
    runinfo = "run on "*hostname*" w/ git commit "*git_hash*"\n"
    infostring = "\n\n"*folder*"\n"*runinfo*"Validation case from Sec 3.2 in https://doi.org/10.1016/j.jhydrol.2016.03.021"

    ax1 = Axis(f[1, 1], ytickcolor = :black, title="Runoff validation "*string(rain_function), 
                xlabel="t [(]minutes]"*infostring, 
                ylabel="discharge rain and infiltration [m^3/s]")
    ax2 = Axis(f[1, 1], ytickcolor = :black, yaxisposition = :right, ygridvisible = false,
                ylabel="accumulated water runoff [m^3]")

    # @show vd.infiltration_rate
    # @show vd.total_infiltration_rate

    xlims!(ax1, 0.0, vd.t[end-1]/60)
    xlims!(ax2, 0.0, vd.t[end-1]/60)
    
    ylims!(ax1, 0.0, 10)
    if rain_function == rain_fcg_1_3
        ylims!(ax2, 0.0, 50000)
    elseif rain_function == rain_fcg_1_4
        ylims!(ax2, 0.0, 15000)
    elseif rain_function == rain_fcg_1_5
        ylims!(ax2, 0.0, 30000)
    else
        ylims!(ax2, 0.0, 75000)
    end

    @show(vd.runoff_volume)

    # runoff_volume = vd.runoff_volume .- vd.runoff_volume[1]
    runoff_rate = zero(vd.runoff_volume)
    runoff_rate[2:end] = (vd.runoff_volume[2:end] - vd.runoff_volume[1:end-1])./((vd.t[2:end] - vd.t[1:end-1]))

    # lines!(ax1, vd.t ./60, [vd.rain_rate, vd.infiltration_rate], label=["rain rate", "infiltration rate"])
    lines!(ax1, vd.t ./60, vd.rain_rate*2000*20, label="rain rate", color = :skyblue2)
    lines!(ax1, vd.t ./60, runoff_rate, label="runoff rate", color = :brown1)
    lines!(ax2, vd.t ./60, vd.runoff_volume, label="runoff volume", color = :blue3)

    hidespines!(ax2)
    hidexdecorations!(ax2)
    # hideydecorations!(ax2, grid = false)
    

    # Situation at x = 1000

    axislegend(ax1)
    display(f)
    
    f2 = Figure()
    ax21 = Axis(f2[1, 1], ytickcolor = :black, title="rain, infiltration and water level at x=1000 "*string(rain_function), 
                xlabel="t [minutes]"*infostring, 
                ylabel="h [m]")
    ax22 = Axis(f2[1, 1], ytickcolor = :black, yaxisposition = :right, ygridvisible = false,
        ylabel="infiltration and rain [m/s]")

    xlims!(ax21, 0.0, vd.t[end-1]/60)
    xlims!(ax22, 0.0, vd.t[end-1]/60)

    ylims!(ax21, 0.0, 0.2)
    ylims!(ax22, 0.0, 0.0002)

    lines!(ax22, vd.t ./60, vd.rain_rate, label="rain rate", color = :skyblue2)
    lines!(ax22, vd.t ./60, vd.infiltration_rate, label="infiltration rate", color = :pink)
    lines!(ax21, vd.t ./60, vd.h_at_1000, label="water depth at", color = :blue3)

    hidespines!(ax22)
    hidexdecorations!(ax22)

    display(f2)
    
    if folder != ""
        save(joinpath(folder, "runoff_validation.png"), f)
        save(joinpath(folder, "situation_at_x_1000.png"), f2)
    end

end

function run_validation_case(backend, topo::ValidationTopography, rain_function; sim_minutes=300)
    
    grid = make_grid(topo, dx=1)
    bottom_topography = make_topography(topo, grid, backend)
    infiltration = make_infiltration(topo, grid, backend)

    depth_cutoff = 10^-3
    equation = SinFVM.ShallowWaterEquations(bottom_topography; depth_cutoff=depth_cutoff, desingularizing_kappa=10^-3)
    reconstruction = SinFVM.LinearReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_fcg2016, Cz=0.003^2)
    rain = SinFVM.FunctionalRain(rain_function, grid)

    source_terms = [SinFVM.SourceTermBottom(), rain, infiltration]
    # source_terms = [SinFVM.SourceTermBottom(), infiltration, rain]
    
    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms, friction)
    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)

    initial = make_initial_water(grid, bottom_topography)
    SinFVM.set_current_state!(simulator, initial)

    init_state =  SinFVM.current_interior_state(simulator)

    B_cells = SinFVM.collect_topography_cells(bottom_topography, grid, interior=true)
    init_volume = sum(collect(init_state.h) - B_cells)*SinFVM.compute_cell_size(grid)
    @show init_volume

    validation_data = ValidationData()
    compute_runoff!(validation_data, topo, simulator, B_cells, rain, infiltration, grid)

    for minute in 1:sim_minutes
        SinFVM.simulate_to_time(simulator, minute*60, maximum_timestep=1.0)# 100*60)
        compute_runoff!(validation_data, topo, simulator, B_cells, rain, infiltration, grid)
        @show minute
    end


    f = Figure(size=(800, 800), fontsize=24)
    ax_h = Axis(f[1, 1], title="hei", xlabel="x", ylabel="z")
    x_cells = SinFVM.cell_centers(grid, XDIR, interior=true)
    x_faces = SinFVM.cell_faces(grid, XDIR, interior=true)
    B_faces = SinFVM.collect_topography_intersections(bottom_topography, grid, interior=true)
    
    state = SinFVM.current_interior_state(simulator)
    w_cells = collect(state.h)

    end_volume = sum(w_cells - B_cells)*SinFVM.compute_cell_size(grid)
    @show end_volume
    mass_loss = init_volume - end_volume 
    @show mass_loss
    lines!(ax_h, x_faces[:], B_faces[:, 1], label="bottom", color="saddlebrown")
    lines!(ax_h, x_cells[:], w_cells[:, 1], label="water level", color="deepskyblue1")
    # lines!(ax_h, x_cells[:], w_cells[:, 1].-B_cells[:, 1], label="water level", color="deepskyblue1")
    axislegend(ax_h)
    display(f)

    @show sum(w_cells[:, 1] .- B_cells[:, 1] .> 1)

    @show size(grid)
    #@show validation_data.t

    folder = write_to_file(validation_data, topo, rain_function)
    
    visualize(validation_data, rain_function, topography)
    return folder
end


backend = SinFVM.make_cuda_backend()
topography = Topography1()
folder_1_1 = run_validation_case(backend, topography, rain_fcg_1_1; sim_minutes=350)
folder_1_2 = run_validation_case(backend, topography, rain_fcg_1_2; sim_minutes=350)
folder_1_3 = run_validation_case(backend, topography, rain_fcg_1_3; sim_minutes=350)
folder_1_4 = run_validation_case(backend, topography, rain_fcg_1_4; sim_minutes=350)
folder_1_5 = run_validation_case(backend, topography, rain_fcg_1_5; sim_minutes=350)
# folder_1_5_2 = run_validation_case(backend, topography, rain_fcg_1_5_2; sim_minutes=350)

# Test read and visualize:
# folder = "validation_data/Topography1_rain_fcg_1_1/240820-121516"
# vd = ValidationData(folder)
# @show vd
# visualize(vd, rain_fcg_1_1, topography, folder=folder)

# @show folder_1_1
# @show folder_1_2
# @show folder_1_3
# @show folder_1_4
# @show folder_1_5

# folder_1_1 =   joinpath("validation_data", "Topography1_rain_fcg_1_1",   "240822-111326")
# folder_1_2 =   joinpath("validation_data", "Topography1_rain_fcg_1_2",   "240822-112602")
# folder_1_3 =   joinpath("validation_data", "Topography1_rain_fcg_1_3",   "240822-113654")
# folder_1_4 =   joinpath("validation_data", "Topography1_rain_fcg_1_4",   "240822-114420")
# folder_1_5 =   joinpath("validation_data", "Topography1_rain_fcg_1_5",   "240822-115414")
# folder_1_5_2 = joinpath("validation_data", "Topography1_rain_fcg_1_5_2", "240822-120437")

folders = [folder_1_1, folder_1_2, folder_1_3, folder_1_4, folder_1_5]
rains = [rain_fcg_1_1, rain_fcg_1_2, rain_fcg_1_3, rain_fcg_1_4, rain_fcg_1_5]
for i in eachindex(folders)
    vd = ValidationData(folders[i])
    visualize(vd, rains[i], topography, folder=folders[i])
    @show i
end