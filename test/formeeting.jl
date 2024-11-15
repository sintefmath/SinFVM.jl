using SinFVM
using Test
using StaticArrays
using CairoMakie
using ForwardDiff
using Optim
import KernelAbstractions
import CUDA
using LinearAlgebra
using Cthulhu
#using BenchmarkTools
# using GLMakie
using Parameters
using ProgressMeter



@with_kw mutable struct TotalWaterAtCell{IndexType,TotalWaterRealType,AreaRealType,CutoffRealType}
    cell_index::IndexType
    total_water::TotalWaterRealType = 0.0
    area_of_cell::AreaRealType
    cutoff::CutoffRealType = 1e-1 # Adjust this as needed!
end
function (tw::TotalWaterAtCell)(time, simulator)
    current_h = SinFVM.current_interior_state(simulator).h[tw.cell_index[1], tw.cell_index[2]]
    dt = SinFVM.current_timestep(simulator)
    area_of_cell = tw.area_of_cell
    if current_h > tw.cutoff
        tw.total_water += dt * area_of_cell * current_h
    end
end
function damBreakHouse2DOptim(wall_height, wall_xpos, wall_ypos; show_progress=true)
    #width_of_wall = 4
    ADType = eltype(wall_height)
    nx = 128
    ny = 32
    T = 150
    grid = SinFVM.CartesianGrid(nx, ny, extent=[0 200; 0 28], gc=2, boundary=SinFVM.WallBC())
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid, XDIR)
    yf = SinFVM.cell_faces(grid, YDIR)
    xVals = SinFVM.cell_centers(grid, XDIR)
    yVals = SinFVM.cell_centers(grid, YDIR)
    xFace = collect(SinFVM.cell_faces(grid, XDIR, interior=false))
    yFace = collect(SinFVM.cell_faces(grid, YDIR, interior=false))
    dx = SinFVM.compute_dx(grid)
    dy = SinFVM.compute_dy(grid)
    bottom_topography_array = zeros(ADType, size(grid) .+ (1, 1))

    max_height = 5.0   # Maximum height at the top
    min_height = 0.0   # Minimum height at the bottom

    wall_width = 13

    iCoord = nx - round(Int, nx / 3)
    jCoord = round(Int, ny / 2)
    # Define interestingDryCellIndex as a tuple
    cell_index = (iCoord, jCoord)
    #@show cell_index

    for i in 1:size(bottom_topography_array, 1)
        for j in 1:size(bottom_topography_array, 2)
            # Linearly interpolate the height based on the x coordinate
            x_ratio = i / size(bottom_topography_array, 1)
            bottom_topography_array[i, j] = max_height - x_ratio * (max_height - min_height)

            # Add the wall contribution if within the specified range
            if abs(yFace[j] - wall_ypos) <= wall_width / 2
                #println("here")
                bottom_topography_array[i, j] += wall_height * exp(-(xFace[i] - wall_xpos)^2 / 30.3)
                #println("here2")
            end
        end
    end
    # Create the heatmap
    # if typeof(wall_height) == ForwardDiff.Dual
    #     @show typeof(wall_height)
    #     fig = Figure(resolution=(800, 600))
    #     ax = Axis(fig[1, 1], title="2D Bottom Topography Heatmap", xlabel="X", ylabel="Y")
    #     hm = heatmap!(ax, xf, yf, collect(bottom_topography_array.value[3:end-2, 3:end-2]), colormap=:viridis)
    #     Colorbar(fig[1, 2], hm, label="Height")
    #     display(fig)
    # end



    backend = SinFVM.KernelAbstractionBackend(KernelAbstractions.get_backend(ones(3)); realtype=ADType)
    #backend_name = SinFVM.name(backend)
    bottom_topography = SinFVM.BottomTopography2D(bottom_topography_array, backend, grid)
    bottom_source = SinFVM.SourceTermBottom()
    Bcenter = SinFVM.collect_topography_cells(bottom_topography, grid)
    equation = SinFVM.ShallowWaterEquations(bottom_topography, depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(equation)
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_bsa2012)
    conserved_system =
        SinFVM.ConservedSystem(backend, rec, flux, equation, grid, [bottom_source], friction)
    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.45)
    function u0(x)
        if x[1] < 30
            return SVector(7, 0.0, 0.0)
        else
            return SVector(10^(-6), 0.0, 0.0)
        end
    end
    initial = u0.(x)
    #u0 = x -> @SVector[w0_height*(x < 100), 0.0]
    #initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)
    #cell_index = Int64(nx - round(nx / 25)) # IMPORTANT: Need to use CartesianIndex in 2d! 
    # Eg.
    # cell_index = CartesianIndex(Int64(nx - round(nx / 25)), some_y_coordinate)
    area_of_cell = dx * dy # IMPORTANT: Different for 2D!
    callback = TotalWaterAtCell(cell_index=cell_index, area_of_cell=area_of_cell, total_water=ADType(0.0))
    SinFVM.simulate_to_time(simulator, T, callback=callback, show_progress=show_progress)
    return callback.total_water
    #display(f)
end

function callbackOptim(x, state)
    global iteration_data
    iter = Optim.num_iters(state)
    f_val = Optim.value(state)
    push!(iteration_data, (iter, f_val))
    println("Iteration: $iter, Cost: $f_val")
    false # Return false to continue optimization
end

function cost_function_2D_house(params)
    println("cost func, ", params)
    wall_height = params[1]
    wall_xPos = params[2]
    wall_yPos = params[3]
    total_water_at_cell = damBreakHouse2DOptim(wall_height, wall_xPos, wall_yPos)
    @show total_water_at_cell
    wall_cost = 1000 * (wall_height - 1.0)^2 + 2000
    #delay_benefit = 100 * (time_until_cellwet-50)^2
    #delay_benefit = 100 * exp(-(total_water_at_cell - 50)^2)
    if total_water_at_cell > 10^-2
        water_damage_cost = (total_water_at_cell - 10^-2)^2
    else
        water_damage_cost = 0
    end
    #delay_benefit = 100 * (time_until_cellwet - 50)
    position_cost = 50 + 1000 * (sin(1 / 50 * pi * wall_xPos)^2) + 1000 * (sin(1 / 50 * pi * wall_yPos)^2)
    @show water_damage_cost
    @show wall_cost
    @show position_cost
    total_costs = wall_cost + water_damage_cost + position_cost
    @show total_costs
    #@show position_cost
    if total_costs isa ForwardDiff.Dual
        @assert all(.!isnan.(total_costs.partials))
    end
    return total_costs
end
# Gradient calculation function using ForwardDiff
function grad2D!(storage, params)
    #println("grad func, ", params)
    ForwardDiff.gradient!(storage, cost_function_2D_house, params)
end
# Define the lower and upper bounds
lower_bound = [0.0, 100, 6]  # Minimum values for wall_height and position
upper_bound = [4.0, 185, 22]  # Maximum values for wall_height and position
# Initial guess for the wall height and position
initial_guess = [2.1, 120, 20]  # Initial guess with position as float for optimization
# Set up the optimizer
inner_optimizer = LBFGS()  # You can use other optimizers if preferred
fminbox_optimizer = Fminbox(inner_optimizer)
number_of_tries = 10
heights = LinRange(lower_bound[1], upper_bound[1], number_of_tries)
placement_x = LinRange(lower_bound[2], upper_bound[2], number_of_tries)
placment_y = LinRange(lower_bound[3], upper_bound[3], number_of_tries)
open("all_results.csv", "w") do io
    write(io, "height x y total_water_at_cell\n")
end
@showprogress for h in heights
    for px in placement_x
        for py in placment_y
            total_water_at_cell = damBreakHouse2DOptim(h, px, py; show_progress=false)
            open("all_results.csv", "a") do io
                write(io, "$h $px $py $total_water_at_cell\n")
            end
        end
    end
end
# Perform bounded optimization using Fminbox with manually provided gradient
#result = optimize(cost_function_2D_house, grad2D!, lower_bound, upper_bound, initial_guess, fminbox_optimizer, Optim.Options(show_trace=true, g_tol=20))
# Extract the optimal values
# optimal_x = result.minimizer
# optimal_f = result.minimum
# # Print the results
# println("Optimal parameters: ", optimal_x)
# println("Minimum cost: ", optimal_f)
