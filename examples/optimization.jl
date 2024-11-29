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

# # Optimization of wall placement using AD
# In this example, we will setup a simple model of a dam break scenario. We will use the shallow water equations to simulate the flow of water over a terrain. We will use the `SinFVM` package to setup and solve the model. The model will be solved using the `RungeKutta2` time-stepping method. We will use the `CairoMakie` package to visualize the results of the simulation.

using SinFVM
using StaticArrays
using ForwardDiff
using Optim
using Parameters
using CairoMakie

# ## Set up the optimization problem

# Struct to gather value of interest from the simulation
@with_kw mutable struct TotalWaterAtCell{IndexType,TotalWaterRealType,AreaRealType,CutoffRealType}
    cell_index::IndexType
    total_water::TotalWaterRealType = 0.0
    area_of_cell::AreaRealType
    cutoff::CutoffRealType = 1e-1 # Adjust this as needed!
end;

# Callback for the simulator
function (tw::TotalWaterAtCell)(time, simulator)
    #Get the current water height at the specified cell
    current_h = SinFVM.current_interior_state(simulator).h[tw.cell_index]
    if current_h > tw.cutoff
        dt = ForwardDiff.value(SinFVM.current_timestep(simulator))
        area_of_cell = tw.area_of_cell
        tw.total_water += dt * area_of_cell * current_h
        #Exit if any gradient is NaN
        if tw.total_water isa ForwardDiff.Dual && any(isnan.(tw.total_water.partials))
            println("Exiting due to NaN gradient")
            exit()
        end
    end
end

# We define a helper function to set up the simulator
function setup_simulator(; backend=make_cpu_backend(), wall_height, wall_position)
    #Set up the grid
    nx = 32
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 200])
    x = SinFVM.cell_centers(grid)

    #Define the terrain
    function terrain(x)
        b = 0.02 * (200.0 - x)
        b += wall_height * exp(-(x - wall_position)^2 / 30.3)
        return b
    end
    B_data = [terrain(x) for x in SinFVM.cell_faces(grid, interior=false)]
    B = SinFVM.BottomTopography1D(B_data, backend, grid)
    Bcenter = SinFVM.collect_topography_cells(B, grid)

    #We choose the Shallow Water Equations in 1D
    eq = SinFVM.ShallowWaterEquations1D(B; depth_cutoff=10^-3, desingularizing_kappa=10^-3)

    #We choose a linear reconstruction to get a second order accurate solution
    rec = SinFVM.LinearReconstruction(2)

    #We choose the Central Upwind numerical flux
    flux = SinFVM.CentralUpwind(eq)

    #We add a source term at the bottom
    bst = SinFVM.SourceTermBottom()

    #We add a friction term
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_bsa2012)

    #And we setup the conserved system
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [bst], friction)

    #We use the RungeKutta2 method to get second order in time as well
    timestepper = SinFVM.RungeKutta2()

    #We create the simulator (with a CFL condition of 0.1 for the time stepping)
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)

    #Define the initial conditions
    function u0(x)
        if x < 50
            @SVector[0.02 * (200.0), 0.0]
        else
            @SVector[10^(-6), 0.0]
        end
    end
    initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)

    return simulator
end

# We define the forward simulation model that will be used in the cost function. It returns the total water at a specific cell.
function simple_dambreak_1D_optim(; T, wall_height, wall_position)
    #We choose a CPU backend for this example
    ADType = eltype(wall_height)
    backend = make_cpu_backend(ADType)

    #Set up the simulator
    simulator = setup_simulator(backend=backend, wall_height=wall_height, wall_position=wall_position)

    #Specify index of the cell in which we wish to minimize the water
    nx = SinFVM.number_of_interior_cells(simulator.grid)
    cell_index = Int64(nx - round(nx / 25)) #IMPORTANT: Need to use CartesianIndex in 2d!

    #Run the simulation and return the total water at the specified cell
    area_of_cell = SinFVM.compute_dx(simulator.grid) #IMPORTANT: Different for 2D!
    callback = TotalWaterAtCell(cell_index=cell_index, area_of_cell=area_of_cell, total_water=ADType(0.0))
    SinFVM.simulate_to_time(simulator, T, callback=callback)
    return callback.total_water
end;

# Define the objective function for the optimizer
function cost_function_pos(params)
    wall_height, wall_position = params
    if params isa Vector{Float64}
        println("Current optimization values: wall height = $wall_height, wall position = $wall_position")
    end

    #Run the simulation and get the total water at the specified cell
    total_water_at_cell = simple_dambreak_1D_optim(;T=400, wall_height, wall_position)

    #Assemble a somewhat arbitrary compound cost function that is both explictly and implicitly dependent on the parameters
    wall_cost = 1000 * (wall_height - 1.0)^2 + 2000
    delay_benefit = 100 * exp(-(total_water_at_cell - 50)^2)
    position_cost = 50 + 1000 * (sin(1 / 50 * pi * wall_position)^2)
    total_costs = wall_cost - delay_benefit + position_cost

    #Check that the gradient values are real
    if total_costs isa ForwardDiff.Dual
        @assert all(.!isnan.(total_costs.partials))
    end
    return total_costs
end;

# Gradient calculation function using ForwardDiff
function grad!(storage, params)
    ForwardDiff.gradient!(storage, cost_function_pos, params)
end;

# Define the lower and upper bounds
lower_bound = [0.0, 100];  # Minimum values for wall_height and position
upper_bound = [2.5, 185];  # Maximum values for wall_height and position

# Initial guess for the wall height and position
initial_guess = [2.0, 140];

# Set up the optimizer
inner_optimizer = LBFGS(); # You can use other optimizers if preferred
fminbox_optimizer = Fminbox(inner_optimizer);

# ## Run the optimization

# Perform bounded optimization using Fminbox with manually provided gradient
opts = Optim.Options(store_trace = true)
result = optimize(cost_function_pos, grad!, lower_bound, upper_bound, initial_guess, fminbox_optimizer, opts)

# Optimal values of the wall height and position:
optimal_x = result.minimizer # [wall_height, wall_position]

# Minimum value of the cost function:
optimal_f = result.minimum

#src # Print the optimal values
println("Optimal values: wall height = $(optimal_x[1]), wall position = $(optimal_x[2])") #src

# ## Visualization

# Extract cost values from the optimization history
cost_vals = begin
    iterations_full = map(os->os.iteration, Optim.trace(result)) # This includes inner iterations from the line-search algrotihm
    outer_idxs = iterations_full .== 0 # We only want to plot the outer iterations
    cost_vals = map(os->os.value, Optim.trace(result))
    cost_vals[outer_idxs]
end;

# Visualize the convergence history of the cost function
fig = Figure(size=(1200, 600), fontsize=24)
ax = Axis(
    fig[1, 1],
    title="Convergence history of the cost function",
    ylabel="cost value",
    xlabel="iteration",
)
lines!(ax, 1:length(cost_vals), cost_vals)
scatter!(ax, 1:length(cost_vals), cost_vals)
fig
