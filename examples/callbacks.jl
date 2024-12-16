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

# # Using the callback functionality
#
# This example demonstrates how to use the callback functionality in the Simulator object of the SinFVM.jl package.
#
# ## Why use callbacks?
# Consider the case where you want to monitor the solution at a particular point in the time. You could do this by saving the solution at every time step and then extracting the solution at that point. However, this is inefficient and can lead to large memory usage. Instead, you can use the callback functionality to monitor the solution at that point without saving the solution at every time step.
# 
# Alternatively, you could run the time loop of the simulation yourself, but this is not recommended as it is easy to make mistakes.
#
# ## What is a callback?
# A callback is a function that is called at a particular point in the time loop of the simulation. The function can be used to monitor the solution, modify the solution, or perform any other action. The callback function is passed to the `simulate_to_time` object as an argument.
#
# ## Example
# We create a simple simulation of the Burgers' equation and use a callback to create an animation of the solution.
#
# First we setup the simulation:
using SinFVM
using CairoMakie

backend = make_cpu_backend()
number_of_cells = 100
grid = grid = CartesianGrid(number_of_cells)
equation = Burgers()
reconstruction = NoReconstruction()
timestepper = ForwardEulerStepper()
numericalflux = CentralUpwind(equation)
system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
simulator = Simulator(backend, conserved_system, timestepper, grid)

# Define the initial condition
u0 = x -> @SVector[sin(2 * pi * x)]
x = cell_centers(grid)
initial = u0.(x)

# Set the initial data created above
SinFVM.set_current_state!(simulator, initial)

# ## Define the callback function
# We will use the plotting functionality of Makie.jl. It is recommended to familiarize yourself with the [animation capabilities of the Makie.jl package](https://docs.makie.org/dev/explanations/animation) before proceeding.
#
# In the callback function, we will update the `Observable` from Makie.jl with the new timestep data. This will automatically update the plot.
#
# Create an `Observable` to store the data. Notice that we are accessing the `.u` field of the current state of the simulator, this will effectively access the conserved variable of the Burgers equation. We are using the `collect` function to be both CPU and GPU compatible.
animation_data = Observable(collect(SinFVM.get_current_state(simulator).u))

# Now we create the plot
fig = Figure()
ax = Axis(fig[1, 1], title = "Burgers' Equation Simulation", xlabel = "x", ylabel = "u")
lines!(ax, x, animation_data)
fig

# Define the callback function
function callback(simulator::Simulator, time::Float64)
    # Update the data
    animation_data[] = collect(SinFVM.get_current_state(simulator).u)
end

# We will wrap the callback function in the IntervalWriter callback, which will call the callback function at regular intervals, in this case every 0.1 simulation time seconds.
callback = IntervalWriter(callback, 0.1)

# ## Run the simulation
# We will run the simulation to a final time of 1.0 and use the callback function to update the plot.
simulate_to_time(simulator, 1.0, callback)



