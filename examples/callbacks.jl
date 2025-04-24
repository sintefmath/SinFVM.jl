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
using StaticArrays

backend = make_cpu_backend()
number_of_cells = 100
grid = grid = CartesianGrid(number_of_cells)
equation = Burgers()
reconstruction = NoReconstruction()
timestepper = ForwardEulerStepper()
numericalflux = Godunov(equation)
system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
simulator = Simulator(backend, system, timestepper, grid)

# Define the initial condition
u0 = x -> @SVector[sin(2 * pi * x)]
x = SinFVM.cell_centers(grid)
initial = u0.(x)

# Set the initial data created above
SinFVM.set_current_state!(simulator, initial)

# ## Define the callback function
# We will use the plotting functionality of Makie.jl. It is recommended to familiarize yourself with the [animation capabilities of the Makie.jl package](https://docs.makie.org/dev/explanations/animation) before proceeding.
#
# In the callback function, we will push back the data to an Vector we create. 
#
# Create the Vector to store the data. Notice that we are accessing the `.u` field of the current state of the simulator, this will effectively access the conserved variable of the Burgers equation. We are using the `collect` function to be both CPU and GPU compatible.
animation_data = [collect(SinFVM.current_interior_state(simulator).u)]


# Define the callback function
function callback(time::Real, simulator::Simulator)
    #!jl Update the data
    push!(animation_data, collect(SinFVM.current_interior_state(simulator).u))
end

# We will wrap the callback function in the IntervalWriter callback, which will call the callback function at regular intervals, in this case every 0.1 simulation time seconds.
interval_callback = IntervalWriter(0.01, callback)

# ## Run the simulation
# We will run the simulation to a final time of 2.0 and use the callback function to update the plot.
#
# Notice that we specify maximum_timestep to be 0.01 to get a smooth animation.
simulate_to_time(simulator, 2.0; callback=interval_callback, maximum_timestep=0.01)

# Now we create the plot
fig = Figure()
ax = Axis(fig[1, 1], title = "Burgers' Equation Simulation", xlabel = "x", ylabel = "u", limits=(0, 1, -1, 1))
lineplot = lines!(ax, x, animation_data[end], color = :blue)


record(fig, "burgers_callback.mp4", 1:length(animation_data), framerate = 24) do index
    lineplot[2] = animation_data[index]
end

```@raw html
<video autoplay loop muted playsinline controls src="./burgers_callback.mp4" />
```
# The animation will be saved as `burgers_callback.mp4` in the current directory. You can view the animation using your favorite video player.