# # Shallow Water Equations in 1D without bottom Bottom topography
# In this example, we will solve the Shallow Water Equations in 1D without bottom topography. We will use the `SinFVM` package to setup and solve the model. The model will be solved using the `ForwardEuler` time-stepping method. We will use the `CairoMakie` package to visualize the results of the simulation.

using CairoMakie
using Cthulhu
using StaticArrays
using SinFVM

# ## Setup the grid and initial conditions
# At the very first, we need to set the backend. For this example, we will use the CPU backend.
backend = make_cpu_backend()

# We then set the number of cells in the grid
nx = 1024
# Then we create the grid
grid = CartesianGrid(nx; gc=2)

# We define the initial conditions for the simulation
u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
x = SinFVM.cell_centers(grid)
initial = u0.(x)

# ## Setup the simulation
# We choose the Shallow Water Equations in 1D, here 'Pure' refers to the version without bottom topography
equation = SinFVM.ShallowWaterEquations1DPure()

# We choose a linear reconstruction to get a second order accurate solution
reconstruction = SinFVM.LinearReconstruction(1.05)

# We choose the Central Upwind numerical flux
numericalflux = SinFVM.CentralUpwind(equation)

# And we setup the conserved system
conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)

# Note that we use the RungeKutta2 method to get second order in time as well
timestepper = SinFVM.RungeKutta2()

# We define the final time
T = 0.05
# Then we create the simulator
simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

# Set the initial data created above
SinFVM.set_current_state!(simulator, initial)
    
# ## Visualization
# We will visualize the results using the `CairoMakie` package, so first we create a figure with two axes
f = Figure(size=(1600, 600), fontsize=24)
ax = Axis(
    f[1, 1],
    title="Simulation of the Shallow Water equations in 1D.\nCentral Upwind and Forward-Euler.\nResolution $(nx) cells.\nT=$(T)",
    ylabel="h",
    xlabel=L"x",
)

ax2 = Axis(
    f[1, 2],
    title="Simulation of the Shallow Water equations in 1D.\nCentral Upwind and Forward-Euler.\nResolution $(nx) cells.\nT=$(T)",
    ylabel="hu",
    xlabel=L"x",
)

# We get the current state of the simulator (initial state since the simulator has not run yet)
initial_state = SinFVM.current_interior_state(simulator)
nothing

# The initial_state is a Volume object (technically an InteriorVolume), so we can access the data using the `h` and `hu` fields
lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")
f

# ## Run the simulation
# We run the simulation to the final time `T`
@time SinFVM.simulate_to_time(simulator, T) 

# ## Visualize the results
# We get the final state of the simulation   
result = SinFVM.current_interior_state(simulator)
nothing

# and plot said results
lines!(
    ax,
    x,
    collect(result.h),
    linestyle=:dot,
    color=:red,
    linewidth=8,
    label=L"h^{\Delta x}(x, t)",
)
lines!(
    ax2,
    x,
    collect(result.hu),
    linestyle=:dashdot,
    color=:red,
    linewidth=8,
    label=L"hu^{\Delta x}(x, t)",
)

axislegend(ax, position=:lt)
axislegend(ax2, position=:lt)
f