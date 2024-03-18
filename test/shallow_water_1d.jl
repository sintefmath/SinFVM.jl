using CairoMakie
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
    nx = 64*1024
    grid = SinSWE.CartesianGrid(nx; gc=2)
    #backend = make_cuda_backend()
    backend = make_cuda_backend()

    equation = SinSWE.ShallowWaterEquations1D(grid)
    reconstruction = SinSWE.NoReconstruction()
    linrec = SinSWE.LinearReconstruction(1.2)
    numericalflux = SinSWE.CentralUpwind(equation)
    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()
    linrec_conserved_system = 
        SinSWE.ConservedSystem(backend, linrec, numericalflux, equation, grid)

    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    T = 0.05

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

   

    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)
    linrec_simulator = SinSWE.Simulator(backend, linrec_conserved_system, timestepper, grid; cfl=0.01)

    
    SinSWE.set_current_state!(linrec_simulator, initial)
    SinSWE.set_current_state!(simulator, initial)


    initial_state = SinSWE.current_interior_state(simulator)
    lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
    lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")

    t = 0.0

 
   

    result = collect(SinSWE.current_state(simulator))
    @time SinSWE.simulate_to_time(simulator, T) 
    @time SinSWE.simulate_to_time(linrec_simulator, T)

    
    result = SinSWE.current_interior_state(simulator)
    linrec_results = SinSWE.current_interior_state(linrec_simulator)
    
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
        color=:green,
        linewidth=8,
        label=L"hu^{\Delta x}(x, t)",
    )
    lines!(
        ax,
        x,
        collect(linrec_results.h),
        linestyle=:dot,
        color=:orange,
        linewidth=4,
        label=L"h_2^{\Delta x}(x, t)",
    )
    lines!(
        ax2,
        x,
        collect(linrec_results.hu),
        linestyle=:dashdot,
        color=:purple,
        linewidth=4,
        label=L"hu_2^{\Delta x}(x, t)",
    )
    axislegend(ax, position=:lb)
    axislegend(ax2, position=:lb)

    
    display(f)



end

run_simulation()
