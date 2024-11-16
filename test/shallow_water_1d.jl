using CairoMakie
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinFVM
function run_simulation()

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
    nx = 1024
    grid = SinFVM.CartesianGrid(nx; gc=2)
    #backend = make_cuda_backend()
    backend = make_cpu_backend()

    equation = SinFVM.ShallowWaterEquations1D()
    pure_equation = SinFVM.ShallowWaterEquations1DPure()
    reconstruction = SinFVM.NoReconstruction()
    linrec = SinFVM.LinearReconstruction(1.05)
    numericalflux = SinFVM.CentralUpwind(equation)
    pure_numericalflux = SinFVM.CentralUpwind(pure_equation)
    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinFVM.ForwardEulerStepper()
    linrec_conserved_system = 
        SinFVM.ConservedSystem(backend, linrec, numericalflux, equation, grid)
    pure_linrec_conserved_system = 
        SinFVM.ConservedSystem(backend, linrec, pure_numericalflux, pure_equation, grid)
    x = SinFVM.cell_centers(grid)
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

   

    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)
    linrec_simulator = SinFVM.Simulator(backend, linrec_conserved_system, timestepper, grid; cfl=0.2)
    pure_linrec_simulator = SinFVM.Simulator(backend, pure_linrec_conserved_system, timestepper, grid; cfl=0.2)

    
    SinFVM.set_current_state!(linrec_simulator, initial)
    SinFVM.set_current_state!(simulator, initial)
    SinFVM.set_current_state!(pure_linrec_simulator, initial)
    


    initial_state = SinFVM.current_interior_state(simulator)
    lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
    lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")

    t = 0.0

 
   

    result = collect(SinFVM.current_state(simulator))
    @time SinFVM.simulate_to_time(simulator, T) 
    @time SinFVM.simulate_to_time(linrec_simulator, T)
    @time SinFVM.simulate_to_time(pure_linrec_simulator, T)

    
    result = SinFVM.current_interior_state(simulator)
    linrec_results = SinFVM.current_interior_state(linrec_simulator)
    pure_results = SinFVM.current_interior_state(pure_linrec_simulator)
    
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
        color=:orange,
        linewidth=4,
        label=L"hu_2^{\Delta x}(x, t)",
    )
    lines!(
        ax,
        x,
        collect(pure_results.h),
        linestyle=:dot,
        color=:green,
        linewidth=4,
        label=L"h_{pure}^{\Delta x}(x, t)",
    )
    lines!(
        ax2,
        x,
        collect(pure_results.hu),
        linestyle=:dashdot,
        color=:green,
        linewidth=4,
        label=L"hu_{pure}^{\Delta x}(x, t)",
    )
    axislegend(ax, position=:lt)
    axislegend(ax2, position=:lt)

    
    display(f)

    @show sum(abs.(collect(pure_results.h) - collect(linrec_results.h)))
    @show sum(abs.(collect(pure_results.hu) - collect(linrec_results.hu)))



end

run_simulation()
