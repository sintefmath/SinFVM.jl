using CairoMakie
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5, 0.0 .* x]
    nx = 32 * 1024
    grid = SinSWE.CartesianGrid(nx)
    backend = make_cuda_backend()

    equation = SinSWE.ShallowWaterEquations1D()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)
    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = u0.(x)#collect(map(z -> SVector{2,Float64}([z]), u0(x)))
    f = Figure(size = (1600, 600), fontsize = 24)
    ax = Axis(
        f[1, 1],
        title = "Simulation of the Shallow Water equations in 1D.\nCentral Upwind and Forward-Euler.\nResolution $(nx) cells.",
        ylabel = "Solution",
        xlabel = L"x",
    )

    ax2 = Axis(
        f[1, 2],
        title = "Conservation",
        ylabel = "Conserved quantity",
        xlabel = L"t",
    )
    lines!(ax, x, first.(initial), label = L"h_0(x)")
    lines!(ax, x, map(x -> x[2], initial), label=L"hu_0(x)")

    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    SinSWE.set_current_state!(simulator, initial)

    t = 0.0

    T = 0.14
    energy_sw(state) = [sum(first.(state)), sum(map(q -> q[2], state))]
    all_energies = []
    callback(time, sim) = push!(all_energies, energy_sw(SinSWE.current_interior_state(sim)))


    result = collect(SinSWE.current_state(simulator))
    @time SinSWE.simulate_to_time(simulator, T, callback = callback)


    result = collect(SinSWE.current_interior_state(simulator))
    lines!(
        ax,
        x,
        first.(result),
        linestyle = :dot,
        color = :red,
        linewidth = 4,
        label = L"h^{\Delta x}(x, t)",
    )
    lines!(
        ax,
        x,
        map(x -> x[2], result),
        linestyle = :dashdot,
        color = :green,
        linewidth = 4,
        label = L"hu^{\Delta x}(x, t)",
    )

    axislegend(ax, position=:lb)

    lines!(ax2, first.(all_energies), label = L"\int_0^1 h(x, t)\; dx")
    lines!(ax2, map(q -> q[2], all_energies), label = L"\int_0^1 hu(x, $(t))\; dx")
    axislegend(ax2)
    display(f)



end

run_simulation()
