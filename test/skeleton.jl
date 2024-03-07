using Cthulhu
using StaticArrays
using Test
using CairoMakie
module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()
    u0 = x -> sin.(2π * x) .+ 1.5
    nx = 32 * 1024
    grid = SinSWE.CartesianGrid(nx)
    backend = make_cpu_backend()

    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Godunov(equation)
    conserved_system = SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    SinSWE.set_current_state!(simulator, initial)
    current_state = SinSWE.current_state(simulator)
    @test current_state[1] == current_state[end-1]
    @test current_state[end] == current_state[2]
    t = 0.0

    T = 0.7

    energy_sw(state) = [sum(first.(state)) * SinSWE.compute_dx(grid)]
    all_energies = []
    callback(time, sim) = push!(all_energies, energy_sw(SinSWE.current_interior_state(sim)))


    @time SinSWE.simulate_to_time(simulator, T; callback=callback)
    f = Figure(size=(1600, 600), fontsize=24)

    ax = Axis(f[1, 1], title="Comparison",
        ylabel="Solution",
        xlabel="x",
    )

    ax2 = Axis(f[1, 2], title="Conserved quantity",
        ylabel="Conserved",
        xlabel="t",)
    lines!(ax, x, first.(initial), label=L"u_0(x)")
    result = collect(SinSWE.current_interior_state(simulator))
    lines!(ax, x, first.(result), linestyle=:dot, color=:red, linewidth=7, label=L"u^{\Delta x}(x, t)")


    lines!(ax2, first.(all_energies), label=L"\int_0^1 u(x,t)\;dx")
    ylims!(ax2, (minimum(first.(all_energies)) - 1, maximum(first.(all_energies)) + 1))
    axislegend(ax2)


    number_of_x_cells = nx

    println("Running bare bones twice")
    @time xcorrect, ucorrect, _ = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    @time xcorrect, ucorrect, _ = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    lines!(ax, xcorrect, ucorrect, label="Reference solution")


    axislegend(ax)

    display(f)

end

run_simulation()