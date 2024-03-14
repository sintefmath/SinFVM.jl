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

  

    swe_timesteps = 0
    count_timesteps(varargs...) = swe_timesteps += 1

    @time SinSWE.simulate_to_time(simulator, T, callback=count_timesteps)
    @show swe_timesteps
    SinSWE.set_current_state!(simulator, initial)
    @time SinSWE.simulate_to_time(simulator, T)

    f = Figure(size=(1600, 600), fontsize=24)

    ax = Axis(f[1, 1], title="Comparison",
        ylabel="Solution",
        xlabel="x",
    )

   
    lines!(ax, x, first.(initial), label=L"u_0(x)")
    result = collect(SinSWE.current_interior_state(simulator))
    lines!(ax, x, first.(result), linestyle=:dot, color=:red, linewidth=7, label=L"u^{\Delta x}(x, t)")

    number_of_x_cells = nx

    println("Running bare bones twice")
    @time xcorrect, ucorrect, _ = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    @time xcorrect, ucorrect, timesteps = Correct.solve_fvm(u0, T, number_of_x_cells, Correct.Burgers())
    @show timesteps
    lines!(ax, xcorrect, ucorrect, label="Reference solution")


    axislegend(ax)

    display(f)

end

run_simulation()