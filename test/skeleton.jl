using Cthulhu
using StaticArrays
using Test
using CairoMakie
module Correct
include("fasit.jl")
end
using SinFVM
function run_simulation()
    u0 = x -> sin.(2π * x) .+ 1.5
    nx = 16 * 1024
    grid = SinFVM.CartesianGrid(nx)
    backend = make_cpu_backend()

    equation = SinFVM.Burgers()
    reconstruction = SinFVM.NoReconstruction()
    numericalflux = SinFVM.Godunov(equation)
    conserved_system = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinFVM.ForwardEulerStepper()

    x = SinFVM.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid)

    SinFVM.set_current_state!(simulator, initial)
    current_state = SinFVM.current_state(simulator)
    @test current_state[1] == current_state[end-1]
    @test current_state[end] == current_state[2]
    t = 0.0

    T = 0.7



    swe_timesteps = 0
    count_timesteps(varargs...) = swe_timesteps += 1

    @time SinFVM.simulate_to_time(simulator, T, callback=count_timesteps)
    @show swe_timesteps
    SinFVM.set_current_state!(simulator, initial)
    @time SinFVM.simulate_to_time(simulator, T)
    # @profview SinFVM.simulate_to_time(simulator, T)
    f = Figure(size=(1600, 600), fontsize=24)

    ax = Axis(f[1, 1], title="Comparison",
        ylabel="Solution",
        xlabel="x",
    )


    lines!(ax, x, first.(initial), label=L"u_0(x)")
    result = collect(SinFVM.current_interior_state(simulator).u)
    lines!(ax, x, result, linestyle=:dot, color=:red, linewidth=7, label=L"u^{\Delta x}(x, t)")

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