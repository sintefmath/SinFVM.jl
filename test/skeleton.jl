using Plots
using StaticArrays

module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()
    u0 = x -> sin.(2π * x)
    nx = 16 * 1024
    grid = SinSWE.CartesianGrid(nx)

    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Godunov(equation)
    conserved_system = SinSWE.ConservedSystem(reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinSWE.Simulator(conserved_system, timestepper, grid)

    SinSWE.set_current_state!(simulator, initial)

    t = 0.0

    T = 1.0
    plot(x, first.(SinSWE.current_interior_state(simulator)))
    @time SinSWE.simulate_to_time(simulator, T)

    plot!(x, first.(SinSWE.current_interior_state(simulator)))


    number_of_x_cells = nx
    number_of_saves = 100

    @time xcorrect, ucorrect, _, _ = Correct.solve_fvm(x -> sin(2π * x), T, number_of_x_cells, number_of_saves, Correct.Burgers())
    plot!(xcorrect, ucorrect)

end

run_simulation()