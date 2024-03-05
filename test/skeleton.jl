using Plots
using Cthulhu
using StaticArrays


module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()
    u0 = x -> sin.(2π * x)
    nx = 64*1024
    grid = SinSWE.CartesianGrid(nx)
    backend = make_cuda_backend()

    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Godunov(equation)
    conserved_system = SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z -> SVector{1,Float64}([z]), u0(x)))
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    SinSWE.set_current_state!(simulator, initial)

    t = 0.0

    T = 1.0
    # plot(x, first.(SinSWE.current_interior_state(simulator)))
    println("Running SinSWE twice")
    @time SinSWE.simulate_to_time(simulator, T)
    SinSWE.set_current_state!(simulator, initial)
    @time SinSWE.simulate_to_time(simulator, T)
    plot!(x, first.(collect(SinSWE.current_interior_state(simulator))))


    number_of_x_cells = nx
    number_of_saves = 100

    println("Running bare bones twice")
    @time xcorrect, ucorrect, _, _, _ = Correct.solve_fvm(u0, T, number_of_x_cells, number_of_saves, Correct.Burgers())
    @time xcorrect, ucorrect, _, _, _ = Correct.solve_fvm(u0, T, number_of_x_cells, number_of_saves, Correct.Burgers())
    pref = plot!(xcorrect, ucorrect)
    display(pref)
    eigenvalue(u) = SinSWE.compute_max_abs_eigenvalue(simulator.system.equation, XDIR, u)
    
    @time SinSWE.compute_wavespeed(simulator.system, grid, initial)
    @time SinSWE.compute_wavespeed(simulator.system, grid, initial)
    @time SinSWE.compute_wavespeed(simulator.system, grid, initial)

    @time maximum(eigenvalue.(initial))
    @time maximum(eigenvalue.(initial))
    @time maximum(eigenvalue.(initial))

end

run_simulation()