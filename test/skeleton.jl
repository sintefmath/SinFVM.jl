using Plots
using StaticArrays

module Correct
include("fasit.jl")
end
using SinSWE
function run_simulation()
    u0 = x->sin.(2π*x)
    nx = 64
    grid = SinSWE.CartesianGrid(nx)
    
    equation = SinSWE.Burgers()
    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.Godunov(equation)
    conserved_system = SinSWE.ConservedSystem(reconstruction, numericalflux, equation, grid)
    timestepper = SinSWE.ForwardEulerStepper()

    x = SinSWE.cell_centers(grid)
    initial = collect(map(z->SVector{1, Float64}([z]), u0(x)))
    #initial = collect(map(z->SVector{1, Float64}([z]), 1.0*(x .< 0.5)))
    simulator = SinSWE.Simulator(conserved_system, timestepper, grid)
    # initial_state = SinSWE.current_state(simulator)

    # for i in 1:nx
    #     # @show initial[i-1]
    #     initial_state[i+grid.ghostcells[1]] = initial[i]
    # end
    

    # SinSWE.current_state(simulator)[2:end-1] .= initial
    SinSWE.set_current_state!(simulator, initial)
    @show SinSWE.current_state(simulator)

    t = 0.0

    T = 1.0 #1000*SinSWE.compute_timestep(simulator)
    @show T
    plot(x, first.(SinSWE.current_interior_state(simulator)))
    while t <= T
        SinSWE.perform_step!(simulator)
        t += SinSWE.current_timestep(simulator)
        #println("$t")
    end

    #print(SinSWE.current_state(simulator))
    plot!(x, first.(SinSWE.current_interior_state(simulator)))


    number_of_x_cells= nx
    number_of_saves = 100
    
    xcorrect, ucorrect, _, _ = Correct.solve_fvm(x->sin(2π*x), T, number_of_x_cells, number_of_saves, Correct.Burgers())
    plot!(xcorrect, ucorrect)
    
end

run_simulation()