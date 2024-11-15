using SinFVM
using Test
using StaticArrays
using CairoMakie
using ForwardDiff
using Optim
import KernelAbstractions
import CUDA
using LinearAlgebra
using Cthulhu
using BenchmarkTools
# using GLMakie

using Parameters

@with_kw mutable struct TotalWaterAtCell{IndexType, RealType}
    cell_index::IndexType
    total_water::RealType = 0.0
end

function simpleDamBreak1DOptim(; T=10, dt=1, w0_height=1.0, bump=false, wall_height=0.0, wall_position=0)
    #width_of_wall = 4
    ADType = eltype(wall_height)
    nx = 128
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 200])
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    #@show wall_height
    #@show wall_position

    #dx = SinFVM.compute_dx(grid)
    function terrain(x)
        if x < 100
            b = 2
        elseif 100 <= x <= 150
            b = (2 - 2 * sin(pi / 100 * (x - 100)))
        else
            b = 0.0
        end
        if bump
            b += exp(-(x - 50)^2 / 100)
        end

        b += wall_height * exp(-(x - wall_position)^2 / 30.3)


        return b
    end

    B_data = ADType[terrain(x) for x in SinFVM.cell_faces(grid, interior=false)]
    backend = SinFVM.KernelAbstractionBackend(KernelAbstractions.get_backend(ones(3)); realtype=ADType)
    #backend_name = SinFVM.name(backend)
    B = SinFVM.BottomTopography1D(B_data, backend, grid)
    # display(plot(xf, ForwardDiff.value.(B.B[3:end-2])))
    #B = SinFVM.BottomTopography1D(B_data, backend, grid)
    Bcenter = SinFVM.collect_topography_cells(B, grid)
    eq = SinFVM.ShallowWaterEquations1D(B; depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, bst)

    #balance_system = SinFVM.BalanceSystem(conserved_system, bst)



    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)

    function u0(x)
        if x < 100
            @SVector[w0_height, 0.0]
            # elseif 100 <= x <= 150
            #     @SVector[2-2*sin(pi/100*(x-100))+10^(-6), 0.0]  # x+1 because Julia arrays are 1-indexed
        else
            @SVector[10^(-6), 0.0]
        end
    end

    initial = u0.(x)
    #u0 = x -> @SVector[w0_height*(x < 100), 0.0]
    #initial = u0.(x)

    SinFVM.set_current_state!(simulator, initial)

    t = 0.0



    is_dry = true
    t_wet = ADType(T)
    interestingDryCellIndex = Int64(nx - round(nx / 25))
    # @show interestingDryCellIndex
    depth_cutoff = 10^-3
    h_prev = nothing
    t_prev = t

    
    while t < T
        t += dt

        SinFVM.simulate_to_time(simulator, t)
        state = SinFVM.current_interior_state(simulator)
        h_curr = state.h[interestingDryCellIndex] - Bcenter[interestingDryCellIndex]
        #@show h_curr
        if wall_height != 0.0 && is_dry && h_curr > depth_cutoff
            if h_curr != h_prev


                t_wet = t_prev + h_prev / (h_curr - h_prev) * (t - t_prev)
                # @show typeof(t_wet)
                # @show typeof(h_curr)
                # @show typeof(h_prev)
                # @show typeof(t_prev)
                # @show typeof(t)
            else
                t_wet = t_prev
                # @show typeof(t_wet)
            end

            is_dry = false

            break
        end
        h_prev = h_curr
        t_prev = t

    end


    return t_wet

    #display(f)
end

simpleDamBreak1DOptim(T=200; w0_height=4.0, bump=true, wall_height=1.0, wall_position=150)
exit()
#w, hu, time_wet, botTop, mesh = simpleDamBreak1DOptim(T=200; dt=0.05, w0_height = 4.0, bump=true, wall_height = 1.0, wall_position =150)
println("done")

#@show time_wet
function cost_function_pos(params)
    println("cost func, ", params)
    wall_height = params[1]
    wall_position = params[2]
    #@show wall_height
    #@show wall_position

    # Simulate and get the outputs
    #@show time_until_wet
    time_until_wet = simpleDamBreak1DOptim(T=400; dt=0.05, w0_height=4.0, bump=true, wall_height, wall_position)
    #@show time_until_wet
    wall_cost = 1000 * (wall_height - 1.0)^2 + 2000
    #delay_benefit = 100 * (time_until_cellwet-50)^2
    delay_benefit = 100 * exp(-(time_until_wet - 50)^2)
    #delay_benefit = 100 * (time_until_cellwet - 50)
    position_cost = 50 + 1000 * (sin(1 / 50 * pi * wall_position)^2)
    total_costs = wall_cost - delay_benefit + position_cost
    @show total_costs
    #@show position_cost
    return total_costs
end
# Gradient calculation function using ForwardDiff

function grad!(storage, params)
    #println("grad func, ", params)
    ForwardDiff.gradient!(storage, cost_function_pos, params)
end

# Define the lower and upper bounds
lower_bound = [0.0, 100]  # Minimum values for wall_height and position
upper_bound = [2.5, 185]  # Maximum values for wall_height and position

# Initial guess for the wall height and position
initial_guess = [2.0, 140]  # Initial guess with position as float for optimization

# Set up the optimizer
inner_optimizer = LBFGS()  # You can use other optimizers if preferred
fminbox_optimizer = Fminbox(inner_optimizer)


# Perform bounded optimization using Fminbox with manually provided gradient
result = optimize(cost_function_pos, grad!, lower_bound, upper_bound, initial_guess, fminbox_optimizer)

# Extract the optimal values
optimal_x = result.minimizer
optimal_f = result.minimum

# Print the results
println("Optimal parameters: ", optimal_x)
println("Minimum cost: ", optimal_f)


initial_guess = [2.0, 140]  # Initial guess with position as float for optimization

# Perform bounded optimization using Fminbox with manually provided gradient
@time result = optimize(cost_function_pos, grad!, lower_bound, upper_bound, initial_guess, fminbox_optimizer)

# Extract the optimal values
optimal_x = result.minimizer
optimal_f = result.minimum

# Print the results
println("Optimal parameters: ", optimal_x)
println("Minimum cost: ", optimal_f)


initial_guess = [2.0, 140]  # Initial guess with position as float for optimization

# Perform bounded optimization using Fminbox with manually provided gradient
@time result = optimize(cost_function_pos, grad!, lower_bound, upper_bound, initial_guess, fminbox_optimizer)

# Extract the optimal values
optimal_x = result.minimizer
optimal_f = result.minimum

# Print the results
println("Optimal parameters: ", optimal_x)
println("Minimum cost: ", optimal_f)
initial_guess = [2.0, 140]  # Initial guess with position as float for optimization

@benchmark result = optimize(cost_function_pos, grad!, lower_bound, upper_bound, initial_guess, fminbox_optimizer)


# Extract the optimal values
optimal_x = result.minimizer
optimal_f = result.minimum

# Print the results
println("Optimal parameters: ", optimal_x)
println("Minimum cost: ", optimal_f)
