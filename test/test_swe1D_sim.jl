using Test
using StaticArrays


using SinFVM
function run_simulation(T, backend, equation, grid; elevate=0.0, source_terms = [])

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5 .+ elevate, 0.0 .* x]
    
    reconstruction = SinFVM.LinearReconstruction(1.05)
    numericalflux = SinFVM.CentralUpwind(equation)
    timestepper = SinFVM.ForwardEulerStepper()
    conserved_system = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms)
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = SinFVM.cell_centers(grid)
    initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)
    
    SinFVM.simulate_to_time(simulator, T)
    @test SinFVM.current_time(simulator) == T
    
    return SinFVM.current_interior_state(simulator)
end

function plot_sols(ref_sol, sol, grid, test_name)
    x = SinFVM.cell_centers(grid)
    f = Figure(size=(1600, 600), fontsize=24)
    ax = Axis(
        f[1, 1],
        title="test_name",
        ylabel="h",
        xlabel="x",
    )
    lines!(ax, x, collect(ref_sol.h), label="ref_sol")
    lines!(ax, x, collect(sol.h), label="sol")
    axislegend(ax, position=:lt)

    display(f)
end

function get_test_name(backend, eq::SinFVM.Equation)
    backend_name = split(match(r"{(.*?)}", string(typeof(backend)))[1], '.')[end]
    eq_name = match(r"\.(.*?){", string(typeof(eq)))[1]
    return eq_name * " " * backend_name
end
function get_test_name(backend, B::SinFVM.AbstractBottomTopography)
    backend_name = split(match(r"{(.*?)}", string(typeof(backend)))[1], '.')[end]
    B_name = match(r"\.(.*?){", string(typeof(B)))[1]
    return B_name * " " * backend_name
end

for backend in SinFVM.get_available_backends()
    nx = 1024  
    grid = SinFVM.CartesianGrid(nx; gc=2)
    T = 0.05

    ref_backend = make_cpu_backend()
    ref_eq = SinFVM.ShallowWaterEquations1DPure()
    ref_sol = run_simulation(T, ref_backend, ref_eq, grid)

    for eq in [SinFVM.ShallowWaterEquations1DPure(), 
               SinFVM.ShallowWaterEquations1D()]
        test_name = get_test_name(backend, eq)
        @testset "$(test_name)" begin
            sol = run_simulation(T, backend, eq, grid)
            #@show test_name
            abs_diff_h  = sum(abs.(collect(ref_sol.h)  - collect(sol.h)))
            abs_diff_hu = sum(abs.(collect(ref_sol.hu) - collect(sol.hu))) 
            # @show abs_diff_h
            # @show abs_diff_hu
            #if abs_diff_h > 10^-6
                # plot_sols(ref_sol, sol, grid, test_name)
            #end
             @test abs_diff_h  ≈ 0 atol = 10^-7
             @test abs_diff_hu ≈ 0 atol = 10^-7
        end
    end

    if backend isa SinFVM.CPUBackend
        # Test the same setup but with +1 for both B and w_initial
        B_const = 1.0
        B_field = Float64[1.0 for x in SinFVM.cell_faces(grid, interior=false)]
        source_terms = [SinFVM.SourceTermBottom()]
        for B in [SinFVM.ConstantBottomTopography(B_const),
                SinFVM.BottomTopography1D(B_field, backend, grid)]
            test_name = get_test_name(backend, B)
            eq = SinFVM.ShallowWaterEquations1D(B)
            @testset "$(test_name)" begin
                sol = run_simulation(T, backend, eq, grid; elevate=1.0, source_terms=source_terms)
                #@show test_name
                abs_diff_h  = sum(abs.((collect(ref_sol.h)) - (collect(sol.h) .-1 )))
                abs_diff_hu = sum(abs.(collect(ref_sol.hu) - collect(sol.hu))) 
                # @show abs_diff_h
                # @show abs_diff_hu
                #if abs_diff_h > 10^-6
                    # plot_sols(ref_sol, sol, grid, test_name)
                #end
                @test abs_diff_h  ≈ 0 atol = 10^-6
                @test abs_diff_hu ≈ 0 atol = 10^-6
            end
        end
    end
end


nothing

