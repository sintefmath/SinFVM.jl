using Test
using StaticArrays


using SinSWE
function run_simulation_friction1d(T, backend, equation, grid; elevate=0.0, source_terms = [])

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5 .+ elevate, 0.0 .* x]
    
    reconstruction = SinSWE.LinearReconstruction(1.05)
    numericalflux = SinSWE.CentralUpwind(equation)
    timestepper = SinSWE.RungeKutta2()
    friction = SinSWE.ImplicitFriction()
    conserved_system = SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms, friction)
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    SinSWE.set_current_state!(simulator, initial)
    
    SinSWE.simulate_to_time(simulator, T)
    
    return SinSWE.current_interior_state(simulator)
end

function plot_sols_friction1d(ref_sol, sol, grid, test_name)
    x = SinSWE.cell_centers(grid)
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

function get_test_name_friction1d(backend, eq::SinSWE.Equation)
    backend_name = split(match(r"{(.*?)}", string(typeof(backend)))[1], '.')[end]
    eq_name = match(r"\.(.*?){", string(typeof(eq)))[1]
    return eq_name * " " * backend_name
end
function get_test_name_friction1d(backend, B::SinSWE.AbstractBottomTopography)
    backend_name = split(match(r"{(.*?)}", string(typeof(backend)))[1], '.')[end]
    B_name = match(r"\.(.*?){", string(typeof(B)))[1]
    return B_name * " " * backend_name
end

for backend in SinSWE.get_available_backends()
    nx = 1024  
    grid = SinSWE.CartesianGrid(nx; gc=2)
    T = 0.05

    ref_backend = make_cpu_backend()
    ref_eq = SinSWE.ShallowWaterEquations1DPure()
    ref_sol = run_simulation_friction1d(T, ref_backend, ref_eq, grid)

    for eq in [SinSWE.ShallowWaterEquations1DPure(), 
               SinSWE.ShallowWaterEquations1D()]
        test_name = get_test_name_friction1d(backend, eq)
        @testset "$(test_name)" begin
            sol = run_simulation_friction1d(T, backend, eq, grid)
            #@show test_name
            abs_diff_h  = sum(abs.(collect(ref_sol.h)  - collect(sol.h)))
            abs_diff_hu = sum(abs.(collect(ref_sol.hu) - collect(sol.hu))) 
             @test abs_diff_h  ≈ 0 atol = 10^-7
             @test abs_diff_hu ≈ 0 atol = 10^-7
        end
    end

    if backend isa SinSWE.CPUBackend
        # Test the same setup but with +1 for both B and w_initial
        B_const = 1.0
        B_field = Float64[1.0 for x in SinSWE.cell_faces(grid, interior=false)]
        source_terms = [SinSWE.SourceTermBottom()]
        for B in [SinSWE.ConstantBottomTopography(B_const),
                SinSWE.BottomTopography1D(B_field, backend, grid)]
            test_name = get_test_name_friction1d(backend, B)
            eq = SinSWE.ShallowWaterEquations1D(B)
            @testset "$(test_name)" begin
                sol = run_simulation_friction1d(T, backend, eq, grid; elevate=1.0, source_terms=source_terms)
                #@show test_name
                abs_diff_h  = sum(abs.((collect(ref_sol.h)) - (collect(sol.h) .-1 )))
                abs_diff_hu = sum(abs.(collect(ref_sol.hu) - collect(sol.hu))) 
                # @show abs_diff_h
                # @show abs_diff_hu
                #if abs_diff_h > 10^-6
                    # plot_sols(ref_sol, sol, grid, test_name)
                #end
                @test abs_diff_h  ≈ 0 atol = 10^-2
                @test abs_diff_hu ≈ 0 atol = 10^-2
            end
        end
    end
end


nothing
