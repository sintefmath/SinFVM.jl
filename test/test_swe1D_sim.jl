# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Test
using StaticArrays


using VolumeFluxes
isdefined(Main, :maybe_display) || include("testing_utils.jl")
function run_simulation(T, backend, equation, grid; elevate=0.0, source_terms = [])

    u0 = x -> @SVector[exp.(-(x - 0.5)^2 / 0.001) .+ 1.5 .+ elevate, 0.0 .* x]
    
    reconstruction = VolumeFluxes.LinearReconstruction(1.05)
    numericalflux = VolumeFluxes.CentralUpwind(equation)
    timestepper = VolumeFluxes.ForwardEulerStepper()
    conserved_system = VolumeFluxes.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, source_terms)
    simulator = VolumeFluxes.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = VolumeFluxes.cell_centers(grid)
    initial = u0.(x)
    VolumeFluxes.set_current_state!(simulator, initial)
    
    VolumeFluxes.simulate_to_time(simulator, T)
    @test VolumeFluxes.current_time(simulator) == T
    
    return VolumeFluxes.current_interior_state(simulator)
end

function plot_sols(ref_sol, sol, grid, test_name)
    x = VolumeFluxes.cell_centers(grid)
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

    maybe_display(f)
end

# NOTE: this used to build the name by regex over `string(typeof(...))`, matching on a
# `.` before the type name. `ShallowWaterEquations1DPure` and friends are exported, so
# they print without a module prefix, `match` returned `nothing`, and indexing it threw.
get_test_name(backend, eq::VolumeFluxes.Equation) =
    string(nameof(typeof(eq)), " ", VolumeFluxes.name(backend))
get_test_name(backend, B::VolumeFluxes.AbstractBottomTopography) =
    string(nameof(typeof(B)), " ", VolumeFluxes.name(backend))

for backend in VolumeFluxes.get_available_backends()
    nx = 1024  
    grid = VolumeFluxes.CartesianGrid(nx; gc=2)
    T = 0.05

    ref_backend = make_cpu_backend()
    ref_eq = VolumeFluxes.ShallowWaterEquations1DPure()
    ref_sol = run_simulation(T, ref_backend, ref_eq, grid)

    for eq in [VolumeFluxes.ShallowWaterEquations1DPure(), 
               VolumeFluxes.ShallowWaterEquations1D()]
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

    if backend isa VolumeFluxes.CPUBackend
        # Test the same setup but with +1 for both B and w_initial
        B_const = 1.0
        B_field = Float64[1.0 for x in VolumeFluxes.cell_faces(grid, interior=false)]
        source_terms = [VolumeFluxes.SourceTermBottom()]
        for B in [VolumeFluxes.ConstantBottomTopography(B_const),
                VolumeFluxes.BottomTopography1D(B_field, backend, grid)]
            test_name = get_test_name(backend, B)
            eq = VolumeFluxes.ShallowWaterEquations1D(B)
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
