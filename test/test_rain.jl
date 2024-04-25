using StaticArrays
using LinearAlgebra
using Test
import CUDA

using SinSWE


function run_sim(backend, rain, T_hours, grid)
    equation = SinSWE.ShallowWaterEquations()
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, rain)
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)

    u0 = x -> @SVector[1.0, 0.0, 0.0]
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    SinSWE.set_current_state!(simulator, initial)

    @time SinSWE.simulate_to_time(simulator, T_hours*3600)
    @test SinSWE.current_time(simulator) ≈ T_hours*3600 atol=1e-10

    results = SinSWE.current_interior_state(simulator)
    @test all(collect(results.hu) .== 0.0)
    @test all(collect(results.hv) .== 0.0)
    return collect(results.h)
end

constant_rain = SinSWE.ConstantRain(0.01)
@test constant_rain.rain_rate == 0.01
@test SinSWE.compute_rain(constant_rain, 12.6, CartesianIndex(12, 56)) == 0.01/3600.0
@test SinSWE.compute_rain(constant_rain, 12.6) == 0.01/3600.0
@test SinSWE.compute_rain(constant_rain) == 0.01/3600.0

time_dependent_rain = SinSWE.TimeDependentRain(@SVector[15.0, 0.3, 20.0].*0.001, @SVector[0.0, 1.0, 2.0].*3600)
@test SinSWE.compute_rain(time_dependent_rain, 10)  == 0.015/3600.0
@test SinSWE.compute_rain(time_dependent_rain, 3610)  == 0.0003/3600.0
@test SinSWE.compute_rain(time_dependent_rain, 7210)  == 0.02/3600.0
@test SinSWE.compute_rain(time_dependent_rain, 1.0e6)  == 0.02/3600.0
@test SinSWE.compute_rain(time_dependent_rain, 0.0)  == 0.015/3600.0

grid1 = SinSWE.CartesianGrid(10,  10;  gc=2, boundary=SinSWE.WallBC(), extent=[0.0 1000.0; 0.0 1000.0])
grid2 = SinSWE.CartesianGrid(100, 100; gc=2, boundary=SinSWE.WallBC(), extent=[0.0 1000.0; 0.0 1000.0])

h1 = run_sim(SinSWE.make_cpu_backend(), constant_rain, 3, grid1)
@test maximum(h1) ≈ (3*0.01 + 1.0) atol=1e-12
@test minimum(h1) ≈ (3*0.01 + 1.0) atol=1e-12
h2 = run_sim(SinSWE.make_cuda_backend(), constant_rain, 3, grid2)
@test maximum(h2) ≈ (3*0.01 + 1.0) atol=1e-12
@test minimum(h2) ≈ (3*0.01 + 1.0) atol=1e-12

# Check time dependent rain
# Comment: We get the wrong rain in the time step where we change intensity.
#   This causes an error on the scale of rain_intensity*dt/3600
h3 = run_sim(SinSWE.make_cpu_backend(), time_dependent_rain, 1, grid1)
@test maximum(h3) ≈ (0.015 + 1.0) atol=1e-12
@test minimum(h3) ≈ (0.015 + 1.0) atol=1e-12
h4 = run_sim(SinSWE.make_cpu_backend(), time_dependent_rain, 2, grid1)
@test maximum(h4) ≈ (0.0153 + 1.0) atol=1e-4
@test minimum(h4) ≈ (0.0153 + 1.0) atol=1e-4
h5 = run_sim(SinSWE.make_cpu_backend(), time_dependent_rain, 3, grid1)
@test maximum(h5) ≈ (0.0353 + 1.0) atol=1e-4
@test minimum(h5) ≈ (0.0353 + 1.0) atol=1e-4

h6 = run_sim(SinSWE.make_cuda_backend(), time_dependent_rain, 3, grid2)
@test maximum(h6) ≈ (0.0353 + 1.0) atol=1e-7
@test minimum(h6) ≈ (0.0353 + 1.0) atol=1e-7


# @show sum(collect(results1.h))*100*100
# @show sum(10.0*1000*1000)

println("success")
