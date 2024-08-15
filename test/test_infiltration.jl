using StaticArrays
using LinearAlgebra
using Test
import CUDA

using SinSWE


function plain_infiltration(backend, infiltration, grid, T; t0=0.0)
    equation = SinSWE.ShallowWaterEquations()
    reconstruction = SinSWE.LinearReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    conserved_system =
        SinSWE.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, infiltration)
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid; t0=t0)

    u0 = x -> @SVector[1.0, 0.0, 0.0]
    x = SinSWE.cell_centers(grid)
    initial = u0.(x)
    SinSWE.set_current_state!(simulator, initial)

    # TODO: Make callback function that obtains the accumulated infiltration and runoff

    @time SinSWE.simulate_to_time(simulator, t0 + T)

    results = SinSWE.current_interior_state(simulator)
    @test all(collect(results.hu) .== 0.0)
    @test all(collect(results.hv) .== 0.0)
    return collect(results.h)
end

function test_infiltration()
    grid = SinSWE.CartesianGrid(10, 10; gc=2, extent=[0 100; 0 100])
    backend = SinSWE.make_cpu_backend()
    test_inf = SinSWE.HortonInfiltration(grid, backend)
    @test all(test_inf.factor .== 1.0)
    @test size(test_inf.factor) ==  size(grid)
    @test SinSWE.compute_infiltration(test_inf, 0.0, CartesianIndex(1, 1)) == test_inf.f0
    @test SinSWE.compute_infiltration(test_inf, 1e6, CartesianIndex(1, 1)) == test_inf.fc

    factor_bad_size = [1.0 for x in SinSWE.cell_centers(grid)]
    @test_throws DomainError SinSWE.HortonInfiltration(SinSWE.CartesianGrid(2,2), backend; factor=factor_bad_size)

    cuda_backend = SinSWE.make_cuda_backend()
    infiltration_cuda = SinSWE.HortonInfiltration(grid, cuda_backend)

    h_cpu = plain_infiltration(backend, test_inf, grid, 1000; t0=1e6)
    @test maximum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10
    @test minimum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10

    h_cuda = plain_infiltration(cuda_backend, infiltration_cuda, grid, 1000; t0=1e6)
    @test maximum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10
    @test minimum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10


    # grid_case1 = SinSWE.CartesianGrid(2000, 10;  gc=2, boundary=SinSWE.WallBC(), extent=[0.0 4000.0; 0.0 20.0])
end

test_infiltration()


