using StaticArrays
using LinearAlgebra
using Test
import CUDA
using Logging
using SinFVM


function plain_infiltration(backend, infiltration, grid, T; t0=0.0)
    equation = SinFVM.ShallowWaterEquations()
    reconstruction = SinFVM.LinearReconstruction()
    numericalflux = SinFVM.CentralUpwind(equation)

    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, infiltration)
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid; t0=t0)

    u0 = x -> @SVector[1.0, 0.0, 0.0]
    x = SinFVM.cell_centers(grid)
    initial = u0.(x)
    SinFVM.set_current_state!(simulator, initial)

    # TODO: Make callback function that obtains the accumulated infiltration and runoff

    @time SinFVM.simulate_to_time(simulator, t0 + T)

    results = SinFVM.current_interior_state(simulator)
    @test all(collect(results.hu) .== 0.0)
    @test all(collect(results.hv) .== 0.0)
    return collect(results.h)
end

function test_infiltration()
    grid = SinFVM.CartesianGrid(10, 10; gc=2, extent=[0 100; 0 100])
    backend = SinFVM.make_cpu_backend()
    test_inf = SinFVM.HortonInfiltration(grid, backend)
    @test all(test_inf.factor .== 1.0)
    @test size(test_inf.factor) ==  size(grid)
    @test SinFVM.compute_infiltration(test_inf, 0.0, CartesianIndex(1, 1)) == test_inf.f0
    @test SinFVM.compute_infiltration(test_inf, 1e6, CartesianIndex(1, 1)) == test_inf.fc

    factor_bad_size = [1.0 for x in SinFVM.cell_centers(grid)]
    @test_throws DomainError SinFVM.HortonInfiltration(SinFVM.CartesianGrid(2,2), backend; factor=factor_bad_size)

    if SinFVM.has_cuda_backend()
        cuda_backend = SinFVM.make_cuda_backend()
        infiltration_cuda = SinFVM.HortonInfiltration(grid, cuda_backend)

        h_cpu = plain_infiltration(backend, test_inf, grid, 1000; t0=1e6)
        @test maximum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10
        @test minimum(h_cpu) .≈ ( 1.0 - test_inf.fc*1000) atol=1e-10

        h_cuda = plain_infiltration(cuda_backend, infiltration_cuda, grid, 1000; t0=1e6)
        @test maximum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10
        @test minimum(h_cuda) .≈ ( 1.0 - infiltration_cuda.fc*1000) atol=1e-10
    else
        @debug "CUDA not available, skipping CUDA tests"
    end
    # grid_case1 = SinFVM.CartesianGrid(2000, 10;  gc=2, boundary=SinFVM.WallBC(), extent=[0.0 4000.0; 0.0 20.0])
end

test_infiltration()


