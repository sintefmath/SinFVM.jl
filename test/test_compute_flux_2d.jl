using Test
using SinSWE
using StaticArrays
import CUDA
using LinearAlgebra

function test_compute_flux_2d(backend)
    backend_name = SinSWE.name(backend)
    u0 = x -> @SVector[exp.(-(norm(x .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0]
    nx = 64
    ny = 128
    grid = SinSWE.CartesianGrid(nx, ny; gc=1)
    
    equation = SinSWE.ShallowWaterEquations()

    reconstruction = SinSWE.NoReconstruction()
    numericalflux = SinSWE.CentralUpwind(equation)

    x = SinSWE.cell_centers(grid)
    initial = u0.(x)

    state = SinSWE.Volume(backend, equation, grid)
    output_state = SinSWE.Volume(backend, equation, grid)
    interior_state = SinSWE.InteriorVolume(state)
    CUDA.@allowscalar interior_state[:, :] = initial

    wavespeeds = SinSWE.create_scalar(backend, grid, equation)
    SinSWE.compute_flux!(backend, numericalflux, output_state, state, state, wavespeeds, grid, equation, XDIR)

    @test !any(isnan.(wavespeeds))
    @test !any(isnan.(collect(output_state.h)))
    @test !any(isnan.(collect(output_state.hv)))
    @test !any(isnan.(collect(output_state.hu)))

    SinSWE.compute_flux!(backend, numericalflux, output_state, state, state, wavespeeds, grid, equation, YDIR)

    @test !any(isnan.(wavespeeds))
    @test !any(isnan.(collect(output_state.h)))
    @test !any(isnan.(collect(output_state.hv)))
    @test !any(isnan.(collect(output_state.hu)))
end

for backend in get_available_backends()
    test_compute_flux_2d(backend)
end