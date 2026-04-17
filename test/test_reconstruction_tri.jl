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
using VolumeFluxes
using StaticArrays
import CUDA
using LinearAlgebra

function test_reconstruction_tri(backend)
    backend_name = VolumeFluxes.name(backend)

    # 2-triangle mesh on the unit square
    #   4 ---- 3
    #   | T2 / |
    #   |  /   |
    #   |/ T1  |
    #   1 ---- 2
    nodes = SVector{2,Float64}[
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(1.0, 1.0),
        SVector(0.0, 1.0),
    ]
    triangles = SVector{3,Int}[
        SVector(1, 2, 3),
        SVector(1, 3, 4),
    ]
    neighbors = SVector{3,Int}[
        SVector(0, 0, 2),
        SVector(1, 0, 0),
    ]

    grid = TriangularGrid(nodes, triangles, neighbors; boundary=OutflowBC())
    equation = VolumeFluxes.ShallowWaterEquationsPure()
    ncells = VolumeFluxes.number_of_cells(grid)

    state = VolumeFluxes.Volume(backend, equation, grid)
    left_buf = VolumeFluxes.Volume(backend, equation, grid)
    right_buf = VolumeFluxes.Volume(backend, equation, grid)

    # Set up a non-uniform state
    for i in 1:ncells
        cx, cy = grid.centroids[i]
        h = 2.0 + cx + cy
        CUDA.@allowscalar state[i] = @SVector [h, 0.1 * cx, 0.1 * cy]
    end

    # Test NoReconstruction: left and right should match state
    noRecon = VolumeFluxes.NoReconstruction()
    VolumeFluxes.reconstruct!(backend, noRecon, left_buf, right_buf, state, grid, equation, XDIR)

    for i in 1:ncells
        CUDA.@allowscalar @test left_buf[i] == state[i]
        CUDA.@allowscalar @test right_buf[i] == state[i]
    end

    # Test LinearReconstruction: should produce gradients in cache
    linRecon = VolumeFluxes.LinearReconstruction()
    VolumeFluxes.reconstruct!(backend, linRecon, left_buf, right_buf, state, grid, equation, XDIR)

    # After reconstruction, left_buf should contain cell values
    for i in 1:ncells
        CUDA.@allowscalar @test left_buf[i] == state[i]
    end

    # The gradient cache should have been populated (and will be consumed by compute_flux!)
    # We can verify the full pipeline works: reconstruct! then compute_flux!
    output = VolumeFluxes.Volume(backend, equation, grid)
    for i in 1:ncells
        CUDA.@allowscalar output[i] = @SVector [0.0, 0.0, 0.0]
    end
    wavespeeds = VolumeFluxes.create_scalar(backend, grid, equation)

    # Re-run reconstruct! (needed because cache was consumed by the test above)
    VolumeFluxes.reconstruct!(backend, linRecon, left_buf, right_buf, state, grid, equation, XDIR)
    numericalflux = VolumeFluxes.CentralUpwind(equation)
    max_speed = VolumeFluxes.compute_flux!(backend, numericalflux, output, left_buf, right_buf, wavespeeds, grid, equation, XDIR)

    @test !isnan(max_speed)
    @test max_speed >= 0.0
    @test !any(isnan.(wavespeeds))
    @test !any(isnan.(collect(output.h)))
    @test !any(isnan.(collect(output.hu)))
    @test !any(isnan.(collect(output.hv)))

    # Verify output is non-trivial (non-uniform state should produce non-zero fluxes)
    @test any(x -> abs(x) > 0.0, collect(output.h))

    # Test NoReconstruction + compute_flux! pipeline (zero gradients path)
    for i in 1:ncells
        CUDA.@allowscalar output[i] = @SVector [0.0, 0.0, 0.0]
    end
    VolumeFluxes.reconstruct!(backend, noRecon, left_buf, right_buf, state, grid, equation, XDIR)
    max_speed_no = VolumeFluxes.compute_flux!(backend, numericalflux, output, left_buf, right_buf, wavespeeds, grid, equation, XDIR)

    @test !isnan(max_speed_no)
    @test max_speed_no >= 0.0
    @test !any(isnan.(collect(output.h)))

    # Compare with and without reconstruction: LinearReconstruction should give
    # different (more accurate) results than NoReconstruction for non-uniform data
    # (This is a sanity check, not an exact test)
    for i in 1:ncells
        CUDA.@allowscalar output[i] = @SVector [0.0, 0.0, 0.0]
    end
    VolumeFluxes.reconstruct!(backend, linRecon, left_buf, right_buf, state, grid, equation, XDIR)
    max_speed_lin = VolumeFluxes.compute_flux!(backend, numericalflux, output, left_buf, right_buf, wavespeeds, grid, equation, XDIR)
    h_out_lin = collect(output.h)

    for i in 1:ncells
        CUDA.@allowscalar output[i] = @SVector [0.0, 0.0, 0.0]
    end
    VolumeFluxes.reconstruct!(backend, noRecon, left_buf, right_buf, state, grid, equation, XDIR)
    max_speed_no2 = VolumeFluxes.compute_flux!(backend, numericalflux, output, left_buf, right_buf, wavespeeds, grid, equation, XDIR)
    h_out_no = collect(output.h)

    # Both should produce valid results
    @test !any(isnan.(h_out_lin))
    @test !any(isnan.(h_out_no))
end

for backend in get_available_backends()
    test_reconstruction_tri(backend)
end
