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

function test_compute_flux_tri(backend)
    backend_name = VolumeFluxes.name(backend)

    # Build a simple triangular grid: 2 triangles on the unit square
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
        SVector(0, 0, 2),   # T1: edges 1,2 boundary; edge 3 -> T2
        SVector(1, 0, 0),   # T2: edge 1 -> T1; edges 2,3 boundary
    ]

    for bc in [OutflowBC(), TriangularWallBC()]
        grid = TriangularGrid(nodes, triangles, neighbors; boundary=bc)
        equation = VolumeFluxes.ShallowWaterEquationsPure()
        numericalflux = VolumeFluxes.CentralUpwind(equation)

        ncells = VolumeFluxes.number_of_cells(grid)
        @test ncells == 2

        # Set up state with a Gaussian bump
        state = VolumeFluxes.Volume(backend, equation, grid)
        output_state = VolumeFluxes.Volume(backend, equation, grid)

        for i in 1:ncells
            cx, cy = grid.centroids[i]
            h = exp(-(cx - 0.5)^2 / 0.1 - (cy - 0.5)^2 / 0.1) + 1.5
            CUDA.@allowscalar state[i] = @SVector [h, 0.0, 0.0]
            CUDA.@allowscalar output_state[i] = @SVector [0.0, 0.0, 0.0]
        end

        wavespeeds = VolumeFluxes.create_scalar(backend, grid, equation)

        # Verify initial state has no NaN
        for i in 1:ncells
            CUDA.@allowscalar @test !any(isnan.(state[i]))
        end

        # compute_flux! should run without error and produce finite results
        max_speed = VolumeFluxes.compute_flux!(backend, numericalflux, output_state,
                                               state, state, wavespeeds,
                                               grid, equation, XDIR)

        @test !isnan(max_speed)
        @test max_speed >= 0.0
        @test !any(isnan.(wavespeeds))

        @test !any(isnan.(collect(output_state.h)))
        @test !any(isnan.(collect(output_state.hu)))
        @test !any(isnan.(collect(output_state.hv)))

        # Run again with YDIR (should also work — direction is ignored for triangular)
        for i in 1:ncells
            CUDA.@allowscalar output_state[i] = @SVector [0.0, 0.0, 0.0]
        end

        max_speed_y = VolumeFluxes.compute_flux!(backend, numericalflux, output_state,
                                                  state, state, wavespeeds,
                                                  grid, equation, YDIR)

        @test !isnan(max_speed_y)
        @test max_speed_y >= 0.0
        @test !any(isnan.(wavespeeds))
        @test !any(isnan.(collect(output_state.h)))
        @test !any(isnan.(collect(output_state.hu)))
        @test !any(isnan.(collect(output_state.hv)))

        # The results should be the same regardless of direction
        @test max_speed ≈ max_speed_y
    end

    # Test with a larger mesh: structured 8-triangle grid on the unit square
    nodes_large = SVector{2,Float64}[
        SVector(0.0, 0.0), SVector(0.5, 0.0), SVector(1.0, 0.0),
        SVector(0.0, 0.5), SVector(0.5, 0.5), SVector(1.0, 0.5),
        SVector(0.0, 1.0), SVector(0.5, 1.0), SVector(1.0, 1.0),
    ]
    triangles_large = SVector{3,Int}[
        SVector(1, 2, 5), SVector(1, 5, 4),
        SVector(2, 3, 6), SVector(2, 6, 5),
        SVector(4, 5, 8), SVector(4, 8, 7),
        SVector(5, 6, 9), SVector(5, 9, 8),
    ]
    neighbors_large = SVector{3,Int}[
        SVector(0, 4, 2),  # T1
        SVector(1, 5, 0),  # T2
        SVector(0, 0, 4),  # T3
        SVector(3, 7, 1),  # T4
        SVector(2, 8, 6),  # T5
        SVector(5, 0, 0),  # T6
        SVector(4, 0, 8),  # T7
        SVector(7, 0, 5),  # T8
    ]

    grid_large = TriangularGrid(nodes_large, triangles_large, neighbors_large;
                                 boundary=OutflowBC())
    equation = VolumeFluxes.ShallowWaterEquationsPure()
    numericalflux = VolumeFluxes.CentralUpwind(equation)
    ncells_large = VolumeFluxes.number_of_cells(grid_large)
    @test ncells_large == 8

    state_large = VolumeFluxes.Volume(backend, equation, grid_large)
    output_large = VolumeFluxes.Volume(backend, equation, grid_large)

    for i in 1:ncells_large
        cx, cy = grid_large.centroids[i]
        h = exp(-(cx - 0.5)^2 / 0.05 - (cy - 0.5)^2 / 0.05) + 1.5
        CUDA.@allowscalar state_large[i] = @SVector [h, 0.0, 0.0]
        CUDA.@allowscalar output_large[i] = @SVector [0.0, 0.0, 0.0]
    end

    wavespeeds_large = VolumeFluxes.create_scalar(backend, grid_large, equation)

    max_speed_large = VolumeFluxes.compute_flux!(backend, numericalflux, output_large,
                                                  state_large, state_large,
                                                  wavespeeds_large,
                                                  grid_large, equation, XDIR)

    @test !isnan(max_speed_large)
    @test max_speed_large > 0.0
    @test !any(isnan.(wavespeeds_large))
    @test !any(isnan.(collect(output_large.h)))
    @test !any(isnan.(collect(output_large.hu)))
    @test !any(isnan.(collect(output_large.hv)))

    # Verify that flux produces non-trivial output (not all zeros)
    h_out = collect(output_large.h)
    @test any(x -> abs(x) > 0.0, h_out)
end

for backend in get_available_backends()
    test_compute_flux_tri(backend)
end
