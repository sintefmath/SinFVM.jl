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

using VolumeFluxes
isdefined(Main, :test_backends) || include("testing_utils.jl")
using CUDA
using Test

@testset "$(backend_label(backend))" for backend in test_backends()
    nx = 64
    ny = 32
    grid = VolumeFluxes.CartesianGrid(nx, ny)
    @test VolumeFluxes.compute_dx(grid, XDIR) == 1.0 / nx
    @test VolumeFluxes.compute_dx(grid, YDIR) == 1.0 / ny

    @test VolumeFluxes.size(grid) == (nx + 2, ny + 2)
    @test VolumeFluxes.inner_cells(grid, XDIR) == (nx, ny)
    @test VolumeFluxes.inner_cells(grid, YDIR) == (nx, ny)
    @test VolumeFluxes.inner_cells(grid, XDIR, 0) == (nx + 2, ny)
    @test VolumeFluxes.inner_cells(grid, YDIR, 0) == (nx, ny + 2)

    @test VolumeFluxes.middle_cell(grid, CartesianIndex(nx, ny), XDIR) == CartesianIndex(nx + 1, ny + 1)
    @test VolumeFluxes.middle_cell(grid, CartesianIndex(nx, ny), YDIR) == CartesianIndex(nx + 1, ny + 1)
    for j in 1:ny
        for i in 1:nx
            @test VolumeFluxes.middle_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i + 1, j + 1)
            @test VolumeFluxes.middle_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j + 1)

            @test VolumeFluxes.left_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i, j + 1)
            @test VolumeFluxes.left_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j)

            @test VolumeFluxes.right_cell(grid, CartesianIndex(i, j), XDIR) == CartesianIndex(i + 2, j + 1)
            @test VolumeFluxes.right_cell(grid, CartesianIndex(i, j), YDIR) == CartesianIndex(i + 1, j + 2)
        end
    end

    # XDIR  
    output_left = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_middle = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_right = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR) do left, middle, right
        output_left[left, 1] = left[1]
        output_left[left, 2] = left[2]

        output_middle[middle, 1] = middle[1]
        output_middle[middle, 2] = middle[2]

        output_right[right, 1] = right[1]
        output_right[right, 2] = right[2]
    end

    for y in 1:(ny+2)
        for x in (1:nx+2)
            if x == 1 || y == 1 || x == nx + 2 || y == ny + 2
                CUDA.@allowscalar @test output_middle[x, y, 1] == -42
                CUDA.@allowscalar @test output_middle[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_middle[x, y, 1] == x
                CUDA.@allowscalar @test output_middle[x, y, 2] == y
            end

            if y == 1 || x == nx + 2 || x == nx + 1 || y == ny + 2
                CUDA.@allowscalar @test output_left[x, y, 1] == -42
                CUDA.@allowscalar @test output_left[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_left[x, y, 1] == x
                CUDA.@allowscalar @test output_left[x, y, 2] == y
            end
            if x == 1 || x == 2 || y == 1 || y == ny + 2
                CUDA.@allowscalar @test output_right[x, y, 1] == -42
                CUDA.@allowscalar @test output_right[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_right[x, y, 1] == x
                CUDA.@allowscalar @test output_right[x, y, 2] == y
            end
        end
    end


    # YDIR  
    output_left = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_middle = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))
    output_right = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2, ny + 2, 2))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, YDIR) do left, middle, right
        output_left[left, 1] = left[1]
        output_left[left, 2] = left[2]

        output_middle[middle, 1] = middle[1]
        output_middle[middle, 2] = middle[2]

        output_right[right, 1] = right[1]
        output_right[right, 2] = right[2]
    end

    for y in 1:(ny+2)
        for x in (1:nx+2)
            if x == 1 || y == 1 || x == nx + 2 || y == ny + 2
                CUDA.@allowscalar @test output_middle[x, y, 1] == -42
                CUDA.@allowscalar @test output_middle[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_middle[x, y, 1] == x
                CUDA.@allowscalar @test output_middle[x, y, 2] == y
            end

            if x == 1 || x == nx + 2 || y == ny + 1 || y == ny + 2
                CUDA.@allowscalar @test output_left[x, y, 1] == -42
                CUDA.@allowscalar @test output_left[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_left[x, y, 1] == x
                CUDA.@allowscalar @test output_left[x, y, 2] == y
            end
            if y == 1 || y == 2 || x == 1 || x == nx + 2
                CUDA.@allowscalar @test output_right[x, y, 1] == -42
                CUDA.@allowscalar @test output_right[x, y, 2] == -42
            else
                CUDA.@allowscalar @test output_right[x, y, 1] == x
                CUDA.@allowscalar @test output_right[x, y, 2] == y
            end
        end
    end
end
