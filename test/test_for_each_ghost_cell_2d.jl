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
    nx = 11
    ny = 8

    for gc in [1, 2]
        grid = VolumeFluxes.CartesianGrid(nx, ny; gc=gc)

        # XDIR
        output_device = VolumeFluxes.convert_to_backend(backend, -42 .* ones(Int64, nx + 2 * gc, ny + 2 * gc, 2))
        VolumeFluxes.@fvmloop VolumeFluxes.for_each_ghost_cell(backend, grid, XDIR) do I
            output_device[I, 1] = I[1]
            output_device[I, 2] = I[2]
        end
        output = collect(output_device)

        for i in 1:nx
            for j in 1:ny
                if i <= gc && (gc < j < (ny + gc))
                    @test output[i, j, 1] == i
                    @test output[i, j, 2] == j
                else
                    @test output[i, j, 1] == -42
                    @test output[i, j, 2] == -42
                end
            end
        end

        # YDIR
        output_device = VolumeFluxes.convert_to_backend(backend, -42 .* ones(Int64, nx + 2 * gc, ny + 2 * gc, 2))
        VolumeFluxes.@fvmloop VolumeFluxes.for_each_ghost_cell(backend, grid, YDIR) do I
            output_device[I, 1] = I[1]
            output_device[I, 2] = I[2]
        end
        output = collect(output_device)

        for i in 1:nx
            for j in 1:ny
                if j <= gc && (gc < i < (nx + gc))
                    @test output[i, j, 1] == i
                    @test output[i, j, 2] == j
                else
                    @test output[i, j, 1] == -42
                    @test output[i, j, 2] == -42
                end
            end
        end
    end
end
