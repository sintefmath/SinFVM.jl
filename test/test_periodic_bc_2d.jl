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
using PrettyTables

@testset "$(backend_label(backend))" for backend in test_backends()
    nx = 11
    ny = 8

    for gc in [1, 2]
        grid = VolumeFluxes.CartesianGrid(nx, ny; gc=gc)
        bc = VolumeFluxes.PeriodicBC()
        equation = VolumeFluxes.Burgers()

        input = -42 * ones(nx + 2 * gc, ny + 2 * gc)
        for j in (gc+1):(ny+gc)
            for i in (gc+1):(nx+gc)
                input[i, j] = j * nx + i
            end
        end
        input_device = VolumeFluxes.convert_to_backend(backend, input)
        VolumeFluxes.update_bc!(backend, bc, grid, equation, input_device)
        output = collect(input_device)

        # pretty_table(output)

        for j in (gc+1):(ny+gc)
            for i in (gc+1):(nx+gc)
                @test output[i, j] == j * nx + i
            end
        end
        # @show gc
        # @show output
        for i in (gc+1):(nx+gc)
            for n in 1:gc
                # if output[i, n] != output[i, end-n]
                #     @info "Failed" n i output[i, n]
                # end
                @test output[i, n] == output[i, end-(gc-n+1)-gc+1]
                @test output[i, end-(gc-n)] == output[i, gc+n]
            end
        end

        for i in (gc+1):(ny+gc)
            for n in 1:gc
                # if output[n, i] != output[end-n, i]
                #     @info "Failed" n i output[n, i]
                # end
                @test output[n, i] == output[end-(gc-n+1)-gc+1, i]
                @test output[end-(gc-n), i] == output[gc+n, i]
            end
        end
    end
end
