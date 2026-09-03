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
    for gc in [1, 2]
        grid = VolumeFluxes.CartesianGrid(nx; gc=gc)
        bc = VolumeFluxes.PeriodicBC()
        equation = VolumeFluxes.Burgers()

        input = -42 * ones(nx + 2 * gc)
        for i in (gc+1):(nx+gc)
            input[i] = i
        end
        # pretty_table(input)

        input_device = VolumeFluxes.convert_to_backend(backend, input)
        VolumeFluxes.update_bc!(backend, bc, grid, equation, input_device)
        output = collect(input_device)

        # pretty_table(output)

        for i in (gc+1):(nx+gc)
            @test output[i] == i
        end
        # @show gc
        # @show output
        for n in 1:gc
            @show n
            @show (gc - n + 1)
            @show output[end-(gc-n+1)]
            @test output[n] == output[end-(gc-n+1)-gc+1]
        end
    end
end
