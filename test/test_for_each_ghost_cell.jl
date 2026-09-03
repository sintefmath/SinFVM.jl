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
    nx = 10
    grid = VolumeFluxes.CartesianGrid(nx)

    x_d = VolumeFluxes.convert_to_backend(backend, fill(1.0, nx))
    y_d = VolumeFluxes.convert_to_backend(backend, fill(2.0, nx))
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_ghost_cell(backend, grid, XDIR) do i
        y_d[i] = x_d[i] - i
    end
    y = collect(y_d)
    @test y[1] == 0.0
    @test y[2:end] == 2.0 * ones(nx - 1)



    output_device = VolumeFluxes.convert_to_backend(backend, -42 .* ones(Int64, nx + 2))
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_ghost_cell(backend, grid, XDIR) do I
        output_device[I] = I[1]
    end
    output = collect(output_device)

    for i in 1:nx
        if i == 1
            @test output[i, 1] == i
        else
            @test output[i, 1] == -42
        end
    end

    grid2 = VolumeFluxes.CartesianGrid(nx, gc=2)
    output_device = VolumeFluxes.convert_to_backend(backend, -42 .* ones(Int64, nx + 4))
    VolumeFluxes.@fvmloop VolumeFluxes.for_each_ghost_cell(backend, grid2, XDIR) do I
        output_device[I] = I[1]
    end
    output = collect(output_device)

    for i in 1:nx
        if i == 1 || i == 2
            @test output[i, 1] == i
        else
            @test output[i, 1] == -42
        end
    end
end
