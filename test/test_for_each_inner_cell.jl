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

    # Device arrays: a kernel cannot write into a host `Vector`. This file built plain
    # host arrays, which went unnoticed while it only ever ran on the CPU.
    leftarrays_d = to_backend(backend, 1000 * ones(nx + 2))
    middlearrays_d = to_backend(backend, 1000 * ones(nx + 2))
    rightarrays_d = to_backend(backend, 1000 * ones(nx + 2))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR) do ileft, imiddle, iright
        leftarrays_d[imiddle] = ileft
        middlearrays_d[imiddle] = imiddle
        rightarrays_d[imiddle] = iright
    end

    leftarrays = collect(leftarrays_d)
    middlearrays = collect(middlearrays_d)
    rightarrays = collect(rightarrays_d)

    @test leftarrays[1] == 1000
    @test middlearrays[1] == 1000
    @test rightarrays[1] == 1000


    @test leftarrays[end] == 1000
    @test middlearrays[end] == 1000
    @test rightarrays[end] == 1000

    @test leftarrays[2:end-1] == 1:(nx)
    @test middlearrays[2:end-1] == 2:(nx+1)
    @test rightarrays[2:end-1] == 3:(nx+2)


    ## Check for ghost cells


    leftarrays_g = to_backend(backend, 1000 * ones(nx + 2))
    middlearrays_g = to_backend(backend, 1000 * ones(nx + 2))
    rightarrays_g = to_backend(backend, 1000 * ones(nx + 2))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_inner_cell(backend, grid, XDIR; ghostcells=3) do ileft, imiddle, iright
        leftarrays_g[imiddle] = ileft
        middlearrays_g[imiddle] = imiddle
        rightarrays_g[imiddle] = iright
    end

    leftarrays = collect(leftarrays_g)
    middlearrays = collect(middlearrays_g)
    rightarrays = collect(rightarrays_g)

    for i in 1:3
        @test leftarrays[i] == 1000
        @test middlearrays[i] == 1000
        @test rightarrays[i] == 1000


        @test leftarrays[end-i+1] == 1000
        @test middlearrays[end-i+1] == 1000
        @test rightarrays[end-i+1] == 1000
    end
    @test leftarrays[4:end-3] == 3:(nx-2)
    @test middlearrays[4:end-3] == 4:(nx-1)
    @test rightarrays[4:end-3] == 5:(nx)
end
