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
    for gc in [1, 2]
        nx = 64
        ny = 32
        grid = VolumeFluxes.CartesianGrid(nx, ny; gc=gc)
        

        
        output = VolumeFluxes.convert_to_backend(backend, -42 * ones(Int64, nx + 2*gc, ny + 2*gc, 3))
        
        VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell(backend, grid) do index
            output[index,1] = index[1]
            output[index,2] = index[2]
            output[index,3] = index[1] + index[2]
            
        end

        for y in 1:(ny+2*gc)
            for x in (1:nx+2*gc)
                CUDA.@allowscalar @test output[x, y,1] == x
                CUDA.@allowscalar @test output[x, y,2] == y
                CUDA.@allowscalar @test output[x, y,3] == x+y
            end
        end
    end
end
