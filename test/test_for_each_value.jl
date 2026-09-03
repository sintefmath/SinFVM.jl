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
isdefined(Main, :test_backends) || include("testing_utils.jl")
using StaticArrays

@testset "$(backend_label(backend))" for backend in test_backends()
    values = collect(1:10)
    values_backend = VolumeFluxes.convert_to_backend(backend, values)
    output = VolumeFluxes.convert_to_backend(backend, zeros(10))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_index_value(backend, values_backend) do index, value
        output[index] = value * 2
    end

    output = collect(output)
    @test output == 2 .* values


    values_svector = [SVector{2,Float64}(i, 2 * i) for i in collect(1:10)]
    values_svector_backend = VolumeFluxes.convert_to_backend(backend, values_svector)
    output_svector = VolumeFluxes.convert_to_backend(backend, zeros(10))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_index_value(backend, values_svector_backend) do index, value
        output_svector[index] = value[2] * 2
    end

    output = collect(output_svector)
    @test output == 4 .* values
end
