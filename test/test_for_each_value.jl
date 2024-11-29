# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Test
using SinFVM
using StaticArrays

for backend in get_available_backends()
    values = collect(1:10)
    values_backend = SinFVM.convert_to_backend(backend, values)
    output = SinFVM.convert_to_backend(backend, zeros(10))

    SinFVM.@fvmloop SinFVM.for_each_index_value(backend, values_backend) do index, value
        output[index] = value * 2
    end

    output = collect(output)
    @test output == 2 .* values


    values_svector = [SVector{2,Float64}(i, 2 * i) for i in collect(1:10)]
    values_svector_backend = SinFVM.convert_to_backend(backend, values_svector)
    output_svector = SinFVM.convert_to_backend(backend, zeros(10))

    SinFVM.@fvmloop SinFVM.for_each_index_value(backend, values_svector_backend) do index, value
        output_svector[index] = value[2] * 2
    end

    output = collect(output_svector)
    @test output == 4 .* values
end
