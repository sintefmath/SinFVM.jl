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

using SinFVM
using Test 
for backend in get_available_backends()
    nx = 10
    grid = SinFVM.CartesianGrid(nx)

    output_array = SinFVM.convert_to_backend(backend, zeros(nx + 2))

    SinFVM.@fvmloop SinFVM.for_each_cell(backend, grid) do index
        output_array[index] = index
    end

    @test collect(output_array) == 1:(nx+2)
end
