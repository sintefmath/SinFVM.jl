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
using CUDA
using Test
using PrettyTables

for backend in get_available_backends()
    nx = 11
    for gc in [1, 2]
        grid = SinFVM.CartesianGrid(nx; gc=gc)
        bc = SinFVM.PeriodicBC()
        equation = SinFVM.Burgers()

        input = -42 * ones(nx + 2 * gc)
        for i in (gc+1):(nx+gc)
            input[i] = i
        end
        # pretty_table(input)

        input_device = SinFVM.convert_to_backend(backend, input)
        SinFVM.update_bc!(backend, bc, grid, equation, input_device)
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
