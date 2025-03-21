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
    ny = 8

    for gc in [1, 2]
        grid = SinFVM.CartesianGrid(nx, ny; gc=gc)
        bc = SinFVM.PeriodicBC()
        equation = SinFVM.Burgers()

        input = -42 * ones(nx + 2 * gc, ny + 2 * gc)
        for j in (gc+1):(ny+gc)
            for i in (gc+1):(nx+gc)
                input[i, j] = j * nx + i
            end
        end
        input_device = SinFVM.convert_to_backend(backend, input)
        SinFVM.update_bc!(backend, bc, grid, equation, input_device)
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
