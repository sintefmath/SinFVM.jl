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

for backend in get_available_backends()
    nx = 11
    ny = 8

    for gc in [1, 2]
        grid = SinFVM.CartesianGrid(nx, ny; gc=gc)

        # XDIR
        output_device = SinFVM.convert_to_backend(backend, -42 .* ones(Int64, nx + 2 * gc, ny + 2 * gc, 2))
        SinFVM.@fvmloop SinFVM.for_each_ghost_cell(backend, grid, XDIR) do I
            output_device[I, 1] = I[1]
            output_device[I, 2] = I[2]
        end
        output = collect(output_device)

        for i in 1:nx
            for j in 1:ny
                if i <= gc && (gc < j < (ny + gc))
                    @test output[i, j, 1] == i
                    @test output[i, j, 2] == j
                else
                    @test output[i, j, 1] == -42
                    @test output[i, j, 2] == -42
                end
            end
        end

        # YDIR
        output_device = SinFVM.convert_to_backend(backend, -42 .* ones(Int64, nx + 2 * gc, ny + 2 * gc, 2))
        SinFVM.@fvmloop SinFVM.for_each_ghost_cell(backend, grid, YDIR) do I
            output_device[I, 1] = I[1]
            output_device[I, 2] = I[2]
        end
        output = collect(output_device)

        for i in 1:nx
            for j in 1:ny
                if j <= gc && (gc < i < (nx + gc))
                    @test output[i, j, 1] == i
                    @test output[i, j, 2] == j
                else
                    @test output[i, j, 1] == -42
                    @test output[i, j, 2] == -42
                end
            end
        end
    end
end
