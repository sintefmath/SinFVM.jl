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
    nx = 10
    grid = SinFVM.CartesianGrid(nx)

    x_d = SinFVM.convert_to_backend(backend, fill(1.0, nx))
    y_d = SinFVM.convert_to_backend(backend, fill(2.0, nx))
    SinFVM.@fvmloop SinFVM.for_each_ghost_cell(backend, grid, XDIR) do i
        y_d[i] = x_d[i] - i
    end
    y = collect(y_d)
    @test y[1] == 0.0
    @test y[2:end] == 2.0 * ones(nx - 1)



    output_device = SinFVM.convert_to_backend(backend, -42 .* ones(Int64, nx + 2))
    SinFVM.@fvmloop SinFVM.for_each_ghost_cell(backend, grid, XDIR) do I
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

    grid2 = SinFVM.CartesianGrid(nx, gc=2)
    output_device = SinFVM.convert_to_backend(backend, -42 .* ones(Int64, nx + 4))
    SinFVM.@fvmloop SinFVM.for_each_ghost_cell(backend, grid2, XDIR) do I
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
