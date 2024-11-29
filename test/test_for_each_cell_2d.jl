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
    for gc in [1, 2]
        nx = 64
        ny = 32
        grid = SinFVM.CartesianGrid(nx, ny; gc=gc)
        

        
        output = SinFVM.convert_to_backend(backend, -42 * ones(Int64, nx + 2*gc, ny + 2*gc, 3))
        
        SinFVM.@fvmloop SinFVM.for_each_cell(backend, grid) do index
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
