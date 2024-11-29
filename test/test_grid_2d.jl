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
using StaticArrays
 # We do not technically need this loop, but adding it to create a scope
for _ in get_available_backends()
    nx = 10
    ny = 6
    gc = 2

    grid = SinFVM.CartesianGrid(nx, ny, extent=[0 nx; 0 ny], gc=gc)
    dx = SinFVM.compute_dx(grid)
    dy = SinFVM.compute_dy(grid)

    x_faces = collect(0:nx)
    y_faces = collect(0:ny)
    x_cellcenters = collect(0.5:1:9.5)
    y_cellcenters = collect(0.5:1:5.5)

    faces_from_grid = SinFVM.cell_faces(grid)
    centers_from_grid = SinFVM.cell_centers(grid)
    for j in 1:(ny+1)
        for i in 1:(nx+1)
            @test faces_from_grid[i, j][1] == x_faces[i]
            @test faces_from_grid[i, j][2] == y_faces[j]
        end
    end
    for j in 1:(ny)
        for i in 1:(nx)
            @test centers_from_grid[i, j][1] == x_cellcenters[i]
            @test centers_from_grid[i, j][2] == y_cellcenters[j]
        end
    end

    @test x_faces == SinFVM.cell_faces(grid, XDIR)
    @test y_faces == SinFVM.cell_faces(grid, YDIR)
    @test x_cellcenters == SinFVM.cell_centers(grid, XDIR)
    @test y_cellcenters == SinFVM.cell_centers(grid, YDIR)

    faces_gc = SinFVM.cell_faces(grid, interior=false)
    @test faces_gc[3:end-2, 3:end-2] ≈ faces_from_grid atol = 10^-14 # Got machine epsilon round-off errors
    for j in 1:2
        for i in 1:2
            @test faces_gc[i, j] ≈ SVector{2, Float64}(-3+i, -3+j) atol = 10^-14
            @test faces_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i +1), ny+ (gc-j+1)) atol = 10^-14
        end
    end

    cells_gc = SinFVM.cell_centers(grid, interior=false)
    @test cells_gc[3:end-2, 3:end-2] ≈ centers_from_grid atol = 10^-14
    for j in 1:2
        for i in 1:2
            @test cells_gc[i, j] ≈ SVector{2, Float64}(-3+i + dx/2, -3+j + dy/2)  atol = 10^-14
            @test cells_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i) + dx/2, ny+ (gc-j)+dy/2) atol = 10^-14
        end
    end

end
