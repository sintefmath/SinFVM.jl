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
import CUDA
using Test

for backend in get_available_backends()
    B_const_default = SinFVM.ConstantBottomTopography()
    @test B_const_default.B == 0
    @test SinFVM.B_cell(B_const_default, 45) == 0
    @test SinFVM.B_face_left(B_const_default, 45, 12) == 0
    @test SinFVM.B_face_right(B_const_default, 45) == 0
    @test SinFVM.is_zero(B_const_default)

    B_const = SinFVM.ConstantBottomTopography(3.14)
    @test B_const.B == 3.14
    @test SinFVM.B_cell(B_const, 45) == 3.14
    @test SinFVM.B_face_left(B_const, 45, 12) == 3.14
    @test SinFVM.B_face_right(B_const, 45) == 3.14
    @test !SinFVM.is_zero(B_const)

    nx = 10
    grid = SinFVM.CartesianGrid(nx; gc=2, extent=[0.0  10] )

    B1_data = [x for x in SinFVM.cell_faces(grid, interior=false)]
    B1_data_zero = [0.0 for x in SinFVM.cell_faces(grid, interior=false)]
    # @show(B1_data)


    B1 = SinFVM.BottomTopography1D(B1_data, backend, grid)
    @test size(B1.B) == (nx + 5, )

    CUDA.@allowscalar @test SinFVM.B_cell(B1, 4+2) ≈ 3.5 atol=10^-10
    CUDA.@allowscalar @test SinFVM.B_cell(B1, 7+2) ≈ 6.5 atol=10^-10
    CUDA.@allowscalar @test SinFVM.B_face_right(B1, 4+2) ≈ 4 atol=10^-10
    CUDA.@allowscalar @test SinFVM.B_face_right(B1, 7+2) ≈ 7 atol=10^-10
    CUDA.@allowscalar @test SinFVM.B_face_left(B1, 4+2) ≈ 3 atol=10^-10
    CUDA.@allowscalar @test SinFVM.B_face_left(B1, 7+2) ≈ 6 atol=10^-10

    B1_bad_data = [x for x in SinFVM.cell_faces(grid)]
    @test_throws DomainError SinFVM.BottomTopography1D(B1_bad_data, backend, grid)

    atol = 10^-14
    @test SinFVM.collect_topography_intersections(B1, grid; interior=false) == B1_data
    @test SinFVM.collect_topography_intersections(B1, grid) == B1_data[3:end-2]
    @test SinFVM.collect_topography_cells(B1, grid; interior=false) ≈ [x for x in SinFVM.cell_centers(grid, interior=false)] atol=atol
    @test SinFVM.collect_topography_cells(B1, grid) ≈ [x for x in SinFVM.cell_centers(grid)] atol=atol

    @test SinFVM.collect_topography_intersections(B_const, grid; interior=false) == [3.14 for x in SinFVM.cell_faces(grid, interior=false)]
    @test SinFVM.collect_topography_intersections(B_const, grid) == [3.14 for x in SinFVM.cell_faces(grid, interior=true)]
    @test SinFVM.collect_topography_cells(B_const, grid; interior=false) == [3.14 for x in SinFVM.cell_centers(grid, interior=false)] 
    @test SinFVM.collect_topography_cells(B_const, grid) == [3.14 for x in SinFVM.cell_centers(grid)]

    B1_zero = SinFVM.BottomTopography1D(B1_data_zero, backend, grid)
    @test SinFVM.is_zero(B1_zero)
    @test !SinFVM.is_zero(B1)

    # TODO Test 2D
    nx = 2
    ny = 2
    grid2D = SinFVM.CartesianGrid(nx, ny; gc=2, extent=[0.0 12.0; 0.0 20.0])
    intersections = SinFVM.cell_faces(grid2D, interior=false)
    #@show intersections
    # B2_data = zeros(nx + 5, ny + 5)
    B2_data_zero = zeros(nx + 5, ny + 5)
    # for i in range(1,nx+5)
    #     for j in range(1, ny+5)
    #         B2_data[i, j] = intersections[i,j][1] + intersections[i,j][2]
    #     end
    # end
    B2_data = [x[1] + x[2] for x in SinFVM.cell_faces(grid2D, interior=false)]

    tol = 10^-10
    bottom2d = SinFVM.BottomTopography2D(B2_data, SinFVM.make_cpu_backend(), grid2D)
    CUDA.@allowscalar @test SinFVM.B_cell(bottom2d, 3, 3) ≈ 8 atol=tol
    CUDA.@allowscalar @test SinFVM.B_cell(bottom2d, 3, 4) ≈ 18 atol=tol
    CUDA.@allowscalar @test SinFVM.B_cell(bottom2d, CartesianIndex(4,3)) ≈ 14 atol=tol
    CUDA.@allowscalar @test SinFVM.B_cell(bottom2d, CartesianIndex(4,4)) ≈ 24 atol=tol

    CUDA.@allowscalar @test SinFVM.B_face_left( bottom2d, 3, 3, XDIR) ≈ 5 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 3, 3, XDIR) ≈ 11 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_left( bottom2d, 3, 3, YDIR) ≈ 3 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 3, 3, YDIR) ≈ 13 atol=tol 
    
    CUDA.@allowscalar @test SinFVM.B_face_left( bottom2d, 4, 4, XDIR) ≈ 21 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 4, 4, XDIR) ≈ 27 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_left( bottom2d, 4, 4, YDIR) ≈ 19 atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 4, 4, YDIR) ≈ 29 atol=tol 
    
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 3, 4, XDIR) ≈ SinFVM.B_face_left( bottom2d, 4, 4, XDIR) atol=tol 
    CUDA.@allowscalar @test SinFVM.B_face_right(bottom2d, 4, 3, YDIR) ≈ SinFVM.B_face_left( bottom2d, 4, 4, YDIR) atol=tol 

    atol = 10^-14
    @test SinFVM.collect_topography_intersections(bottom2d, grid2D; interior=false) == B2_data
    @test SinFVM.collect_topography_intersections(bottom2d, grid2D) == B2_data[3:end-2, 3:end-2]
    @test SinFVM.collect_topography_cells(bottom2d, grid2D; interior=false) ≈ [x[1] + x[2] for x in SinFVM.cell_centers(grid2D, interior=false)] atol=atol
    @test SinFVM.collect_topography_cells(bottom2d, grid2D) ≈ [x[1] + x[2] for x in SinFVM.cell_centers(grid2D)] atol=atol

    @test SinFVM.collect_topography_intersections(B_const, grid2D; interior=false) == [3.14 for x in SinFVM.cell_faces(grid2D, interior=false)]
    @test SinFVM.collect_topography_intersections(B_const, grid2D) == [3.14 for x in SinFVM.cell_faces(grid2D, interior=true)]
    @test SinFVM.collect_topography_cells(B_const, grid2D; interior=false) == [3.14 for x in SinFVM.cell_centers(grid2D, interior=false)] 
    @test SinFVM.collect_topography_cells(B_const, grid2D) == [3.14 for x in SinFVM.cell_centers(grid2D)]

    bottom2d_zero = SinFVM.BottomTopography2D(B2_data_zero, backend, grid2D)
    @test SinFVM.is_zero(bottom2d_zero)
    @test !SinFVM.is_zero(bottom2d)
end
