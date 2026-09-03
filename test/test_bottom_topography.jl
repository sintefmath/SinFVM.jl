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

using VolumeFluxes
isdefined(Main, :test_backends) || include("testing_utils.jl")
import CUDA
using Test

@testset "$(backend_label(backend))" for backend in test_backends()
    B_const_default = VolumeFluxes.ConstantBottomTopography()
    @test B_const_default.B == 0
    @test VolumeFluxes.B_cell(B_const_default, 45) == 0
    @test VolumeFluxes.B_face_left(B_const_default, 45, 12) == 0
    @test VolumeFluxes.B_face_right(B_const_default, 45) == 0
    @test VolumeFluxes.is_zero(B_const_default)

    B_const = VolumeFluxes.ConstantBottomTopography(3.14)
    @test B_const.B == 3.14
    @test VolumeFluxes.B_cell(B_const, 45) == 3.14
    @test VolumeFluxes.B_face_left(B_const, 45, 12) == 3.14
    @test VolumeFluxes.B_face_right(B_const, 45) == 3.14
    @test !VolumeFluxes.is_zero(B_const)

    nx = 10
    grid = VolumeFluxes.CartesianGrid(nx; gc=2, extent=[0.0  10] )

    B1_data = [x for x in VolumeFluxes.cell_faces(grid, interior=false)]
    B1_data_zero = [0.0 for x in VolumeFluxes.cell_faces(grid, interior=false)]
    # @show(B1_data)


    B1 = VolumeFluxes.BottomTopography1D(B1_data, backend, grid)
    @test size(B1.B) == (nx + 5, )

    CUDA.@allowscalar @test VolumeFluxes.B_cell(B1, 4+2) ≈ 3.5 atol=test_atol(backend, 10^-10)
    CUDA.@allowscalar @test VolumeFluxes.B_cell(B1, 7+2) ≈ 6.5 atol=test_atol(backend, 10^-10)
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(B1, 4+2) ≈ 4 atol=test_atol(backend, 10^-10)
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(B1, 7+2) ≈ 7 atol=test_atol(backend, 10^-10)
    CUDA.@allowscalar @test VolumeFluxes.B_face_left(B1, 4+2) ≈ 3 atol=test_atol(backend, 10^-10)
    CUDA.@allowscalar @test VolumeFluxes.B_face_left(B1, 7+2) ≈ 6 atol=test_atol(backend, 10^-10)

    B1_bad_data = [x for x in VolumeFluxes.cell_faces(grid)]
    @test_throws DomainError VolumeFluxes.BottomTopography1D(B1_bad_data, backend, grid)

    atol = test_atol(backend, 10^-14)
    # `==` against a Float64 reference cannot hold for a Float32 backend, and this is
    # testing the gather, not the arithmetic -- compare with a tolerance instead.
    @test VolumeFluxes.collect_topography_intersections(B1, grid; interior=false) ≈ B1_data atol=atol
    @test VolumeFluxes.collect_topography_intersections(B1, grid) ≈ B1_data[3:end-2] atol=atol
    @test VolumeFluxes.collect_topography_cells(B1, grid; interior=false) ≈ [x for x in VolumeFluxes.cell_centers(grid, interior=false)] atol=atol
    @test VolumeFluxes.collect_topography_cells(B1, grid) ≈ [x for x in VolumeFluxes.cell_centers(grid)] atol=atol

    @test VolumeFluxes.collect_topography_intersections(B_const, grid; interior=false) == [3.14 for x in VolumeFluxes.cell_faces(grid, interior=false)]
    @test VolumeFluxes.collect_topography_intersections(B_const, grid) == [3.14 for x in VolumeFluxes.cell_faces(grid, interior=true)]
    @test VolumeFluxes.collect_topography_cells(B_const, grid; interior=false) == [3.14 for x in VolumeFluxes.cell_centers(grid, interior=false)] 
    @test VolumeFluxes.collect_topography_cells(B_const, grid) == [3.14 for x in VolumeFluxes.cell_centers(grid)]

    B1_zero = VolumeFluxes.BottomTopography1D(B1_data_zero, backend, grid)
    @test VolumeFluxes.is_zero(B1_zero)
    @test !VolumeFluxes.is_zero(B1)

    # TODO Test 2D
    nx = 2
    ny = 2
    grid2D = VolumeFluxes.CartesianGrid(nx, ny; gc=2, extent=[0.0 12.0; 0.0 20.0])
    intersections = VolumeFluxes.cell_faces(grid2D, interior=false)
    #@show intersections
    # B2_data = zeros(nx + 5, ny + 5)
    B2_data_zero = zeros(nx + 5, ny + 5)
    # for i in range(1,nx+5)
    #     for j in range(1, ny+5)
    #         B2_data[i, j] = intersections[i,j][1] + intersections[i,j][2]
    #     end
    # end
    B2_data = [x[1] + x[2] for x in VolumeFluxes.cell_faces(grid2D, interior=false)]

    tol = 10^-10
    bottom2d = VolumeFluxes.BottomTopography2D(B2_data, VolumeFluxes.make_cpu_backend(), grid2D)
    CUDA.@allowscalar @test VolumeFluxes.B_cell(bottom2d, 3, 3) ≈ 8 atol=tol
    CUDA.@allowscalar @test VolumeFluxes.B_cell(bottom2d, 3, 4) ≈ 18 atol=tol
    CUDA.@allowscalar @test VolumeFluxes.B_cell(bottom2d, CartesianIndex(4,3)) ≈ 14 atol=tol
    CUDA.@allowscalar @test VolumeFluxes.B_cell(bottom2d, CartesianIndex(4,4)) ≈ 24 atol=tol

    CUDA.@allowscalar @test VolumeFluxes.B_face_left( bottom2d, 3, 3, XDIR) ≈ 5 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 3, 3, XDIR) ≈ 11 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_left( bottom2d, 3, 3, YDIR) ≈ 3 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 3, 3, YDIR) ≈ 13 atol=tol 
    
    CUDA.@allowscalar @test VolumeFluxes.B_face_left( bottom2d, 4, 4, XDIR) ≈ 21 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 4, 4, XDIR) ≈ 27 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_left( bottom2d, 4, 4, YDIR) ≈ 19 atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 4, 4, YDIR) ≈ 29 atol=tol 
    
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 3, 4, XDIR) ≈ VolumeFluxes.B_face_left( bottom2d, 4, 4, XDIR) atol=tol 
    CUDA.@allowscalar @test VolumeFluxes.B_face_right(bottom2d, 4, 3, YDIR) ≈ VolumeFluxes.B_face_left( bottom2d, 4, 4, YDIR) atol=tol 

    atol = 10^-14
    @test VolumeFluxes.collect_topography_intersections(bottom2d, grid2D; interior=false) == B2_data
    @test VolumeFluxes.collect_topography_intersections(bottom2d, grid2D) == B2_data[3:end-2, 3:end-2]
    @test VolumeFluxes.collect_topography_cells(bottom2d, grid2D; interior=false) ≈ [x[1] + x[2] for x in VolumeFluxes.cell_centers(grid2D, interior=false)] atol=atol
    @test VolumeFluxes.collect_topography_cells(bottom2d, grid2D) ≈ [x[1] + x[2] for x in VolumeFluxes.cell_centers(grid2D)] atol=atol

    @test VolumeFluxes.collect_topography_intersections(B_const, grid2D; interior=false) == [3.14 for x in VolumeFluxes.cell_faces(grid2D, interior=false)]
    @test VolumeFluxes.collect_topography_intersections(B_const, grid2D) == [3.14 for x in VolumeFluxes.cell_faces(grid2D, interior=true)]
    @test VolumeFluxes.collect_topography_cells(B_const, grid2D; interior=false) == [3.14 for x in VolumeFluxes.cell_centers(grid2D, interior=false)] 
    @test VolumeFluxes.collect_topography_cells(B_const, grid2D) == [3.14 for x in VolumeFluxes.cell_centers(grid2D)]

    bottom2d_zero = VolumeFluxes.BottomTopography2D(B2_data_zero, backend, grid2D)
    @test VolumeFluxes.is_zero(bottom2d_zero)
    @test !VolumeFluxes.is_zero(bottom2d)
end
