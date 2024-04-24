using SinSWE
import CUDA
using Test

for backend in get_available_backends()
    B_const_default = SinSWE.ConstantBottomTopography()
    @test B_const_default.B == 0
    @test SinSWE.B_cell(B_const_default, 45) == 0
    @test SinSWE.B_face_left(B_const_default, 45, 12) == 0
    @test SinSWE.B_face_right(B_const_default, 45) == 0
    @test SinSWE.is_zero(B_const_default)

    B_const = SinSWE.ConstantBottomTopography(3.14)
    @test B_const.B == 3.14
    @test SinSWE.B_cell(B_const, 45) == 3.14
    @test SinSWE.B_face_left(B_const, 45, 12) == 3.14
    @test SinSWE.B_face_right(B_const, 45) == 3.14
    @test !SinSWE.is_zero(B_const)

    nx = 10
    grid = SinSWE.CartesianGrid(nx; gc=2, extent=[0.0  10] )

    B1_data = [x for x in SinSWE.cell_faces(grid, interior=false)]
    B1_data_zero = [0.0 for x in SinSWE.cell_faces(grid, interior=false)]
    # @show(B1_data)


    B1 = SinSWE.BottomTopography1D(B1_data, backend, grid)
    @test size(B1.B) == (nx + 5, )

    CUDA.@allowscalar @test SinSWE.B_cell(B1, 4+2) ≈ 3.5 atol=10^-10
    CUDA.@allowscalar @test SinSWE.B_cell(B1, 7+2) ≈ 6.5 atol=10^-10
    CUDA.@allowscalar @test SinSWE.B_face_right(B1, 4+2) ≈ 4 atol=10^-10
    CUDA.@allowscalar @test SinSWE.B_face_right(B1, 7+2) ≈ 7 atol=10^-10
    CUDA.@allowscalar @test SinSWE.B_face_left(B1, 4+2) ≈ 3 atol=10^-10
    CUDA.@allowscalar @test SinSWE.B_face_left(B1, 7+2) ≈ 6 atol=10^-10

    B1_bad_data = [x for x in SinSWE.cell_faces(grid)]
    @test_throws DomainError SinSWE.BottomTopography1D(B1_bad_data, backend, grid)

    atol = 10^-14
    @test SinSWE.collect_topography_intersections(B1, grid; interior=false) == B1_data
    @test SinSWE.collect_topography_intersections(B1, grid) == B1_data[3:end-2]
    @test SinSWE.collect_topography_cells(B1, grid; interior=false) ≈ [x for x in SinSWE.cell_centers(grid, interior=false)] atol=atol
    @test SinSWE.collect_topography_cells(B1, grid) ≈ [x for x in SinSWE.cell_centers(grid)] atol=atol

    @test SinSWE.collect_topography_intersections(B_const, grid; interior=false) == [3.14 for x in SinSWE.cell_faces(grid, interior=false)]
    @test SinSWE.collect_topography_intersections(B_const, grid) == [3.14 for x in SinSWE.cell_faces(grid, interior=true)]
    @test SinSWE.collect_topography_cells(B_const, grid; interior=false) == [3.14 for x in SinSWE.cell_centers(grid, interior=false)] 
    @test SinSWE.collect_topography_cells(B_const, grid) == [3.14 for x in SinSWE.cell_centers(grid)]

    B1_zero = SinSWE.BottomTopography1D(B1_data_zero, backend, grid)
    @test SinSWE.is_zero(B1_zero)
    @test !SinSWE.is_zero(B1)

    # TODO Test 2D
    nx = 2
    ny = 2
    grid2D = SinSWE.CartesianGrid(nx, ny; gc=2, extent=[0.0 12.0; 0.0 20.0])
    intersections = SinSWE.cell_faces(grid2D, interior=false)
    #@show intersections
    # B2_data = zeros(nx + 5, ny + 5)
    B2_data_zero = zeros(nx + 5, ny + 5)
    # for i in range(1,nx+5)
    #     for j in range(1, ny+5)
    #         B2_data[i, j] = intersections[i,j][1] + intersections[i,j][2]
    #     end
    # end
    B2_data = [x[1] + x[2] for x in SinSWE.cell_faces(grid2D, interior=false)]

    tol = 10^-10
    bottom2d = SinSWE.BottomTopography2D(B2_data, SinSWE.make_cpu_backend(), grid2D)
    CUDA.@allowscalar @test SinSWE.B_cell(bottom2d, 3, 3) ≈ 8 atol=tol
    CUDA.@allowscalar @test SinSWE.B_cell(bottom2d, 3, 4) ≈ 18 atol=tol
    CUDA.@allowscalar @test SinSWE.B_cell(bottom2d, CartesianIndex(4,3)) ≈ 14 atol=tol
    CUDA.@allowscalar @test SinSWE.B_cell(bottom2d, CartesianIndex(4,4)) ≈ 24 atol=tol

    CUDA.@allowscalar @test SinSWE.B_face_left( bottom2d, 3, 3, XDIR) ≈ 5 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 3, 3, XDIR) ≈ 11 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_left( bottom2d, 3, 3, YDIR) ≈ 3 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 3, 3, YDIR) ≈ 13 atol=tol 
    
    CUDA.@allowscalar @test SinSWE.B_face_left( bottom2d, 4, 4, XDIR) ≈ 21 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 4, 4, XDIR) ≈ 27 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_left( bottom2d, 4, 4, YDIR) ≈ 19 atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 4, 4, YDIR) ≈ 29 atol=tol 
    
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 3, 4, XDIR) ≈ SinSWE.B_face_left( bottom2d, 4, 4, XDIR) atol=tol 
    CUDA.@allowscalar @test SinSWE.B_face_right(bottom2d, 4, 3, YDIR) ≈ SinSWE.B_face_left( bottom2d, 4, 4, YDIR) atol=tol 

    atol = 10^-14
    @test SinSWE.collect_topography_intersections(bottom2d, grid2D; interior=false) == B2_data
    @test SinSWE.collect_topography_intersections(bottom2d, grid2D) == B2_data[3:end-2, 3:end-2]
    @test SinSWE.collect_topography_cells(bottom2d, grid2D; interior=false) ≈ [x[1] + x[2] for x in SinSWE.cell_centers(grid2D, interior=false)] atol=atol
    @test SinSWE.collect_topography_cells(bottom2d, grid2D) ≈ [x[1] + x[2] for x in SinSWE.cell_centers(grid2D)] atol=atol

    @test SinSWE.collect_topography_intersections(B_const, grid2D; interior=false) == [3.14 for x in SinSWE.cell_faces(grid2D, interior=false)]
    @test SinSWE.collect_topography_intersections(B_const, grid2D) == [3.14 for x in SinSWE.cell_faces(grid2D, interior=true)]
    @test SinSWE.collect_topography_cells(B_const, grid2D; interior=false) == [3.14 for x in SinSWE.cell_centers(grid2D, interior=false)] 
    @test SinSWE.collect_topography_cells(B_const, grid2D) == [3.14 for x in SinSWE.cell_centers(grid2D)]

    bottom2d_zero = SinSWE.BottomTopography2D(B2_data_zero, backend, grid2D)
    @test SinSWE.is_zero(bottom2d_zero)
    @test !SinSWE.is_zero(bottom2d)
end

