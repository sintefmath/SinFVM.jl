using SinSWE
import CUDA

B_const_default = SinSWE.ConstantBottomTopography()
@test B_const_default.B == 0
@test SinSWE.B_cell(B_const_default, 45) == 0
@test SinSWE.B_face_left(B_const_default, 45, 12) == 0
@test SinSWE.B_face_right(B_const_default, 45) == 0

B_const = SinSWE.ConstantBottomTopography(3.14)
@test B_const.B == 3.14
@test SinSWE.B_cell(B_const, 45) == 3.14
@test SinSWE.B_face_left(B_const, 45, 12) == 3.14
@test SinSWE.B_face_right(B_const, 45) == 3.14

nx = 10
grid = SinSWE.CartesianGrid(nx; gc=2, extent=[0.0  10] )

B1_data = [x for x in SinSWE.cell_faces(grid, interior=false)]
# @show(B1_data)

for backend in SinSWE.get_available_backends()
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
    
end



# TODO Test 2D
nx = 2
ny = 2
grid2D = SinSWE.CartesianGrid(nx, ny; gc=2, extent=[0.0 12.0; 0.0 20.0])
intersections = SinSWE.cell_faces(grid2D, interior=false)
#@show intersections
B2_data = zeros(nx + 5, ny + 5)
for i in range(1,nx+5)
    for j in range(1, ny+5)
        B2_data[i, j] = intersections[i,j][1] + intersections[i,j][2]
    end
end

tol = 10^-10
for backend in get_available_backends()
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
end




