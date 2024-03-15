using SinSWE
using Test

nx = 10
grid = SinSWE.CartesianGrid(nx, extent=[0 nx])

x_faces = collect(0:nx)
x_cellcenters = collect(0.5:1:9.5)

@test SinSWE.cell_faces(grid) == x_faces
@test SinSWE.cell_centers(grid) == x_cellcenters

const_b = 3.14
const_B = SinSWE.constant_bottom_topography(grid, const_b)

@test size(const_B)[1] == nx+2+1
for b in const_B
    @test b == const_b
end