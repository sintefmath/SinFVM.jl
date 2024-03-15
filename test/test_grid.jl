using SinSWE
using Test

nx = 10
grid = SinSWE.CartesianGrid(nx, extent=[0 nx])

x_faces = collect(0:nx)
x_cellcenters = collect(0.5:1:9.5)

@test SinSWE.cell_faces(grid) == x_faces
@test SinSWE.cell_centers(grid) == x_cellcenters


