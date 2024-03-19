using SinSWE
using CUDA
using Test


nx = 10
ny = 9
grid = SinSWE.CartesianGrid(nx, ny)
@test SinSWE.compute_dx(grid, XDIR) == 1.0 / nx
@test SinSWE.compute_dx(grid, YDIR) == 1.0 / ny
