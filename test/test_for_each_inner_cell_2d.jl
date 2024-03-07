using SinSWE
using CUDA
using Test


nx = 10
ny = 9
grid = SinSWE.CartesianGrid(nx, ny)
