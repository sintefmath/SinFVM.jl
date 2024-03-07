using SinSWE
using CUDA
using Test
using StaticArrays

nx = 10
grid = SinSWE.CartesianGrid(nx)
backend = make_cpu_backend()

x = collect(1:(nx+2))

SinSWE.update_bc!(backend, grid, x)
@test x[1] == 11
@test x[end] == 2
@test x[2:end-1] == collect(2:11)

xvec = [SVector{2, Float64}(i, 2*i) for i in 1:(nx+2)]
xvecorig = [SVector{2, Float64}(i, 2*i) for i in 1:(nx+2)]

SinSWE.update_bc!(backend, grid, xvec)

@test xvec[1] == xvec[end-1]
@test xvec[end] == xvec[2]
@test xvec[2:end-1] == xvecorig[2:end-1]