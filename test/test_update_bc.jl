using SinSWE
using CUDA
using Test
using StaticArrays

nx = 10
grid = SinSWE.CartesianGrid(nx)
backend = make_cpu_backend()
equation = SinSWE.Burgers()

x = collect(1:(nx+2))

SinSWE.update_bc!(backend, grid, equation, x)
@test x[1] == 11
@test x[end] == 2
@test x[2:end-1] == collect(2:11)

xvec = [SVector{2, Float64}(i, 2*i) for i in 1:(nx+2)]
xvecorig = [SVector{2, Float64}(i, 2*i) for i in 1:(nx+2)]

SinSWE.update_bc!(backend, grid, equation, xvec)

@test xvec[1] == xvec[end-1]
@test xvec[end] == xvec[2]
@test xvec[2:end-1] == xvecorig[2:end-1]

## Test wall boundary condition for shallow water equations

wall_grid = SinSWE.CartesianGrid(nx, gc=2, boundary=SinSWE.WallBC())
swe = SinSWE.ShallowWaterEquations1D()

x = collect(1:(nx+4))
u     = [SVector{2, Float64}(x, x*10) for x in 1:(nx+4)]
uorig = [SVector{2, Float64}(x, x*10) for x in 1:(nx+4)]

SinSWE.update_bc!(backend, wall_grid, swe, u)

@test u[3:end-2] == uorig[3:end-2]
@test u[2][1] == u[3][1]
@test u[1][1] == u[4][1]
@test u[nx+4][1] == u[nx+1][1]
@test u[nx+3][1] == u[nx+2][1]
@test u[2][2] == -u[3][2]
@test u[1][2] == -u[4][2]
@test u[nx+4][2] == -u[nx+1][2]
@test u[nx+3][2] == -u[nx+2][2]

