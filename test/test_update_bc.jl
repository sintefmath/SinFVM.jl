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
using CUDA
using Test
using StaticArrays


@testset "$(backend_label(backend))" for backend in test_backends()
    nx = 10
    grid = VolumeFluxes.CartesianGrid(nx)
    equation = VolumeFluxes.Burgers()

    # `update_bc!` launches a kernel, so its data has to live on the backend and the
    # assertions have to work on a collected host copy. This file used host arrays
    # throughout, which went unnoticed while it only ever ran on the CPU.
    x_d = to_backend(backend, collect(1:(nx+2)))
    VolumeFluxes.update_bc!(backend, grid, equation, x_d)
    x = collect(x_d)
    @test x[1] == 11
    @test x[end] == 2
    @test x[2:end-1] == collect(2:11)

    xvec_d = to_backend(backend, [SVector{2,Float64}(i, 2 * i) for i in 1:(nx+2)])
    xvecorig = collect(xvec_d)

    VolumeFluxes.update_bc!(backend, grid, equation, xvec_d)
    xvec = collect(xvec_d)

    @test xvec[1] == xvec[end-1]
    @test xvec[end] == xvec[2]
    @test xvec[2:end-1] == xvecorig[2:end-1]

    ## Test wall boundary condition for shallow water equations

    wall_grid = VolumeFluxes.CartesianGrid(nx, gc=2, boundary=VolumeFluxes.WallBC())
    swe = backend_params(backend, VolumeFluxes.ShallowWaterEquations1D())

    u_d = to_backend(backend, [SVector{2,Float64}(x, x * 10) for x in 1:(nx+4)])
    uorig = collect(u_d)

    VolumeFluxes.update_bc!(backend, wall_grid, swe, u_d)
    u = collect(u_d)

    @test u[3:end-2] == uorig[3:end-2]
    @test u[2][1] == u[3][1]
    @test u[1][1] == u[4][1]
    @test u[nx+4][1] == u[nx+1][1]
    @test u[nx+3][1] == u[nx+2][1]
    @test u[2][2] == -u[3][2]
    @test u[1][2] == -u[4][2]
    @test u[nx+4][2] == -u[nx+1][2]
    @test u[nx+3][2] == -u[nx+2][2]


    ## Test wall boundary condition for shallow water equations 2D

    ny = 5
    wall_grid_2d = VolumeFluxes.CartesianGrid(nx, ny, gc=2, boundary=VolumeFluxes.WallBC())
    swe_2d = backend_params(backend, VolumeFluxes.ShallowWaterEquationsPure())
    u0 = x -> @SVector[x[1], (x[1] + x[2]) * 10, x[1] * (x[2] - 5)]

    x = VolumeFluxes.cell_centers(wall_grid_2d; interior=false)
    u_d = to_backend(backend, u0.(x))
    uorig = collect(u_d)

    VolumeFluxes.update_bc!(backend, wall_grid_2d, swe_2d, u_d)
    u = collect(u_d)

    @test u[3:end-2, 3:end-2] == uorig[3:end-2, 3:end-2]
    # h
    function f(u, i)
        [x[i] for x in u]
    end
    @test f(u[2, 3:end-2], 1) == f(u[3, 3:end-2], 1)
    @test f(u[1, 3:end-2], 1) == f(u[4, 3:end-2], 1)
    @test f(u[nx+4, 3:end-2], 1) == f(u[nx+1, 3:end-2], 1)
    @test f(u[nx+3, 3:end-2], 1) == f(u[nx+2, 3:end-2], 1)
    @test f(u[3:end-2, 2], 1) == f(u[3:end-2, 3], 1)
    @test f(u[3:end-2, 1], 1) == f(u[3:end-2, 4], 1)
    @test f(u[3:end-2, ny+4], 1) == f(u[3:end-2, ny+1], 1)
    @test f(u[3:end-2, ny+3], 1) == f(u[3:end-2, ny+2], 1)
    # hu
    @test f(u[2, 3:end-2], 2) == -f(u[3, 3:end-2], 2)
    @test f(u[1, 3:end-2], 2) == -f(u[4, 3:end-2], 2)
    @test f(u[nx+4, 3:end-2], 2) == -f(u[nx+1, 3:end-2], 2)
    @test f(u[nx+3, 3:end-2], 2) == -f(u[nx+2, 3:end-2], 2)
    @test f(u[3:end-2, 2], 2) == f(u[3:end-2, 3], 2)
    @test f(u[3:end-2, 1], 2) == f(u[3:end-2, 4], 2)
    @test f(u[3:end-2, ny+4], 2) == f(u[3:end-2, ny+1], 2)
    @test f(u[3:end-2, ny+3], 2) == f(u[3:end-2, ny+2], 2)
    # hv
    @test f(u[2, 3:end-2], 3) == f(u[3, 3:end-2], 3)
    @test f(u[1, 3:end-2], 3) == f(u[4, 3:end-2], 3)
    @test f(u[nx+4, 3:end-2], 3) == f(u[nx+1, 3:end-2], 3)
    @test f(u[nx+3, 3:end-2], 3) == f(u[nx+2, 3:end-2], 3)
    @test f(u[3:end-2, 2], 3) == -f(u[3:end-2, 3], 3)
    @test f(u[3:end-2, 1], 3) == -f(u[3:end-2, 4], 3)
    @test f(u[3:end-2, ny+4], 3) == -f(u[3:end-2, ny+1], 3)
    @test f(u[3:end-2, ny+3], 3) == -f(u[3:end-2, ny+2], 3)
end
