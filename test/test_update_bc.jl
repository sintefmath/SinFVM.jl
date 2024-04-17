using SinSWE
using CUDA
using Test
using StaticArrays


for backend in [make_cpu_backend()] #TODO: Make test for CUDA
    nx = 10
    grid = SinSWE.CartesianGrid(nx)
    backend = make_cpu_backend()
    equation = SinSWE.Burgers()

    x = collect(1:(nx+2))

    SinSWE.update_bc!(backend, grid, equation, x)
    @test x[1] == 11
    @test x[end] == 2
    @test x[2:end-1] == collect(2:11)

    xvec = [SVector{2,Float64}(i, 2 * i) for i in 1:(nx+2)]
    xvecorig = [SVector{2,Float64}(i, 2 * i) for i in 1:(nx+2)]

    SinSWE.update_bc!(backend, grid, equation, xvec)

    @test xvec[1] == xvec[end-1]
    @test xvec[end] == xvec[2]
    @test xvec[2:end-1] == xvecorig[2:end-1]

    ## Test wall boundary condition for shallow water equations

    wall_grid = SinSWE.CartesianGrid(nx, gc=2, boundary=SinSWE.WallBC())
    swe = SinSWE.ShallowWaterEquations1D()

    x = collect(1:(nx+4))
    u = [SVector{2,Float64}(x, x * 10) for x in 1:(nx+4)]
    uorig = [SVector{2,Float64}(x, x * 10) for x in 1:(nx+4)]

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


    ## Test wall boundary condition for shallow water equations 2D

    ny = 5
    wall_grid_2d = SinSWE.CartesianGrid(nx, ny, gc=2, boundary=SinSWE.WallBC())
    swe_2d = SinSWE.ShallowWaterEquationsPure()
    u0 = x -> @SVector[x[1], (x[1] + x[2]) * 10, x[1] * (x[2] - 5)]

    x = SinSWE.cell_centers(wall_grid_2d; interior=false)
    u = u0.(x)
    uorig = u0.(x)

    SinSWE.update_bc!(backend, wall_grid_2d, swe_2d, u)

    @test u[3:end-2, 3:end-2] == uorig[3:end-2, 3:end-2]
    # h
    @show size(u[2, :])
    function f(u, i)
        [x[i] for x in u]
    end
    @show f(u[2, 3:end-2], 1)
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
