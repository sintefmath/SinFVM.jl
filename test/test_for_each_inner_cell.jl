using SinSWE
using CUDA
using Test


nx = 10
grid = SinSWE.CartesianGrid(nx)
backend = make_cpu_backend()

leftarrays = 1000*ones(nx + 2)
middlearrays = 1000*ones(nx + 2)
rightarrays = 1000*ones(nx + 2)

SinSWE.@fvmloop SinSWE.for_each_inner_cell(backend, grid, XDIR) do ileft, imiddle, iright
    leftarrays[imiddle] = ileft
    middlearrays[imiddle] = imiddle
    rightarrays[imiddle] = iright
end

@test leftarrays[1] == 1000
@test middlearrays[1] == 1000
@test rightarrays[1] == 1000


@test leftarrays[end] == 1000
@test middlearrays[end] == 1000
@test rightarrays[end] == 1000

@test leftarrays[2:end-1] == 1:(nx)
@test middlearrays[2:end-1] == 2:(nx+1)
@test rightarrays[2:end-1] == 3:(nx+2)