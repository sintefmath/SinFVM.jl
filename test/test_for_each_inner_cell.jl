using SinFVM
using CUDA
using Test

for backend in get_available_backends()
    nx = 10
    grid = SinFVM.CartesianGrid(nx)
    backend = make_cpu_backend()

    leftarrays = 1000 * ones(nx + 2)
    middlearrays = 1000 * ones(nx + 2)
    rightarrays = 1000 * ones(nx + 2)

    SinFVM.@fvmloop SinFVM.for_each_inner_cell(backend, grid, XDIR) do ileft, imiddle, iright
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


    ## Check for ghost cells


    leftarrays = 1000 * ones(nx + 2)
    middlearrays = 1000 * ones(nx + 2)
    rightarrays = 1000 * ones(nx + 2)

    SinFVM.@fvmloop SinFVM.for_each_inner_cell(backend, grid, XDIR; ghostcells=3) do ileft, imiddle, iright
        leftarrays[imiddle] = ileft
        middlearrays[imiddle] = imiddle
        rightarrays[imiddle] = iright
    end

    for i in 1:3
        @test leftarrays[i] == 1000
        @test middlearrays[i] == 1000
        @test rightarrays[i] == 1000


        @test leftarrays[end-i+1] == 1000
        @test middlearrays[end-i+1] == 1000
        @test rightarrays[end-i+1] == 1000
    end
    @test leftarrays[4:end-3] == 3:(nx-2)
    @test middlearrays[4:end-3] == 4:(nx-1)
    @test rightarrays[4:end-3] == 5:(nx)
end