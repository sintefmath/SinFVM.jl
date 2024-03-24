using SinSWE
using CUDA
using Test



for backend in get_available_backends()
    nx = 10
    grid = SinSWE.CartesianGrid(nx)

    x_d = SinSWE.convert_to_backend(backend, fill(1.0, nx))
    y_d = SinSWE.convert_to_backend(backend, fill(2.0, nx))
    SinSWE.@fvmloop SinSWE.for_each_ghost_cell(backend, grid, XDIR) do i
        y_d[i] = x_d[i] - i
    end
    y = collect(y_d)
    @test y[1] == 0.0
    @test y[2:end] == 2.0 * ones(nx - 1)



    output_device = SinSWE.convert_to_backend(backend, -42 .* ones(Int64, nx + 2))
    SinSWE.@fvmloop SinSWE.for_each_ghost_cell(backend, grid, XDIR) do I
        output_device[I] = I[1]
    end
    output = collect(output_device)

    for i in 1:nx
        if i == 1
            @test output[i, 1] == i
        else
            @test output[i, 1] == -42
        end
    end

    grid2 = SinSWE.CartesianGrid(nx, gc=2)
    output_device = SinSWE.convert_to_backend(backend, -42 .* ones(Int64, nx + 4))
    SinSWE.@fvmloop SinSWE.for_each_ghost_cell(backend, grid2, XDIR) do I
        output_device[I] = I[1]
    end
    output = collect(output_device)

    for i in 1:nx
        if i == 1 || i == 2
            @test output[i, 1] == i
        else
            @test output[i, 1] == -42
        end
    end
end