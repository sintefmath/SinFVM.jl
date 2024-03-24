using SinSWE
using CUDA
using Test

for backend in get_available_backends()
    nx = 11
    ny = 8

    for gc in [1, 2]
        grid = SinSWE.CartesianGrid(nx, ny; gc=gc)
        bc = SinSWE.PeriodicBC()
        equation = SinSWE.Burgers()

        input = -42 * ones(nx + 2 * gc, ny + 2 * gc)
        for j in (2*gc):(ny+gc)
            for i in (2*gc):(nx+gc)
                input[i, j] = j * nx + i
            end
        end
        input_device = SinSWE.convert_to_backend(backend, input)
        SinSWE.update_bc!(backend, bc, grid, equation, input_device)
        output = collect(input_device)

        for j in (2*gc):(ny+gc)
            for i in (2*gc):(nx+gc)
                @test output[i, j] == j * nx + i
            end
        end
        @show gc
        @show output
        for i in (2*gc):(nx+gc)
            for n in 1:gc
                # if output[i, n] != output[i, end-n]
                #     @info "Failed" n i output[i, n]
                # end
                @test output[i, n] == output[i, end-(gc-n+1)]
            end
        end

        for i in (2*gc):(ny+gc)
            for n in 1:gc
                # if output[n, i] != output[end-n, i]
                #     @info "Failed" n i output[n, i]
                # end
                @test output[n, i] == output[end-(gc-n+1), i]
            end
        end
    end
end