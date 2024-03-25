using SinSWE
using CUDA
using Test
using PrettyTables

for backend in get_available_backends()
    nx = 11
    for gc in [1, 2]
        grid = SinSWE.CartesianGrid(nx; gc=gc)
        bc = SinSWE.PeriodicBC()
        equation = SinSWE.Burgers()

        input = -42 * ones(nx + 2 * gc)
        for i in (gc+1):(nx+gc)
            input[i] = i
        end
        # pretty_table(input)

        input_device = SinSWE.convert_to_backend(backend, input)
        SinSWE.update_bc!(backend, bc, grid, equation, input_device)
        output = collect(input_device)

        # pretty_table(output)

        for i in (gc+1):(nx+gc)
            @test output[i] == i
        end
        # @show gc
        # @show output
        for n in 1:gc
            @show n
            @show (gc - n + 1)
            @show output[end-(gc-n+1)]
            @test output[n] == output[end-(gc-n+1)-gc+1]
        end
    end
end