using SinFVM
using Test 
for backend in get_available_backends()
    nx = 10
    grid = SinFVM.CartesianGrid(nx)

    output_array = SinFVM.convert_to_backend(backend, zeros(nx + 2))

    SinFVM.@fvmloop SinFVM.for_each_cell(backend, grid) do index
        output_array[index] = index
    end

    @test collect(output_array) == 1:(nx+2)
end