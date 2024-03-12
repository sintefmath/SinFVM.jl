using Test
using SinSWE
@show get_available_backends()

for backend in get_available_backends()
    values = collect(1:10)
    values_backend = SinSWE.convert_to_backend(backend, values)
    output = SinSWE.convert_to_backend(backend, zeros(10))

    SinSWE.@fvmloop SinSWE.for_each_index_value(backend, values_backend) do index, value
        output[value] = value * 2
    end

    output = collect(output)
    @show output == 2 .* values
    @test output == 2 .* values
end