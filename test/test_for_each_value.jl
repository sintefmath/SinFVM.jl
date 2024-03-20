using Test
using SinSWE
using StaticArrays
@show get_available_backends()

for backend in get_available_backends()
    values = collect(1:10)
    values_backend = SinSWE.convert_to_backend(backend, values)
    output = SinSWE.convert_to_backend(backend, zeros(10))

    SinSWE.@fvmloop SinSWE.for_each_index_value(backend, values_backend) do index, value
        output[index] = value * 2
    end

    output = collect(output)
    @test output == 2 .* values


    values_svector = [SVector{2,Float64}(i, 2 * i) for i in collect(1:10)]
    values_svector_backend = SinSWE.convert_to_backend(backend, values_svector)
    output_svector = SinSWE.convert_to_backend(backend, zeros(10))

    SinSWE.@fvmloop SinSWE.for_each_index_value(backend, values_svector_backend) do index, value
        output_svector[index] = value[2] * 2
    end

    output = collect(output_svector)
    @test output == 4 .* values
end