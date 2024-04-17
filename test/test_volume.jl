using SinSWE
using StaticArrays
using Test
import CUDA


for backend in get_available_backends()
    nx = 10
    grid = SinSWE.CartesianGrid(nx)
    equation = SinSWE.ShallowWaterEquations1D()
    volume = SinSWE.Volume(backend, equation, grid)

    CUDA.@allowscalar volume[1] = @SVector [4.0, 4.0]
    CUDA.@allowscalar @test volume[1][1] == 4.0

    hu = volume.hu

    CUDA.@allowscalar @test hu[1] == 4.0

    hu[3:7] = 3:7

    @test collect(hu[3:7]) == collect(3:7)

    for i = 3:7
        CUDA.@allowscalar @test volume[i] == @SVector [0.0, i]
    end

    inner_volume = SinSWE.InteriorVolume(volume)

    CUDA.@allowscalar @test inner_volume[1] == volume[2]

    new_values = zeros(SVector{2,Float64}, (9 - 3))
    for i = 4:9
        new_values[i-3] = @SVector [i, 2.0 * i]
    end

    new_values_backend = SinSWE.convert_to_backend(backend, new_values)
    volume[4:9] = new_values_backend

    for i = 4:9
        CUDA.@allowscalar @test volume[i] == @SVector [i, 2.0 * i]
    end

    newer_values = zeros(SVector{2,Float64}, (9 - 3))
    for i = 4:9
        newer_values[i-3] = @SVector [4 * i, i]
    end

    newer_values_backend = SinSWE.convert_to_backend(backend, newer_values)
    inner_volume[3:8] = newer_values_backend

    for i = 4:9
        CUDA.@allowscalar @test volume[i] == @SVector [4 * i, i]
    end

    f(v) = v[1]^2

    @test size(volume) == size(grid)
    @test length(volume) == prod(size(grid))

    @test size(similar(volume))[1] == size(volume)[1]
    @test size(similar(volume, size(volume, 1))) == size(volume, 1)
    @test size(similar(volume, Int64, size(volume, 1))) == size(volume, 1)
    @test eltype(similar(volume, Int64, size(volume, 1))) == Int64
    @test size(similar(volume, Int64))[1] == size(volume)[1]
    @test eltype(similar(volume, Int64)) == Int64

    squared = zeros(length(volume))
    # @show typeof(f.(volume))
    # squared .= f.(volume)
    # for i = 1:10
    #     CUDA.@allowscalar @test squared[i] == volume[i][1] ^ 2
    # end

    # @show typeof(f.(volume))
    # squared_interior = f.(inner_volume)
    # for i = 2:9
    #     CUDA.@allowscalar  @test squared_interior[i-1] == inner_volume[i-1][1] ^ 2
    # end

    collected_volume = collect(volume)
    collected_interior_volume = collect(inner_volume)

    collect_hu = collect(volume.hu)

    collect_inner_hu = collect(inner_volume.hu)

    if backend isa SinSWE.CPUBackend
        all_elements = []
        for (n, element) in enumerate(volume)
            @test element isa SVector{2}

            push!(all_elements, element)
        end
        @test length(all_elements) == length(volume)
    end
end
