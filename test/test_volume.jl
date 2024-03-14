using SinSWE
using StaticArrays
using Test

for backend in get_available_backends()
    @info "Backend " backend
    nx = 10
    grid = SinSWE.CartesianGrid(nx)
    equation = SinSWE.ShallowWaterEquations1D()
    volume = Volume(backend, equation, grid)

    volume[1] = @SVector [4.0, 4.0]
    @test volume[1][1] == 4.0

    hu = volume.hu

    @test hu[1] == 4.0

    hu[3:7] = 3:7

    @test collect(hu[3:7]) == collect(3:7)

    for i = 3:7
        @test volume[i] == @SVector [0.0, i]
    end

    inner_volume = SinSWE.InteriorVolume(volume)

    @test inner_volume[1] == volume[2]

    new_values = zeros(SVector{2,Float64}, (9 - 3))
    for i = 4:9
        new_values[i-3] = @SVector [i, 2.0 * i]
    end

    new_values_backend = SinSWE.convert_to_backend(backend, new_values)
    volume[4:9] = new_values_backend

    for i = 4:9
        @test volume[i] == @SVector [i, 2.0 * i]
    end

    newer_values = zeros(SVector{2,Float64}, (9 - 3))
    for i = 4:9
        newer_values[i-3] = @SVector [4 * i, i]
    end

    newer_values_backend = SinSWE.convert_to_backend(backend, newer_values)
    inner_volume[3:8] = newer_values_backend

    for i = 4:9
        @test volume[i] == @SVector [4 * i, i]
    end

    f(v) = v[1]^2
    @show f(volume[1])

    for (n, v) in enumerate(volume)
        @show n
        @show f(v)
    end
    @show size(volume)
    @show length(volume)
    squared = zeros(length(volume))
    @show similar(volume)
    @show similar(volume, size(volume, 1))
    @show similar(volume, Int64, size(volume, 1))
    @show similar(volume, Int64)

    # @show typeof(f.(volume))
    squared .= f.(volume)
    for i = 1:10
        @test squared[i] == volume[i][1] ^ 2
    end

end
