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

    for i in 3:7
        @test volume[i] == @SVector [0.0, i]
    end

    inner_volume = SinSWE.InteriorVolume(volume)

    @test inner_volume[1] == volume[2]

    new_values = zeros(SVector{2,Float64}, (9 - 3))
    for i in 4:9
        new_values[i-3] = @SVector [i, 2.0 * i]
    end

    new_values_backend = SinSWE.convert_to_backend(backend, new_values)
    volume[4:9] = new_values_backend

    for i in 4:9
        @SVector [i, 2.0 * i]
        @test volume[i] == @SVector [i, 2.0 * i]
    end
end