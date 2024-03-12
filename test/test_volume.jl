using SinSWE
using StaticArrays
using Test

for backend in get_available_backends()
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
end