using SinSWE
using StaticArrays
using Test

nx = 10
grid = SinSWE.CartesianGrid(nx)
backend = make_cuda_backend()
equation = SinSWE.ShallowWaterEquations1D()
volume = Volume(backend, equation, grid)

volume[1] = @SVector [4.0, 4.0]
@test volume[1][1] == 4.0

hu = volume.hu

@test hu[1] == 4.0