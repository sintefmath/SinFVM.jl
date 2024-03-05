using SinSWE
using CUDA
using Test
import Adapt
using StaticArrays



struct someotherstruct{dimension}
    x::Float64
    y::SVector{dimension, Float64}
end


Adapt.@adapt_structure someotherstruct

function gpu_add1!(y, x, g)
    for i = 1:length(y)
        @inbounds y[i] += x[i] + g.x + g.y[2]
    end
    return nothing
end


nx = 1024
grid = SinSWE.CartesianGrid(nx)
some = someotherstruct(10.0, SVector{3, Float64}(1, 2, 3))
backend = make_cuda_backend()

N = 10
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0
fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d, some)
@test all(Array(y_d) .== 3.0f0 + some.x + some.y[2])

SinSWE.@fvmloop SinSWE.for_each_ghost_cell(backend, grid, XDIR) do i
end

println("Done")