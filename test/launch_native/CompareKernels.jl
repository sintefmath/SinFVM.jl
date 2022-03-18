using Test

using CUDA
using Plots
using NPZ

include("GPUOceanUtils.jl")

function diffBump()
    MyType = Float32

    Nx::Int32 = Ny::Int32 = 256
    dx::MyType = dy::MyType = 200.0

    ngc = 2
    data_shape = (Ny + 2 * ngc, Nx + 2 * ngc)

    eta1 = zeros(MyType, data_shape)
    makeCentralBump!(eta1, Nx, Ny, dx, dy)

    etaFromFile = npzread("data/eta_init.npy")


    return eta1 - etaFromFile
end

function maxDiff(step)
    etaInit = npzread("data/eta_init.npy")
    etaStep = npzread("data/eta_$(step).npy")
    return maximum(broadcast(abs, etaInit-etaStep))
end



@test all(diffBump() .== 0.0)
for i ∈ range(0, 14)
    #print(i, "\n")
    print("$(i): $(maxDiff(i)) \n")
end

maxDiff(1)

