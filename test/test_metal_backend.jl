using Test

include(joinpath(@__DIR__, "..", "src", "SinFVM.jl"))

@testset "Metal backend" begin
    if VolumeFluxes.has_metal_backend()
        backend = VolumeFluxes.make_metal_backend()
        @test backend isa VolumeFluxes.MetalBackend
        @test VolumeFluxes.name(backend) == "Metal"

        arr = VolumeFluxes.convert_to_backend(backend, Float32[1, 2, 3])
        @test Array(arr) == Float32[1, 2, 3]

        buffer = VolumeFluxes.create_buffer(backend, 2, (3, 4))
        @test size(buffer) == (3, 4, 2)
        @test eltype(Array(buffer)) == Float32
    else
        @test true
    end
end
