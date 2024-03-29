using SinSWE
using Test
import CUDA
using StaticArrays
using Logging

for backend in get_available_backends()
    nx = 10
    ny = 15
    grid = SinSWE.CartesianGrid(nx, ny)
    equation = SinSWE.ShallowWaterEquationsPure()
    @debug "Creating volume"
    volume = SinSWE.Volume(backend, equation, grid)
    @debug "Setting volume"
    CUDA.@allowscalar volume[1, 1] = @SVector [2.0, 3.0, 4.0]
    @debug "Getting volume"
    CUDA.@allowscalar @test volume[1, 1] == @SVector [2.0, 3.0, 4.0]
    @debug "Got volume"
    @test size(volume) == size(grid) == (nx + 2, ny + 2)

    if backend isa SinSWE.CPUBackend
        all_elements = []
        for (n, element) in enumerate(volume)
            @test element isa SVector{3}
            if n == 1
                @test element == @SVector [2.0, 3.0, 4.0]
            end

            push!(all_elements, element)
        end
        @test length(all_elements) == length(volume)
    end
    ranged_index_set = zeros(SVector{3, Float64}, 9, 8)

    for i in 1:9
        for j in 1:8
            ranged_index_set[i, j] = @SVector [i, j, i + j]
        end
    end
    volume[2:10, 3:10] = ranged_index_set

    for i in 2:10
        for j in 3:10
            CUDA.@allowscalar @test volume[i,j] == @SVector [i-1, j - 2, i + j - 3]
        end
    end

    for i in 2:10
        for j in 3:10
            CUDA.@allowscalar @test volume.h[i,j] == i - 1
            CUDA.@allowscalar @test volume.hu[i,j] == j - 2
            CUDA.@allowscalar @test volume.hv[i,j] == i + j - 3
        end
    end

    volume.hv[4:9, 1:3] = 8 * ones(6, 3)

    for i in 4:9
        for j in 1:3
            CUDA.@allowscalar @test volume[i, j][3] == 8
        end
    end

    if backend isa SinSWE.CPUBackend
        all_elements = []
        for (n, element) in enumerate(volume.hu)
            @test element isa Real
            push!(all_elements, element)
        end
        @test length(all_elements) == length(volume)
    end

    interior_volume = SinSWE.InteriorVolume(volume)
    collected_interior_volume = collect(interior_volume)

    CUDA.@allowscalar interior_volume[3, 2] = @SVector [42, 43, 44]
    CUDA.@allowscalar @test interior_volume[3, 2] == @SVector [42, 43, 44]
    CUDA.@allowscalar @test volume[4, 3] == @SVector [42, 43, 44]

    if backend isa SinSWE.CPUBackend
        all_elements = []
        for (n, element) in enumerate(interior_volume)
            @test element isa SVector{3, <:Real}
            push!(all_elements, element)
        end
        @test length(all_elements) == prod(size(volume) .- 2 .* grid.ghostcells)
    end

    interior_h = interior_volume.h  

    @test length(interior_h) == prod(size(volume) .- 2 .* grid.ghostcells)
    @test size(interior_h) == Tuple(Int64(x) for x in size(volume) .- 2 .* grid.ghostcells)
    if backend isa SinSWE.CPUBackend
        all_elements = []
        for (n, element) in enumerate(interior_h)
            @test element isa Real
            push!(all_elements, element)
        end
        @test length(all_elements) == prod(size(volume) .- 2 .* grid.ghostcells)
    end

    interior_h[2:4, 2:3] = ones(3, 2)

    for j in 2:3
        for i in 2:4
            CUDA.@allowscalar @test volume[i+1, j+1][1] == 1.0
        end
    end
end