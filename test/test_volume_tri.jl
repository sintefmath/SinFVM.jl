# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using VolumeFluxes
using StaticArrays
using Test
import CUDA

# Helper: two-triangle mesh on the unit square
#
#   4 ---- 3
#   |  T2 /|
#   |   /  |
#   | /  T1|
#   1 ---- 2
function make_test_triangular_grid()
    nodes = [
        SVector(0.0, 0.0),
        SVector(1.0, 0.0),
        SVector(1.0, 1.0),
        SVector(0.0, 1.0),
    ]
    triangles = [
        SVector(1, 2, 3),
        SVector(1, 3, 4),
    ]
    neighbors = [
        SVector(0, 0, 2),
        SVector(1, 0, 0),
    ]
    return TriangularGrid(nodes, triangles, neighbors; boundary=OutflowBC())
end

for backend in get_available_backends()
    grid = make_test_triangular_grid()
    equation = VolumeFluxes.ShallowWaterEquationsPure()
    volume = VolumeFluxes.Volume(backend, equation, grid)

    ncells = VolumeFluxes.number_of_cells(grid)
    @test ncells == 2

    # Test size and length
    @test size(volume) == (2,)
    @test length(volume) == 2

    # Test setindex! / getindex
    CUDA.@allowscalar volume[1] = @SVector [4.0, 5.0, 6.0]
    CUDA.@allowscalar @test volume[1] == @SVector [4.0, 5.0, 6.0]

    CUDA.@allowscalar volume[2] = @SVector [7.0, 8.0, 9.0]
    CUDA.@allowscalar @test volume[2] == @SVector [7.0, 8.0, 9.0]

    # Test VolumeVariable access via property
    h = volume.h
    CUDA.@allowscalar @test h[1] == 4.0
    CUDA.@allowscalar @test h[2] == 7.0

    hu = volume.hu
    CUDA.@allowscalar @test hu[1] == 5.0
    CUDA.@allowscalar @test hu[2] == 8.0

    hv = volume.hv
    CUDA.@allowscalar @test hv[1] == 6.0
    CUDA.@allowscalar @test hv[2] == 9.0

    # Test VolumeVariable setindex!
    CUDA.@allowscalar h[1] = 42.0
    CUDA.@allowscalar @test volume[1][1] == 42.0

    # Test range setindex! on Volume
    new_values = SVector{3,Float64}[
        SVector{3,Float64}(10.0, 20.0, 30.0),
        SVector{3,Float64}(40.0, 50.0, 60.0),
    ]
    new_values_backend = VolumeFluxes.convert_to_backend(backend, new_values)
    volume[1:2] = new_values_backend

    CUDA.@allowscalar @test volume[1] == @SVector [10.0, 20.0, 30.0]
    CUDA.@allowscalar @test volume[2] == @SVector [40.0, 50.0, 60.0]

    # Test InteriorVolume — for triangular grids, interior == full
    inner_volume = VolumeFluxes.InteriorVolume(volume)

    CUDA.@allowscalar @test inner_volume[1] == volume[1]
    CUDA.@allowscalar @test inner_volume[2] == volume[2]

    @test length(inner_volume) == 2
    @test size(inner_volume) == (2,)

    # Test InteriorVolume setindex!
    CUDA.@allowscalar inner_volume[1] = @SVector [100.0, 200.0, 300.0]
    CUDA.@allowscalar @test volume[1] == @SVector [100.0, 200.0, 300.0]

    # Test InteriorVolume range setindex!
    newer_values = SVector{3,Float64}[
        SVector{3,Float64}(1.0, 2.0, 3.0),
        SVector{3,Float64}(4.0, 5.0, 6.0),
    ]
    newer_values_backend = VolumeFluxes.convert_to_backend(backend, newer_values)
    inner_volume[1:2] = newer_values_backend

    CUDA.@allowscalar @test volume[1] == @SVector [1.0, 2.0, 3.0]
    CUDA.@allowscalar @test volume[2] == @SVector [4.0, 5.0, 6.0]

    # Test InteriorVolumeVariable
    interior_h = inner_volume.h
    CUDA.@allowscalar @test interior_h[1] == 1.0
    CUDA.@allowscalar @test interior_h[2] == 4.0

    @test length(interior_h) == 2
    @test size(interior_h) == (2,)

    # Test collect
    collected_volume = collect(volume)
    collected_interior = collect(inner_volume)
    collect_h = collect(volume.h)
    collect_inner_h = collect(inner_volume.h)

    # Test similar
    @test size(similar(volume))[1] == size(volume)[1]
    @test size(similar(volume, size(volume, 1))) == size(volume, 1)
    @test size(similar(volume, Int64, size(volume, 1))) == size(volume, 1)
    @test eltype(similar(volume, Int64, size(volume, 1))) == Int64
    @test size(similar(volume, Int64))[1] == size(volume)[1]
    @test eltype(similar(volume, Int64)) == Int64

    # Test iteration (CPU only)
    if backend isa VolumeFluxes.CPUBackend
        all_elements = []
        for (n, element) in enumerate(volume)
            @test element isa SVector{3}
            push!(all_elements, element)
        end
        @test length(all_elements) == length(volume)

        # Test VolumeVariable iteration
        all_h = []
        for (n, element) in enumerate(volume.h)
            @test element isa Real
            push!(all_h, element)
        end
        @test length(all_h) == length(volume)

        # Test InteriorVolume iteration
        all_inner = []
        for (n, element) in enumerate(inner_volume)
            @test element isa SVector{3, <:Real}
            push!(all_inner, element)
        end
        @test length(all_inner) == length(inner_volume)

        # Test InteriorVolumeVariable iteration
        all_inner_h = []
        for (n, element) in enumerate(interior_h)
            @test element isa Real
            push!(all_inner_h, element)
        end
        @test length(all_inner_h) == length(interior_h)
    end
end
