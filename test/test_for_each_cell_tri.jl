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

# Helper: two-triangle mesh
function make_test_triangular_grid_loop()
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
    grid = make_test_triangular_grid_loop()
    ncells = VolumeFluxes.number_of_cells(grid)
    @test ncells == 2

    # ────────────────────────────────────────────────────────────────
    # Test for_each_cell: iterate over all cells
    # ────────────────────────────────────────────────────────────────
    output_array = VolumeFluxes.convert_to_backend(backend, zeros(ncells))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell(backend, grid) do index
        output_array[index] = index
    end

    @test collect(output_array) == [1.0, 2.0]

    # ────────────────────────────────────────────────────────────────
    # Test for_each_cell_neighbor: iterate over (cell, edge, neighbor)
    # ────────────────────────────────────────────────────────────────
    # Grid topology:
    #   T1: neighbors = (0, 0, 2)  -> edges 1,2 are boundary, edge 3 -> T2
    #   T2: neighbors = (1, 0, 0)  -> edge 1 -> T1, edges 2,3 are boundary

    # Count boundary edges per cell
    boundary_count = VolumeFluxes.convert_to_backend(backend, zeros(Int64, ncells))
    interior_count = VolumeFluxes.convert_to_backend(backend, zeros(Int64, ncells))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell_neighbor(backend, grid) do cell, edge_idx, neighbor
        if neighbor == 0
            boundary_count[cell] += 1
        else
            interior_count[cell] += 1
        end
    end

    @test collect(boundary_count) == [2, 2]  # T1 has 2 boundary, T2 has 2 boundary
    @test collect(interior_count) == [1, 1]  # T1 has 1 interior, T2 has 1 interior

    # Verify neighbor indices
    neighbor_sum = VolumeFluxes.convert_to_backend(backend, zeros(Int64, ncells))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell_neighbor(backend, grid) do cell, edge_idx, neighbor
        neighbor_sum[cell] += neighbor
    end

    @test collect(neighbor_sum) == [2, 1]  # T1: 0+0+2=2, T2: 1+0+0=1

    # ────────────────────────────────────────────────────────────────
    # Test for_each_cell with a larger grid
    # ────────────────────────────────────────────────────────────────
    # 4-triangle mesh:  split a 2×1 rectangle into 4 triangles
    #   4 ---- 5 ---- 6
    #   |  T2 /|  T4 /|
    #   |   /  |   /  |
    #   | /  T1| /  T3|
    #   1 ---- 2 ---- 3
    nodes4 = [
        SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(2.0, 0.0),
        SVector(0.0, 1.0), SVector(1.0, 1.0), SVector(2.0, 1.0),
    ]
    tris4 = [
        SVector(1, 2, 5),  # T1
        SVector(1, 5, 4),  # T2
        SVector(2, 3, 6),  # T3
        SVector(2, 6, 5),  # T4
    ]
    nbrs4 = [
        SVector(0, 4, 2),  # T1: edge1=boundary, edge2=T4, edge3=T2
        SVector(1, 0, 0),  # T2: edge1=T1, edge2=boundary, edge3=boundary
        SVector(0, 0, 4),  # T3: edge1=boundary, edge2=boundary, edge3=T4
        SVector(3, 0, 1),  # T4: edge1=T3, edge2=boundary, edge3=T1
    ]

    grid4 = TriangularGrid(nodes4, tris4, nbrs4; boundary=OutflowBC())
    @test VolumeFluxes.number_of_cells(grid4) == 4

    output4 = VolumeFluxes.convert_to_backend(backend, zeros(4))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell(backend, grid4) do index
        output4[index] = index * 10
    end

    @test collect(output4) == [10.0, 20.0, 30.0, 40.0]

    boundary_count4 = VolumeFluxes.convert_to_backend(backend, zeros(Int64, 4))

    VolumeFluxes.@fvmloop VolumeFluxes.for_each_cell_neighbor(backend, grid4) do cell, edge_idx, neighbor
        if neighbor == 0
            boundary_count4[cell] += 1
        end
    end

    @test collect(boundary_count4) == [1, 2, 2, 1]
end
