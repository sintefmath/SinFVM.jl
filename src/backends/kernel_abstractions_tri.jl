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


"""
    for_each_cell(f, backend, grid::TriangularGrid, y...)

Iterate over every cell in the triangular grid, calling `f(cell_index, y...)`
for each cell index from `1` to `number_of_cells(grid)`.
"""
function for_each_cell(f, backend::KernelAbstractionBackend{T}, grid::TriangularGrid, y...) where {T}
    ev = for_each_cell_kernel(backend.backend, 1024)(f, grid, y..., ndrange=size(grid))
end


@kernel function for_each_cell_neighbor_kernel(f, grid, y...)
    I = @index(Global, Cartesian)
    cell = toint(I)
    for k in 1:3
        nb = grid.neighbors[cell][k]
        f(cell, k, nb, y...)
    end
end

"""
    for_each_cell_neighbor(f, backend, grid::TriangularGrid, y...)

Iterate over every (cell, edge, neighbor) triple in the triangular grid.
Calls `f(cell, edge_index, neighbor, y...)` where `neighbor` is `0`
for boundary edges and a positive integer for interior edges.
"""
function for_each_cell_neighbor(f, backend::KernelAbstractionBackend{T}, grid::TriangularGrid, y...) where {T}
    ev = for_each_cell_neighbor_kernel(backend.backend, 1024)(f, grid, y..., ndrange=size(grid))
end
