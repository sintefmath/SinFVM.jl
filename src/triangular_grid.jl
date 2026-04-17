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
    OutflowBC <: BoundaryCondition

Outflow (zero-gradient / transmissive) boundary condition for triangular grids.
The boundary flux is computed using the interior cell state on both sides of
the interface, so that waves leave the domain freely.
"""
struct OutflowBC <: BoundaryCondition end

"""
    TriangularWallBC <: BoundaryCondition

Reflective wall boundary condition for triangular grids.
The boundary flux is computed by mirroring the normal component of
momentum so that there is no flow through the boundary edge.
"""
struct TriangularWallBC <: BoundaryCondition end

"""
    boundary(bc::OutflowBC, U)

Compute the ghost state for an outflow boundary.  The ghost state is
identical to the interior state `U`, producing a zero-gradient condition.
"""
boundary(::OutflowBC, U) = U

"""
    boundary(bc::TriangularWallBC, U, normal)

Compute the ghost state for a wall boundary.  The velocity component normal
to the wall is reflected (sign-flipped), while the tangential component is
preserved.  `U` is an `SVector{3}` with `(h, hu, hv)` and `normal` is the
outward unit normal `SVector{2}`.
"""
function boundary(::TriangularWallBC, U::SVector{3}, normal::SVector{2})
    h  = U[1]
    hu = U[2]
    hv = U[3]
    # Reflect the momentum vector: m_ghost = m - 2(m·n)n
    mn = hu * normal[1] + hv * normal[2]
    return @SVector [h, hu - 2 * mn * normal[1], hv - 2 * mn * normal[2]]
end

"""
    TriangularGrid{BoundaryType} <: Grid{2}

An unstructured triangular grid in two dimensions.

# Fields
- `nodes::Vector{SVector{2,Float64}}`     – vertex coordinates.
- `triangles::Vector{SVector{3,Int}}`     – vertex indices for each triangle.
- `neighbors::Vector{SVector{3,Int}}`     – neighbor cell index per edge (0 = boundary).
- `edge_normals::Vector{SVector{3,SVector{2,Float64}}}` – outward unit normal per edge.
- `edge_lengths::Vector{SVector{3,Float64}}`             – length of each edge.
- `areas::Vector{Float64}`               – area of each triangle.
- `centroids::Vector{SVector{2,Float64}}`  – centroid of each triangle.
- `boundary::BoundaryType`                – boundary condition instance.
"""
_norm2(v::SVector{2,Float64}) = sqrt(v[1]^2 + v[2]^2)
_dot2(a::SVector{2,Float64}, b::SVector{2,Float64}) = a[1]*b[1] + a[2]*b[2]

struct TriangularGrid{BoundaryType} <: Grid{2}
    nodes::Vector{SVector{2,Float64}}
    triangles::Vector{SVector{3,Int}}
    neighbors::Vector{SVector{3,Int}}
    edge_normals::Vector{SVector{3,SVector{2,Float64}}}
    edge_lengths::Vector{SVector{3,Float64}}
    areas::Vector{Float64}
    centroids::Vector{SVector{2,Float64}}
    boundary::BoundaryType
end

"""
    TriangularGrid(nodes, triangles, neighbors; boundary=OutflowBC())

Construct a `TriangularGrid` from raw connectivity data.

# Arguments
- `nodes`      – `Vector{SVector{2,Float64}}` of vertex positions.
- `triangles`  – `Vector{SVector{3,Int}}` of vertex indices per triangle.
- `neighbors`  – `Vector{SVector{3,Int}}` of neighbor indices per edge (0 = boundary).
- `boundary`   – boundary condition (default `OutflowBC()`).
"""
function TriangularGrid(nodes::Vector{SVector{2,Float64}},
                        triangles::Vector{SVector{3,Int}},
                        neighbors::Vector{SVector{3,Int}};
                        boundary=OutflowBC())
    ncells = length(triangles)

    edge_normals = Vector{SVector{3,SVector{2,Float64}}}(undef, ncells)
    edge_lengths = Vector{SVector{3,Float64}}(undef, ncells)
    areas        = Vector{Float64}(undef, ncells)
    centroids    = Vector{SVector{2,Float64}}(undef, ncells)

    for i in 1:ncells
        v1 = nodes[triangles[i][1]]
        v2 = nodes[triangles[i][2]]
        v3 = nodes[triangles[i][3]]

        centroids[i] = (v1 + v2 + v3) / 3.0

        # Area via cross product
        areas[i] = 0.5 * abs((v2[1] - v1[1]) * (v3[2] - v1[2]) -
                              (v3[1] - v1[1]) * (v2[2] - v1[2]))

        # Edges: e1 = v1→v2, e2 = v2→v3, e3 = v3→v1
        edges = (v2 - v1, v3 - v2, v1 - v3)

        lens = SVector{3,Float64}(_norm2(edges[1]), _norm2(edges[2]), _norm2(edges[3]))

        # Outward unit normals (rotate edge tangent 90° clockwise, then
        # orient outward with respect to the opposite vertex).
        normals = Vector{SVector{2,Float64}}(undef, 3)
        opposite_vertices = (v3, v1, v2)                  # vertex opposite to each edge
        edge_midpoints    = ((v1 + v2) / 2, (v2 + v3) / 2, (v3 + v1) / 2)
        for k in 1:3
            # Rotate the edge vector 90° clockwise to get a candidate normal
            n = SVector{2,Float64}(edges[k][2], -edges[k][1])
            n = n / _norm2(n)
            # Ensure it points away from the opposite vertex
            if _dot2(n, edge_midpoints[k] - opposite_vertices[k]) < 0
                n = -n
            end
            normals[k] = n
        end

        edge_normals[i] = SVector{3,SVector{2,Float64}}(normals[1], normals[2], normals[3])
        edge_lengths[i] = lens
    end

    return TriangularGrid{typeof(boundary)}(nodes, triangles, neighbors,
                                            edge_normals, edge_lengths,
                                            areas, centroids, boundary)
end

"""
    number_of_cells(grid::TriangularGrid)

Return the number of triangular cells.
"""
number_of_cells(grid::TriangularGrid) = length(grid.triangles)

number_of_interior_cells(grid::TriangularGrid) = number_of_cells(grid)

"""
    interior_size(grid::TriangularGrid)

Return the interior size of the triangular grid.  Since triangular grids
have no ghost cells, this is the same as `size(grid)`.
"""
interior_size(grid::TriangularGrid) = (number_of_cells(grid),)

Base.size(grid::TriangularGrid) = (number_of_cells(grid),)

"""
    cell_centers(grid::TriangularGrid)

Return a vector of centroid coordinates.
"""
cell_centers(grid::TriangularGrid) = grid.centroids
