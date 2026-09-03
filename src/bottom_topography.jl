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

abstract type AbstractBottomTopography end


struct ConstantBottomTopography{T} <: AbstractBottomTopography
    B::T
    ConstantBottomTopography(B=0.0) = new{typeof(B)}(B)
end
Adapt.@adapt_structure ConstantBottomTopography


struct BottomTopography1D{T} <: AbstractBottomTopography
    B::T
    function BottomTopography1D(B, backend, grid::CartesianGrid{1})
        # TODO: If B is only defined for interior, should we then extend B?
        validate_bottom_topography(B, grid)
        B = convert_to_backend(backend, B)
        return new{typeof(B)}(B)
    end

    BottomTopography1D(B; should_never_be_called) = new{typeof(B)}(B)
end

function Adapt.adapt_structure(
    to,
    topo::BottomTopography1D
) 
    B = Adapt.adapt_structure(to, topo.B)  
    BottomTopography1D(B; should_never_be_called=nothing)
end

struct BottomTopography2D{T} <: AbstractBottomTopography
    B::T
    function BottomTopography2D(B, backend, grid::CartesianGrid{2})
        validate_bottom_topography(B, grid)
        B = convert_to_backend(backend, B)
        return new{typeof(B)}(B)
    end
    BottomTopography2D(B; should_never_be_called) = new{typeof(B)}(B)
end

function Adapt.adapt_structure(
    to,
    topo::BottomTopography2D
) 
    B = Adapt.adapt_structure(to, topo.B)  
    BottomTopography2D(B; should_never_be_called=nothing)
end

## Validate
function validate(::ConstantBottomTopography, ::Grid)
    nothing
end

function validate(bottom::AbstractBottomTopography, grid::Grid)
    validate_bottom_topography(bottom.B, grid)
end

function validate_bottom_topography(B, grid::Grid)
    if size(B) != size(grid) .+1
        throw(DomainError("Bottom topography should be of size $(size(grid) .+ 1) but got $(size(B))"))
    end
end

# Split so the array case reduces on the device instead of broadcasting against a
# Float64 literal, and so the constant case does not go through `all` at all.
is_zero(B::ConstantBottomTopography) = iszero(B.B)
is_zero(B::AbstractBottomTopography) = all(iszero, B.B)

# Lookup for ConstantBottomTopography
B_cell(B::ConstantBottomTopography, i...) = B.B
B_face_left(B::ConstantBottomTopography, i...) = B.B
B_face_right(B::ConstantBottomTopography, i...) = B.B

# Lookup for BottomTopography1D
B_cell(B::BottomTopography1D, index, dir::XDIRT=XDIR) = (B.B[index] + B.B[index + 1])/2
B_face_left(B::BottomTopography1D, index, dir::XDIRT=XDIR) = B.B[index]
B_face_right(B::BottomTopography1D, index, dir::XDIRT=XDIR) = B.B[index + 1]

# Lookup for BottomTopography2D
B_cell(B::BottomTopography2D, x, y) = (B.B[x, y] + B.B[x+1, y] + B.B[x, y + 1] + B.B[x + 1, y + 1])/4
B_face_left(B::BottomTopography2D, x, y, ::XDIRT) = (B.B[x, y] + B.B[x, y + 1])/2
B_face_right(B::BottomTopography2D, x, y, ::XDIRT) = (B.B[x + 1, y] + B.B[x + 1, y + 1])/2
B_face_left(B::BottomTopography2D, x, y, ::YDIRT) = (B.B[x, y] + B.B[x + 1, y])/2
B_face_right(B::BottomTopography2D, x, y, ::YDIRT) = (B.B[x, y + 1] + B.B[x + 1, y + 1])/2

# Lookup for 2D with Cartesian indices
# TODO: Clearify assumption: These functions assume that I already accounts for ghost cells
B_cell(B::BottomTopography2D, I::CartesianIndex) = B_cell(B, I[1], I[2])
B_face_left(B::BottomTopography2D, I::CartesianIndex, dir) = B_face_left(B, I[1], I[2], dir)
B_face_right(B::BottomTopography2D, I::CartesianIndex, dir) = B_face_right(B, I[1], I[2], dir)


# Collect all cell/face values
function collect_topography_cells(B::ConstantBottomTopography, grid::CartesianGrid; interior=true)
    dims = interior ? interior_size(grid) : size(grid)
    return ones(dims)*B.B
end
function collect_topography_intersections(B::ConstantBottomTopography, grid::CartesianGrid; interior=true)
    dims = interior ? interior_size(grid) .+1 : size(grid) .+1
    return ones(dims)*B.B
end

function collect_topography_cells(B::BottomTopography1D, grid::CartesianGrid{1}; interior=true) 
    topo_intersections = collect(B.B)
    topo = 0.5.*(topo_intersections[1:end-1] + topo_intersections[2:end]) 
    if interior
        return topo[grid.ghostcells[1] + 1 : end - grid.ghostcells[1]]
    else
        return topo
    end
end
function collect_topography_intersections(B::BottomTopography1D, grid::CartesianGrid{1}; interior=true)
    if interior
        return collect(B.B)[grid.ghostcells[1] + 1 : end - grid.ghostcells[1]]
    else
        return collect(B.B)
    end
end

function collect_topography_cells(B::BottomTopography2D, grid::CartesianGrid{2}; interior=true) 
    topo_intersections = collect(B.B)
    topo_y = 0.5.*(topo_intersections[1:end-1, :] .+ topo_intersections[2:end, :]) 
    topo = 0.5.*(topo_y[:, 1:end-1] .+ topo_y[:, 2:end]) 
    if interior
        return topo[grid.ghostcells[1] + 1 : end - grid.ghostcells[1], grid.ghostcells[2] + 1 : end - grid.ghostcells[2]]
    else
        return topo
    end
end

function collect_topography_intersections(B::BottomTopography2D, grid::CartesianGrid{2}; interior=true)
    if interior
        return collect(B.B)[grid.ghostcells[1] + 1 : end - grid.ghostcells[1], grid.ghostcells[2] + 1 : end - grid.ghostcells[2]]
    else
        return collect(B.B)
    end
end
