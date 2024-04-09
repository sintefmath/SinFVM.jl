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

# Lookup for ConstantBottomTopography
B_cell(B::ConstantBottomTopography, i...) = B.B
B_face_left(B::ConstantBottomTopography, i...) = B.B
B_face_right(B::ConstantBottomTopography, i...) = B.B

# Lookup for BottomTopography1D
B_cell(B::BottomTopography1D, index, dir::XDIRT=XDIR) = 0.5*(B.B[index] + B.B[index + 1])
B_face_left(B::BottomTopography1D, index, dir::XDIRT=XDIR) = B.B[index]
B_face_right(B::BottomTopography1D, index, dir::XDIRT=XDIR) = B.B[index + 1]

# Lookup for BottomTopography2D
B_cell(B::BottomTopography2D, x, y) = 0.25*(B.B[x, y] + B.B[x+1, y] + B.B[x, y + 1] + B.B[x + 1, y + 1])
B_face_left(B::BottomTopography2D, x, y, ::XDIRT) = 0.5*(B.B[x, y] + B.B[x, y + 1])
B_face_right(B::BottomTopography2D, x, y, ::XDIRT) = 0.5*(B.B[x + 1, y] + B.B[x + 1, y + 1])
B_face_left(B::BottomTopography2D, x, y, ::YDIRT) = 0.5*(B.B[x, y] + B.B[x + 1, y])
B_face_right(B::BottomTopography2D, x, y, ::YDIRT) = 0.5*(B.B[x, y + 1] + B.B[x + 1, y + 1])

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