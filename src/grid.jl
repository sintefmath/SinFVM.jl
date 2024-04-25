struct PeriodicBC <: BoundaryCondition
end

struct WallBC <: BoundaryCondition
end



dimension(::Type{<:Grid{d}}) where {d} = d
dimension(::T) where {T<:Grid} = dimension(T)

struct CartesianGrid{dimension,BoundaryType,dimension2} <: Grid{dimension}
    ghostcells::SVector{dimension,Int64}
    totalcells::SVector{dimension,Int64}

    boundary::BoundaryType
    extent::SVector{dimension2,Float64} # NOTE: SMatrix seems to mess up CUDA.jl
    Δ::SVector{dimension,Float64}
end

start_extent(grid::CartesianGrid, direction) = grid.extent[(Base.to_index(direction)-1)*2 + 1]
end_extent(grid::CartesianGrid, direction) = grid.extent[(Base.to_index(direction)-1)*2 + 2]
extent(grid, direction) = SVector{2, Int64}(start_extent(grid, direction), end_extent(grid, direction))

directions(::Grid{1}) = (XDIR,)
directions(::Grid{2}) = (XDIR, YDIR)
directions(::Grid{3}) = (XDIR, YDIR, ZDIR)

function number_of_interior_cells(grid::CartesianGrid)
    return prod(interior_size(grid))
end

function interior_size(grid::CartesianGrid)
    return Tuple(Int64(i) for i in (grid.totalcells .- 2 .* grid.ghostcells))
end

function interior_size(grid::CartesianGrid, direction)
    return (grid.totalcells[direction] .- 2 .* grid.ghostcells[direction])
end

function Base.size(grid::CartesianGrid{1})
    return (grid.totalcells[1],)
end

function Base.size(grid::CartesianGrid{2})
    return (grid.totalcells[1], grid.totalcells[2])
end


function CartesianGrid(nx; gc=1, boundary=PeriodicBC(), extent=[0.0 1.0])
    domain_width = extent[1, 2] - extent[1, 1]
    Δx = domain_width / nx
    return CartesianGrid(SVector{1,Int64}([gc]),
        SVector{1,Int64}([nx + 2 * gc]),
        boundary, SVector{2,Float64}(extent[1], extent[2]),
        SVector{1,Float64}([Δx])
    )
end

function CartesianGrid(nx, ny; gc=1, boundary=PeriodicBC(), extent=[0.0 1.0; 0.0 1.0])
    domain_width = extent[1, 2] - extent[1, 1]
    domain_height = extent[2, 2] - extent[2, 1]
    Δ = SVector{2,Float64}([domain_width / nx, domain_height / ny])
    return CartesianGrid(SVector{2,Int64}([gc, gc]),
        SVector{2,Int64}([nx + 2 * gc, ny + 2 * gc]),
        boundary, SVector{4,Float64}(extent[1, 1], extent[1, 2], extent[2, 1], extent[2, 2]),
        Δ)
end

function cell_faces(grid::CartesianGrid{1}; interior=true)
    if interior
        return collect(LinRange(grid.extent[1], grid.extent[2], grid.totalcells[1] - 2 * grid.ghostcells[1] + 1))
    else
        dx = compute_dx(grid)
        ghost_extend = [grid.extent[1] - grid.ghostcells[1] * dx
            grid.extent[2] + grid.ghostcells[1] * dx]
        collect(LinRange(ghost_extend[1], ghost_extend[2], grid.totalcells[1] + 1))
    end
end

function cell_faces(grid::CartesianGrid{2}, dir::Direction; interior=true)
    if interior
        faces = LinRange(start_extent(grid, dir), end_extent(grid, dir), interior_size(grid, dir) + 1)
        return faces
    else
        dx = compute_dx(grid, dir)
        ghost_extend = [start_extent(grid, dir) - grid.ghostcells[dir] * dx
            end_extent(grid, dir)  + grid.ghostcells[dir] * dx]
        
        faces = LinRange(ghost_extend[1], ghost_extend[2], grid.totalcells[dir] + 1)
        return faces
    end
end

function cell_faces(grid::CartesianGrid{2}; interior=true)
    x_faces = cell_faces(grid, XDIR; interior=interior)
    y_faces = cell_faces(grid, YDIR; interior=interior)
    
    all_faces = zeros(SVector{2, Float64}, length(x_faces), length(y_faces))
    for (j, y) in enumerate(y_faces)
        for (i, x) in enumerate(x_faces)
            all_faces[i, j] = @SVector [x, y]
        end
    end

    return all_faces
end
   


function cell_centers(grid::CartesianGrid{1}; interior=true)
    xinterface = cell_faces(grid; interior)
    xcell = xinterface[1:end-1] .+ (xinterface[2] - xinterface[1]) / 2.0
    return xcell
end

function cell_centers(grid::CartesianGrid{2}, dir::Direction; interior=true)
    faces = cell_faces(grid, dir; interior)
    xcell = faces[1:end-1] .+ (faces[2] - faces[1]) / 2.0
    return xcell
end

function cell_centers(grid::CartesianGrid{2}; interior=true)
    xinterface = cell_faces(grid; interior)
    dx = compute_dx(grid)
    dy = compute_dy(grid)
    centers = zeros(eltype(xinterface), (size(xinterface) .- 1)...)
    for index in CartesianIndices(centers)
        centers[index] = xinterface[index] + eltype(xinterface)(dx, dy) ./ 2
    end

    return centers
end

compute_dx(grid::CartesianGrid{1}, direction=XDIR) = grid.Δ[direction]
compute_dx(grid::CartesianGrid{dimension}, direction=XDIR) where {dimension} = grid.Δ[direction]
compute_dy(grid::CartesianGrid{2}) = grid.Δ[YDIR]

compute_cell_size(grid::CartesianGrid) = prod(grid.Δ)

function constant_bottom_topography(grid::CartesianGrid{1}, value)
    # TODO: Allow constant bottom topography to be represented by a scalar with arbitrary index...
    return ones(grid.totalcells[1] + 1) .* value
end


function inner_cells(g::CartesianGrid{1}, direction, ghostcells=g.ghostcells[direction])
    return g.totalcells[direction] - 2 * ghostcells
end

function left_cell(g::CartesianGrid{1}, I::Int64, direction, ghostcells=g.ghostcells[direction])
    return I + ghostcells - 1
end

function middle_cell(g::CartesianGrid{1}, I::Int64, direction, ghostcells=g.ghostcells[direction])
    return I + ghostcells
end


function right_cell(g::CartesianGrid{1}, I::Int64, direction, ghostcells=g.ghostcells[direction])
    return I + ghostcells + 1
end


function left_cell(g::CartesianGrid{1}, I::CartesianIndex, direction, ghostcells=g.ghostcells[direction])
    return left_cell(g, I[1], direction, ghostcells)
end

function middle_cell(g::CartesianGrid{1}, I::CartesianIndex, direction, ghostcells=g.ghostcells[direction])
    return middle_cell(g, I[1], direction, ghostcells)
end


function right_cell(g::CartesianGrid{1}, I::CartesianIndex, direction, ghostcells=g.ghostcells[direction])
    return right_cell(g, I[1], direction, ghostcells)
end

function ghost_cells(g::CartesianGrid{1}, direction)
    return g.ghostcells[direction]
end




function inner_cells(g::CartesianGrid{2}, direction, ghostcells=g.ghostcells[direction])
    # TODO: Review. Do we want to include the full grid in the non-active dimension?
    dirs = directions(g)

    inner_size = (dirs .!= direction) .* (g.totalcells .- 2 .* g.ghostcells) .+
                 (dirs .== direction) .* (g.totalcells .- 2 .* ghostcells)
    return (inner_size[1], inner_size[2])
end

# TODO: This can be done nicer (left_cell, middle_cell, right_cell)
function left_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::XDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + ghostcells - 1, I[2] + g.ghostcells[2])
end

function left_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::YDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + g.ghostcells[1], I[2] + ghostcells - 1)
end

function middle_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::XDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + ghostcells, I[2] + g.ghostcells[2])
end

function middle_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::YDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + g.ghostcells[1], I[2] + ghostcells)
end
function right_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::XDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + ghostcells + 1, I[2] + g.ghostcells[2])
end

function right_cell(g::CartesianGrid{2}, I::CartesianIndex, direction::YDIRT, ghostcells=g.ghostcells[direction])
    return CartesianIndex(I[1] + g.ghostcells[1], I[2] + ghostcells + 1)
end

function ghost_cells(g::CartesianGrid{2}, ::XDIRT)
    return (g.ghostcells[1], inner_cells(g, XDIR)[2])
end


function ghost_cells(g::CartesianGrid{2}, ::YDIRT)
    return (inner_cells(g, YDIR)[1], g.ghostcells[2])
end