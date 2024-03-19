struct PeriodicBC <: BoundaryCondition
end

struct WallBC <: BoundaryCondition
end



dimension(::Type{<:Grid{d}}) where {d} = d
dimension(::T) where T<:Grid = dimension(T)

struct CartesianGrid{dimension,BoundaryType,dimension2} <: Grid{dimension}
    ghostcells::SVector{dimension,Int64}
    totalcells::SVector{dimension,Int64}

    boundary::BoundaryType
    extent::SVector{dimension2,Float64} # NOTE: SMatrix seems to mess up CUDA.jl
    Δ::SVector{dimension,Float64}
end

directions(::Grid{1}) = (XDIR,)
directions(::Grid{2}) = (XDIR, YDIR)
directions(::Grid{3}) = (XDIR, YDIR, ZDIR)

function number_of_interior_cells(grid::CartesianGrid)
    return prod(interior_size(grid))
end

function interior_size(grid::CartesianGrid)
    return Tuple(Int64(i) for i in (grid.totalcells .- 2 .* grid.ghostcells))
end

function Base.size(grid::CartesianGrid)
    return Tuple(Int64(i) for i in grid.totalcells)
end


function CartesianGrid(nx; gc = 1, boundary = PeriodicBC(), extent = [0.0 1.0])
    domain_width = extent[1, 2] - extent[1, 1]
    Δx = domain_width / nx
    return CartesianGrid(SVector{1,Int64}([gc]),
        SVector{1,Int64}([nx + 2 * gc]),
        boundary, SVector{2,Float64}(extent[1], extent[2]),
        SVector{1,Float64}([Δx])
    )
end

function CartesianGrid(nx, ny; gc = 1, boundary = PeriodicBC(), extent = [0.0 1.0; 0.0 1.0])
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
        ghost_extend = [grid.extent[1] - grid.ghostcells[1]*dx 
                        grid.extent[2] + grid.ghostcells[1]*dx ]
        collect(LinRange(ghost_extend[1], ghost_extend[2], grid.totalcells[1] + 1))
    end
end

function cell_centers(grid::CartesianGrid{1}; interior=true)
    xinterface = cell_faces(grid; interior)
    xcell = xinterface[1:end-1] .+ (xinterface[2] - xinterface[1]) / 2.0
    return xcell
end

compute_dx(grid::CartesianGrid{1}, direction=XDIR) = grid.Δ[direction]
compute_dx(grid::CartesianGrid{dimension}, direction=XDIR) where {dimension} = grid.Δ[direction]

function constant_bottom_topography(grid::CartesianGrid{1}, value)
    # TODO: Allow constant bottom topography to be represented by a scalar with arbitrary index...
    return ones(grid.totalcells[1] + 1) .* value
end




function for_each_inner_cell(f, g::CartesianGrid{1}, include_ghostcells=0)
    for i in (g.ghostcells[1]-include_ghostcells+1):(g.totalcells[1]-g.ghostcells[1]+include_ghostcells)
        f(i - 1, i, i + 1)
    end
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

function ghost_cells(g::CartesianGrid{1}, direction)
    return g.ghostcells[direction]
end
