struct PeriodicBC
end



abstract type Grid{dimension} end
struct CartesianGrid{dimension,BoundaryType, dimension2} <: Grid{dimension}
    ghostcells::SVector{dimension,Int64}
    totalcells::SVector{dimension,Int64}

    boundary::BoundaryType
    extent::SVector{dimension2, Float64} # NOTE: SMatrix seems to mess up CUDA.jl
    Δx::Float64
end

directions(::Grid{1}) = (XDIR,)

function CartesianGrid(nx; gc=1, boundary=PeriodicBC(), extent=[0.0 1.0])
    domain_width = extent[1, 2] - extent[1, 1]
    Δx = domain_width / nx
    return CartesianGrid(SVector{1,Int64}([gc]),
        SVector{1,Int64}([nx + 2 * gc]),
        boundary, SVector{2,Float64}(extent[1], extent[2]),
        Δx)
end

function cell_centers(grid::CartesianGrid{1}; interior=true)
    @assert interior

    xinterface = collect(LinRange(grid.extent[1], grid.extent[2], grid.totalcells[1] - 2 * grid.ghostcells[1] + 1))
    xcell = xinterface[1:end-1] .+ (xinterface[2] - xinterface[1]) / 2.0
    return xcell
end

compute_dx(grid::CartesianGrid{1}) = grid.Δx


function for_each_inner_cell(f, g::CartesianGrid{1}, include_ghostcells=0)
    for i in (g.ghostcells[1]-include_ghostcells+1):(g.totalcells[1]-g.ghostcells[1]+include_ghostcells)
        f(i - 1, i, i + 1)
    end
end

function inner_cells(g::CartesianGrid{1}, direction)
    return ((g.totalcells[1]-g.ghostcells[1]) - (g.ghostcells[1]+1), )
end

function left_cell(g::CartesianGrid{1}, I::Int64, direction::XDIRT)
    return I + g.ghostcells[1]  - 1
end

function middle_cell(g::CartesianGrid{1}, I::Int64, direction::XDIRT)
    return I + g.ghostcells[1]
end


function right_cell(g::CartesianGrid{1}, I::Int64, direction::XDIRT)
    return I + g.ghostcells[1] + 1
end

function ghost_cells(g::CartesianGrid{1}, direction::XDIRT)
    return g.ghostcells[1]
end