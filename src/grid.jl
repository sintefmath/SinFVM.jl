struct PeriodicBC 
end



abstract type Grid{dimension} end
struct CartesianGrid{dimension, BoundaryType} <: Grid{dimension}
    ghostcells::SVector{dimension, Int64}
    totalcells::SVector{dimension, Int64}

    boundary::BoundaryType
    extent::SMatrix{dimension, 2, Float64}
    Δx::Float64
end

function CartesianGrid(nx; gc=1, boundary=PeriodicBC(), extent=[0.0 1.0])
    domain_width = extent[1,2]-extent[1,1]
    Δx = domain_width / nx
    return CartesianGrid(SVector{1, Int64}([gc]),
        SVector{1, Int64}([nx + 2 * gc]),
        boundary, SMatrix{1, 2, Float64}(extent), 
        Δx)
end

function cell_centers(grid::CartesianGrid{1}; interior=true)
    @assert interior

    xinterface = collect(LinRange(grid.extent[1, 1], grid.extent[1, 2], grid.totalcells[1] - 2 * grid.ghostcells[1] + 1))
    xcell = xinterface[1:end-1] .+ (xinterface[2]-xinterface[1])/2.0
    return xcell
end

compute_dx(grid::CartesianGrid{1}) = grid.Δx

function update_bc!(::PeriodicBC, grid::CartesianGrid{1}, data)
    for ghostcell in 1:grid.ghostcells[1]
        data[ghostcell] = data[end + ghostcell - 2 * grid.ghostcells[1] ]
        data[end - (grid.ghostcells[1]-ghostcell)] = data[grid.ghostcells[1] + ghostcell]
    end
end

function update_bc!(grid::CartesianGrid{1}, data)
    update_bc!(grid.boundary, grid, data)
end

function for_each_inner_cell(f, g::CartesianGrid{1}, include_ghostcells=0)
    for i in (g.ghostcells[1]-include_ghostcells+1):(g.totalcells[1]-g.ghostcells[1]+include_ghostcells)
        f(i-1, i, i+1)
    end
end