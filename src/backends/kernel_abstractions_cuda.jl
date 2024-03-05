

import Adapt

struct NewCartesianGrid{dimension,BoundaryType}
    ghostcells::SVector{dimension,Int64}
    totalcells::SVector{dimension,Int64}

    boundary::BoundaryType
    extent::SMatrix{dimension,2,Float64}
    Δx::Float64
end
function Adapt.adapt_structure(to, bc::PeriodicBC)
    return bc
end
function Adapt.adapt_structure(to, grid::CartesianGrid{dimension,BoundaryType}) where {dimension, BoundaryType}
    println("Adapting")
    ghostcells = Adapt.adapt_structure(to, grid.ghostcells)
    totalcells = Adapt.adapt_structure(to, grid.totalcells)
    boundary = Adapt.adapt_structure(to, grid.boundary)
    extent = Adapt.adapt_structure(to, grid.extent)
    Δx = Adapt.adapt_structure(to, grid.Δx)
    retval = CartesianGrid{dimension, BoundaryType}(ghostcells, totalcells, boundary, extent, Δx)
    @show isbits(retval)

    @show isbits(ghostcells)
    @show isbits(totalcells)
    @show isbits(boundary)
    @show isbits(extent)
    @show isbits(Δx)

    @show typeof(grid.ghostcells) == typeof(ghostcells)
    @show typeof(grid.totalcells) == typeof(totalcells)
    @show typeof(grid.boundary) == typeof(boundary)
    @show typeof(grid.extent) == typeof(extent)
    @show typeof(grid.Δx) == typeof(Δx)

    @show typeof(grid.ghostcells)
    @show typeof(grid.totalcells)
    @show typeof(grid.boundary)
    @show typeof(grid.extent)
    @show typeof(grid.Δx)

    for field in fieldnames(typeof(retval))
        print("isbits($(field)) = ")
        print("isbits($(getfield(retval, field))) = ")
        println("$(isbitstype(typeof(getfield(retval, field))))")
    end

    retval
end