
function update_bc!(backend, ::PeriodicBC, grid::CartesianGrid{1}, data)
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[grid.totalcells[1]+ghostcell-2*grid.ghostcells[1]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell)] = data[grid.ghostcells[1]+ghostcell]
    end
end

function update_bc!(backend, grid::CartesianGrid{1}, data)
    update_bc!(backend, grid.boundary, grid, data)
end