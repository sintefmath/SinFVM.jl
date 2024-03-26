
function update_bc!(backend, ::PeriodicBC, grid::CartesianGrid{1}, ::Equation, data)
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[grid.totalcells[1]+ghostcell-2*grid.ghostcells[1]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell)] = data[grid.ghostcells[1]+ghostcell]
    end
end

function update_bc!(backend, ::PeriodicBC, grid::CartesianGrid{2}, ::Equation, data)
    # TODO: Introduce some helper functions here...
    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[grid.totalcells[1]+ghostcell[1]-2*grid.ghostcells[1], ghostcell[2]]
        data[grid.totalcells[1]-(grid.ghostcells[1]-ghostcell[1]), ghostcell[2]] = data[grid.ghostcells[1]+ghostcell[1], ghostcell[2]]
    end

    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        data[ghostcell] = data[ghostcell[1], grid.totalcells[2]+ghostcell[2]-2*grid.ghostcells[2]]
        data[ghostcell[1], grid.totalcells[2]-(grid.ghostcells[2]-ghostcell[2])] = data[ghostcell[1], grid.ghostcells[2]+ghostcell[2]]
    end
end


function update_bc!(backend, ::WallBC, grid::CartesianGrid{1}, ::ShallowWaterEquations1D, data)


    function local_update_ghostcell!(data, ghost, inner)
        # h(ghost) = h(inner) and hu(ghost) = -hu(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], -data[inner][2])
    end

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = grid.ghostcells[1] * 2 + 1 - ghostcell
        local_update_ghostcell!(data, left_ghostcell, left_innercell)

        right_ghostcell = grid.totalcells[1] - grid.ghostcells[1] + ghostcell
        right_innercell = grid.totalcells[1] - grid.ghostcells[1] - ghostcell + 1
        local_update_ghostcell!(data, right_ghostcell, right_innercell)
    end
end

update_bc!(backend, grid::CartesianGrid, eq::Equation, data) = update_bc!(backend, grid.boundary, grid, eq, data)
update_bc!(simulator::Simulator, data) = update_bc!(simulator.backend, simulator.grid, simulator.system.equation, data)
