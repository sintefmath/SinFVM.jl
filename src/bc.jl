# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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

function update_bc!(backend, ::NeumannBC, grid::CartesianGrid{1}, ::Equation, data)
    g = grid.ghostcells[1]
    N = grid.totalcells[1]

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        data[ghostcell] = data[g + 1]
        data[N - g + ghostcell] = data[N - g]
    end
end

function update_bc!(backend, ::NeumannBC, grid::CartesianGrid{2}, ::Equation, data)
    gx, gy = grid.ghostcells
    Nx, Ny = grid.totalcells

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        i, j = ghostcell[1], ghostcell[2]
        data[i, j] = data[gx + 1, j]                # Left ghost cells: copy first interior column
        data[Nx - gx + i, j] = data[Nx - gx, j]     # Right ghost cells: copy last interior column
    end

    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        i, j = ghostcell[1], ghostcell[2]
        data[i, j] = data[i, gy + 1]                # Bottom ghost cells: copy first interior row
        data[i, Ny - gy + j] = data[i, Ny - gy]     # Top ghost cells: copy last interior row
    end
end


function update_bc!(backend, ::WallBC, grid::CartesianGrid{1}, ::AllSWE1D, data)
    function local_update_ghostcell!(data, ghost, inner)
        # Odd-indexed components are depth-like (keep), even-indexed are momenta (negate).
        # Works for both 1-layer (h, hu) and 2-layer (h1, q1, w, q2).
        v = data[inner]
        data[ghost] = typeof(data[ghost])(ntuple(i -> isodd(i) ? v[i] : -v[i], length(v)))
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


function update_bc!(backend, ::WallBC, grid::CartesianGrid{2}, ::AllSWE2D, data)

    function local_update_ghostcell!(data, ghost, inner, ::XDIRT)
        # h(ghost) = h(inner),  hu(ghost) = -hu(inner), hv(ghost) = hv(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], -data[inner][2], data[inner][3])
    end
        function local_update_ghostcell!(data, ghost, inner, ::YDIRT)
        # h(ghost) = h(inner),  hu(ghost) = hu(inner), hv(ghost) = -hv(inner)
        data[ghost] = typeof(data[ghost])(data[inner][1], data[inner][2], -data[inner][3])
    end

    @fvmloop for_each_ghost_cell(backend, grid, XDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = CartesianIndex(grid.ghostcells[1] * 2 + 1 - ghostcell[1], ghostcell[2])
        local_update_ghostcell!(data, left_ghostcell, left_innercell, XDIR)

        right_ghostcell = CartesianIndex(grid.totalcells[1] - grid.ghostcells[1] + ghostcell[1], ghostcell[2])
        right_innercell = CartesianIndex(grid.totalcells[1] - grid.ghostcells[1] - ghostcell[1] + 1, ghostcell[2])
        local_update_ghostcell!(data, right_ghostcell, right_innercell, XDIR)
    end
    @fvmloop for_each_ghost_cell(backend, grid, YDIR) do ghostcell
        left_ghostcell = ghostcell
        left_innercell = CartesianIndex(ghostcell[1], grid.ghostcells[2] * 2 + 1 - ghostcell[2])
        local_update_ghostcell!(data, left_ghostcell, left_innercell, YDIR)

        right_ghostcell = CartesianIndex(ghostcell[1], grid.totalcells[2] - grid.ghostcells[2] + ghostcell[2])
        right_innercell = CartesianIndex(ghostcell[1], grid.totalcells[2] - grid.ghostcells[2] - ghostcell[2] + 1)
        local_update_ghostcell!(data, right_ghostcell, right_innercell, YDIR)
    end
end


update_bc!(backend, grid::CartesianGrid, eq::Equation, data) = update_bc!(backend, grid.boundary, grid, eq, data)
update_bc!(simulator::Simulator, data) = update_bc!(simulator.backend, simulator.grid, simulator.system.equation, data)
