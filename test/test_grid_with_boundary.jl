using SinFVM
using Test
for backend in get_available_backends()
    backend = SinFVM.make_cpu_backend()
    nx = 10
    ny = 12

    cartesian_grid = SinFVM.CartesianGrid(nx, ny; gc=2)
    grid_with_boundary = SinFVM.GridWithBoundary(cartesian_grid, backend)

    is_active = ones(Bool, size(cartesian_grid))

    # Make an L-shape not active

    is_active[3:5, 2] .= false
    is_active[5, 2:7] .= false

    grid_with_boundary_and_non_active_cells = SinFVM.GridWithBoundary(cartesian_grid, 
        SinFVM.convert_to_backend(backend, is_active))

    recorded_active = SinFVM.convert_to_backend(backend, zeros(Bool, size(is_active)))
    SinFVM.@fvmloop SinFVM.for_each_cell(backend, grid_with_boundary_and_non_active_cells) do index
        recorded_active[index] = true
    end

    @test all(collect(recorded_active) .== is_active)

    # Now test with ghost cells
    is_active_with_bc = copy(is_active)

    function grow_inactive(active)
        active_copy = copy(active)
        for (i, j) in CartesianIndices(active)
            if !active[i,j]
                active_copy[i+1, j] = false
                active_copy[i-1, j] = false
                active_copy[i, j+1] = false
                active_copy[i, j-1] = false
                active_copy[i+1, j+1] = false
                active_copy[i+1, j-1] = false
                active_copy[i-1, j+1] = false
                active_copy[i-1, j-1] = false
            end
        end
        return active_copy
    end

    active_with_bc_recorded = SinFVM.convert_to_backend(backend, zeros(Bool, size(is_active)))

    SinFVM.@fvmloop SinFVM.for_each_inner_cell(backend, grid_with_boundary_and_non_active_cells,  XDIR) do left, middle, right

    end
end
