
update_bc!(backend, grid::CartesianGridWithBoundary, eq::Equation, data) = update_bc!(backend, grid.grid, eq, data)
update_bc!(backend, bc, grid::CartesianGridWithBoundary, eq, data) = update_bc!(backend, bc, grid.grid, eq, data)