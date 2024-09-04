struct GridWithBoundary{GridType, dimension} <: Grid{dimension}
    grid::GridType

    GridWithBoundary(grid) = new{typeof(grid), dimension(grid)}(grid)
end

const CartesianGridWithBoundary{dimension} = GridWithBoundary{<:CartesianGrid, dimension} where dimension

start_extent(grid::CartesianGridWithBoundary, direction) = start_extent(grid.grid, direction)
end_extent(grid::CartesianGridWithBoundary, direction) = end_extent(grid.grid, direction)

number_of_interior_cells(grid::CartesianGridWithBoundary) = number_of_interior_cells(grid.grid)

interior_size(grid::CartesianGridWithBoundary) = interior_size(grid.grid)

interior_size(grid::CartesianGridWithBoundary, direction) = interior_size(grid, direction)

Base.size(grid::CartesianGridWithBoundary) = Base.size(grid.grid)


cell_faces(grid::CartesianGridWithBoundary{1}; interior=true) = cell_faces(grid.grid; interior=interior)

cell_faces(grid::CartesianGridWithBoundary{2}, dir::Direction; interior=true) = cell_faces(grid.grid, dir; interior=interior)

cell_center(grid::CartesianGridWithBoundary{2}, I::CartesianIndex) = cell_center(grid.grid, I)

cell_faces(grid::CartesianGridWithBoundary{2}; interior=true) = cell_faces(grid.grid; interior=interior)

cell_centers(grid::CartesianGridWithBoundary{1}; interior=true) = cell_centers(grid.grid; interior=interior)

cell_centers(grid::CartesianGridWithBoundary{2}, dir::Direction; interior=true) = cell_centers(grid.grid, dir; interior=interior)

cell_centers(grid::CartesianGridWithBoundary{2}; interior=true) = cell_centers(grid.grid; interior=interior)
    
compute_dx(grid::CartesianGridWithBoundary, direction=XDIR) = compute_dx(grid.grid, direction)
compute_dy(grid::CartesianGridWithBoundary{2}) = compute_dx(grid.grid)

compute_cell_size(grid::CartesianGridWithBoundary) = compute_cell_size(grid.grid)

constant_bottom_topography(grid::CartesianGridWithBoundary, value) = constant_bottom_topography(grid.grid, value)


function inner_cells(g::CartesianGridWithBoundary, direction, ghostcells=g.grid.ghostcells[direction])
    return inner_cells(g.grid, direction, ghostcells)
end

function left_cell(g::CartesianGridWithBoundary, I::Int64, direction, ghostcells=g.grid.ghostcells[direction])
    return left_cell(g.grid, I, direction, ghostcells)
end

function middle_cell(g::CartesianGridWithBoundary, I::Int64, direction, ghostcells=g.grid.ghostcells[direction])
    return middle_cell(g.grid, I, direction, ghostcells)
end


function right_cell(g::CartesianGridWithBoundary, I::Int64, direction, ghostcells=g.grid.ghostcells[direction])
    return right_cell(g.grid, I, direction, ghostcells)
end


function left_cell(g::CartesianGridWithBoundary, I::CartesianIndex, direction, ghostcells=g.grid.ghostcells[direction])
    return left_cell(g.grid, I, direction, ghostcells)
end

function middle_cell(g::CartesianGridWithBoundary, I::CartesianIndex, direction, ghostcells=g.grid.ghostcells[direction])
    return middle_cell(g.grid, I, direction, ghostcells)
end


function right_cell(g::CartesianGridWithBoundary, I::CartesianIndex, direction, ghostcells=g.grid.ghostcells[direction])
    return right_cell(g.grid, I, direction, ghostcells)
end

function ghost_cells(g::CartesianGridWithBoundary, direction)
    return ghost_cells(g.grid, direction)
end

