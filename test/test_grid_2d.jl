using SinSWE
using Test
using StaticArrays

nx = 10
ny = 6
gc = 2

grid = SinSWE.CartesianGrid(nx, ny, extent=[0 nx; 0 ny], gc=gc)
dx = SinSWE.compute_dx(grid)
dy = SinSWE.compute_dy(grid)

x_faces = collect(0:nx)
y_faces = collect(0:ny)
x_cellcenters = collect(0.5:1:9.5)
y_cellcenters = collect(0.5:1:5.5)

faces_from_grid = SinSWE.cell_faces(grid)
centers_from_grid = SinSWE.cell_centers(grid)
for j in 1:(ny+1)
    for i in 1:(nx+1)
        @test faces_from_grid[i, j][1] == x_faces[i]
        @test faces_from_grid[i, j][2] == y_faces[j]
    end
end
for j in 1:(ny)
    for i in 1:(nx)
        @test centers_from_grid[i, j][1] == x_cellcenters[i]
        @test centers_from_grid[i, j][2] == y_cellcenters[j]
    end
end

@test x_faces == SinSWE.cell_faces(grid, XDIR)
@test y_faces == SinSWE.cell_faces(grid, YDIR)
@test x_cellcenters == SinSWE.cell_centers(grid, XDIR)
@test y_cellcenters == SinSWE.cell_centers(grid, YDIR)

faces_gc = SinSWE.cell_faces(grid, interior=false)
@test faces_gc[3:end-2, 3:end-2] ≈ faces_from_grid atol = 10^-14 # Got machine epsilon round-off errors
for j in 1:2
    for i in 1:2
        @test faces_gc[i, j] ≈ SVector{2, Float64}(-3+i, -3+j) atol = 10^-14
        @test faces_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i +1), ny+ (gc-j+1)) atol = 10^-14
    end
end

cells_gc = SinSWE.cell_centers(grid, interior=false)
@test cells_gc[3:end-2, 3:end-2] ≈ centers_from_grid atol = 10^-14
for j in 1:2
    for i in 1:2
        @test cells_gc[i, j] ≈ SVector{2, Float64}(-3+i + dx/2, -3+j + dy/2)  atol = 10^-14
        @test cells_gc[end-(i-1), end-(j-1)] ≈ SVector{2, Float64}(nx+(gc-i) + dx/2, ny+ (gc-j)+dy/2) atol = 10^-14
    end
end


