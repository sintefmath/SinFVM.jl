using SinSWE
using Test
 # We do not technically need this loop, but adding it to create a scope
for _ in get_available_backends()
    nx = 10
    grid = SinSWE.CartesianGrid(nx, extent=[0 nx], gc=2)

    x_faces = collect(0:nx)
    x_cellcenters = collect(0.5:1:9.5)

    @test SinSWE.cell_faces(grid) == x_faces
    @test SinSWE.cell_centers(grid) == x_cellcenters

    const_b = 3.14
    const_B = SinSWE.constant_bottom_topography(grid, const_b)

    @test size(const_B)[1] == nx + 4 + 1
    for b in const_B
        @test b == const_b
    end

    x_faces_gc = SinSWE.cell_faces(grid, interior=false)
    @test x_faces_gc[3:end-2] ≈ x_faces atol = 10^-14 # Got machine epsilon round-off errors
    @test x_faces_gc[1:2] ≈ [-2, -1] atol = 10^-14
    @test x_faces_gc[end-1:end] ≈ [11, 12] atol = 10^-14

    x_cells_gc = SinSWE.cell_centers(grid, interior=false)
    @test x_cells_gc[3:end-2] ≈ x_cellcenters atol = 10^-14
    @test x_cells_gc[1:2] ≈ [-1.5, -0.5] atol = 10^-14
    @test x_cells_gc[end-1:end] ≈ [10.5, 11.5] atol = 10^-14
end