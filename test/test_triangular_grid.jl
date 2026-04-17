# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using VolumeFluxes
using Test
using StaticArrays

# ------------------------------------------------------------------
# Helper: build a simple two-triangle mesh on the unit square
#
#   4 ---- 3
#   |  T2 /|
#   |   /  |
#   | /  T1|
#   1 ---- 2
#
# ------------------------------------------------------------------
function make_unit_square_mesh(; boundary=OutflowBC())
    nodes = [
        SVector(0.0, 0.0),  # 1
        SVector(1.0, 0.0),  # 2
        SVector(1.0, 1.0),  # 3
        SVector(0.0, 1.0),  # 4
    ]
    triangles = [
        SVector(1, 2, 3),   # T1
        SVector(1, 3, 4),   # T2
    ]
    # T1 edge 1 (1→2): boundary, edge 2 (2→3): boundary, edge 3 (3→1): shared with T2
    # T2 edge 1 (1→3): shared with T1, edge 2 (3→4): boundary, edge 3 (4→1): boundary
    neighbors = [
        SVector(0, 0, 2),   # T1: edges 1,2 are boundary, edge 3 neighbours T2
        SVector(1, 0, 0),   # T2: edge 1 neighbours T1, edges 2,3 are boundary
    ]
    return TriangularGrid(nodes, triangles, neighbors; boundary=boundary)
end

# ==================================================================
# 1. Grid construction
# ==================================================================

grid = make_unit_square_mesh()

@test VolumeFluxes.number_of_cells(grid) == 2
@test length(VolumeFluxes.cell_centers(grid)) == 2
@test grid isa VolumeFluxes.Grid{2}

# Centroids of the two right-triangles on the unit square
c1 = (SVector(0.0,0.0) + SVector(1.0,0.0) + SVector(1.0,1.0)) / 3
c2 = (SVector(0.0,0.0) + SVector(1.0,1.0) + SVector(0.0,1.0)) / 3
@test grid.centroids[1] ≈ c1 atol=1e-14
@test grid.centroids[2] ≈ c2 atol=1e-14

# Each triangle has area = 0.5
@test grid.areas[1] ≈ 0.5 atol=1e-14
@test grid.areas[2] ≈ 0.5 atol=1e-14

# Edge lengths: for T1 (0,0)→(1,0) = 1, (1,0)→(1,1) = 1, (1,1)→(0,0) = √2
@test grid.edge_lengths[1] ≈ SVector(1.0, 1.0, sqrt(2.0)) atol=1e-14

# ==================================================================
# 2. Boundary conditions
# ==================================================================

# --- OutflowBC ---
U = SVector(1.0, 2.0, 3.0)
@test VolumeFluxes.boundary(OutflowBC(), U) == U

# --- TriangularWallBC ---
normal_x = SVector(1.0, 0.0)
normal_y = SVector(0.0, 1.0)
U_wall = SVector(1.0, 2.0, 3.0)  # h=1, hu=2, hv=3

# Wall with x-normal: reflect hu, keep hv
U_reflected_x = VolumeFluxes.boundary(TriangularWallBC(), U_wall, normal_x)
@test U_reflected_x[1] ≈ 1.0 atol=1e-14   # h unchanged
@test U_reflected_x[2] ≈ -2.0 atol=1e-14   # hu reflected
@test U_reflected_x[3] ≈ 3.0 atol=1e-14    # hv unchanged

# Wall with y-normal: keep hu, reflect hv
U_reflected_y = VolumeFluxes.boundary(TriangularWallBC(), U_wall, normal_y)
@test U_reflected_y[1] ≈ 1.0 atol=1e-14
@test U_reflected_y[2] ≈ 2.0 atol=1e-14
@test U_reflected_y[3] ≈ -3.0 atol=1e-14

# Wall with diagonal normal: m·n = (2·1/√2 + 3·1/√2),
# reflected = m - 2(m·n)n
n45 = SVector(1.0, 1.0) / sqrt(2.0)
U_ref_45 = VolumeFluxes.boundary(TriangularWallBC(), U_wall, n45)
@test U_ref_45[1] ≈ 1.0 atol=1e-14
mn = 2.0 * n45[1] + 3.0 * n45[2]
@test U_ref_45[2] ≈ 2.0 - 2*mn*n45[1] atol=1e-14
@test U_ref_45[3] ≈ 3.0 - 2*mn*n45[2] atol=1e-14

# ==================================================================
# 3. Reconstruction (triangular minmod)
# ==================================================================

# Uniform state → zero gradient
eq = ShallowWaterEquationsPure()
uniform_vals = [SVector(1.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)]
grads = VolumeFluxes.reconstruct_triangular(grid, uniform_vals)
for i in 1:2
    for v in 1:3
        @test grads[i][v] ≈ SVector(0.0, 0.0) atol=1e-14
    end
end

# ==================================================================
# 4. Physical flux & eigenvalues
# ==================================================================

U_test = SVector(2.0, 1.0, 0.5)
n = SVector(1.0, 0.0)
F = VolumeFluxes.physical_flux(eq, U_test, n)
h, hu, hv = 2.0, 1.0, 0.5
un = hu / h  # = 0.5
@test F[1] ≈ h * un atol=1e-14
@test F[2] ≈ hu * un + 0.5 * 9.81 * h^2 atol=1e-14
@test F[3] ≈ hv * un atol=1e-14

λ_max, λ_min = VolumeFluxes.normal_eigenvalues(eq, U_test, n)
@test λ_max ≈ un + sqrt(9.81 * h) atol=1e-14
@test λ_min ≈ un - sqrt(9.81 * h) atol=1e-14

# ==================================================================
# 5. Central-upwind flux (identical states → physical flux)
# ==================================================================

F_hat, speed = VolumeFluxes.central_upwind_flux(eq, U_test, U_test, n)
F_phys = VolumeFluxes.physical_flux(eq, U_test, n)
@test F_hat ≈ F_phys atol=1e-12

# ==================================================================
# 6. Full triangular flux computation – lake at rest
# ==================================================================

# Uniform "lake at rest": h = const, hu = hv = 0 everywhere.
# The right-hand side (time derivative) should be zero.
grid_lr = make_unit_square_mesh()
vals = [SVector(1.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)]
grads_lr = VolumeFluxes.reconstruct_triangular(grid_lr, vals)
output = [SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)]
max_speed = VolumeFluxes.compute_triangular_fluxes!(output, grid_lr, eq, vals, grads_lr)

@test output[1] ≈ SVector(0.0, 0.0, 0.0) atol=1e-12
@test output[2] ≈ SVector(0.0, 0.0, 0.0) atol=1e-12

# ==================================================================
# 7. Boundary type parameterisation
# ==================================================================

grid_wall = make_unit_square_mesh(boundary=TriangularWallBC())
@test grid_wall isa TriangularGrid{TriangularWallBC}
@test grid_wall.boundary isa TriangularWallBC

grid_out = make_unit_square_mesh(boundary=OutflowBC())
@test grid_out isa TriangularGrid{OutflowBC}

# ==================================================================
# 8. Wall boundary flux computation – lake at rest with wall
# ==================================================================

grid_wall_lr = make_unit_square_mesh(boundary=TriangularWallBC())
vals_w = [SVector(1.0, 0.0, 0.0), SVector(1.0, 0.0, 0.0)]
grads_w = VolumeFluxes.reconstruct_triangular(grid_wall_lr, vals_w)
output_w = [SVector(0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)]
VolumeFluxes.compute_triangular_fluxes!(output_w, grid_wall_lr, eq, vals_w, grads_w)
@test output_w[1] ≈ SVector(0.0, 0.0, 0.0) atol=1e-12
@test output_w[2] ≈ SVector(0.0, 0.0, 0.0) atol=1e-12

# ==================================================================
# 9. Edge normal orientation check
# ==================================================================

# All outward normals should point away from the cell centroid
for i in 1:VolumeFluxes.number_of_cells(grid)
    ci = grid.centroids[i]
    for k in 1:3
        vi1 = grid.triangles[i][k]
        vi2 = grid.triangles[i][k % 3 + 1]
        edge_mid = (grid.nodes[vi1] + grid.nodes[vi2]) / 2.0
        n_k = grid.edge_normals[i][k]
        # The outward normal dotted with (edge_mid - centroid) should be non-negative
        # (or at least consistent, depending on triangle shape)
        # For a convex cell: n · (edge_mid - centroid) >= 0
        @test n_k[1] * (edge_mid[1] - ci[1]) + n_k[2] * (edge_mid[2] - ci[2]) >= -1e-14
    end
end
