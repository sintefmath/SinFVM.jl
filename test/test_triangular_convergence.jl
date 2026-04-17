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

# Convergence test:  As the triangular and Cartesian meshes are refined,
# the cell-averaged solutions of the shallow-water equations with a
# **constant** bottom topography and an initial water-height bump should
# converge to the same solution — so the difference between them must go
# to zero under refinement.

using VolumeFluxes
using StaticArrays
using Test
using LinearAlgebra
import CUDA

# --------------------------------------------------------------------------
# Build a regular triangular grid of the unit square [0,1]^2 by splitting
# each cell of an N×N Cartesian grid into two triangles.
# --------------------------------------------------------------------------
function make_regular_triangular_grid(N::Int; boundary=TriangularWallBC())
    h = 1.0 / N
    nodes = Vector{SVector{2,Float64}}(undef, (N + 1)^2)
    node_id(i, j) = (j - 1) * (N + 1) + i
    for j in 1:(N + 1), i in 1:(N + 1)
        nodes[node_id(i, j)] = SVector((i - 1) * h, (j - 1) * h)
    end

    ntri = 2 * N * N
    triangles = Vector{SVector{3,Int}}(undef, ntri)
    neighbors = Vector{SVector{3,Int}}(undef, ntri)

    # Tri index for lower / upper triangle of cell (i, j):
    lower_id(i, j) = 2 * ((j - 1) * N + (i - 1)) + 1
    upper_id(i, j) = 2 * ((j - 1) * N + (i - 1)) + 2

    for j in 1:N, i in 1:N
        tl = lower_id(i, j)
        tu = upper_id(i, j)

        # Lower triangle: (i,j) (i+1,j) (i+1,j+1)
        triangles[tl] = SVector(node_id(i, j), node_id(i + 1, j), node_id(i + 1, j + 1))
        # Upper triangle: (i,j) (i+1,j+1) (i,j+1)
        triangles[tu] = SVector(node_id(i, j), node_id(i + 1, j + 1), node_id(i, j + 1))

        # Lower edges: (v1→v2) south, (v2→v3) east, (v3→v1) diagonal
        nb_s = j > 1 ? upper_id(i, j - 1) : 0
        nb_e = i < N ? lower_id(i + 1, j) : 0
        nb_d = tu                         # diagonal always shares with upper triangle
        neighbors[tl] = SVector(nb_s, nb_e, nb_d)

        # Upper edges: (v1→v2) diagonal (shared with lower), (v2→v3) north, (v3→v1) west
        nb_n = j < N ? lower_id(i, j + 1) : 0
        nb_w = i > 1 ? upper_id(i - 1, j) : 0
        neighbors[tu] = SVector(tl, nb_n, nb_w)
    end

    return TriangularGrid(nodes, triangles, neighbors; boundary=boundary)
end

# Bump initial condition for water height on the unit square.
_bump(x, y) = 1.0 + 0.2 * exp(-80.0 * ((x - 0.5)^2 + (y - 0.5)^2))

# --------------------------------------------------------------------------
# Triangular simulation:  piecewise-linear reconstruction + central-upwind
# flux + forward Euler time integration.  Uses the triangular looping
# abstractions via `reconstruct_triangular` and `compute_triangular_fluxes!`.
# --------------------------------------------------------------------------
function run_triangular(backend, N::Int, T_end::Float64)
    eq = VolumeFluxes.ShallowWaterEquationsPure()
    grid = make_regular_triangular_grid(N; boundary=TriangularWallBC())
    ncells = VolumeFluxes.number_of_cells(grid)

    # Initial condition (constant bottom => h = water height directly).
    cell_values = Vector{SVector{3,Float64}}(undef, ncells)
    for i in 1:ncells
        cx, cy = grid.centroids[i]
        cell_values[i] = SVector(_bump(cx, cy), 0.0, 0.0)
    end

    cfl = 0.4
    t = 0.0
    h_min = 1.0 / N

    while t < T_end
        gradients = VolumeFluxes.reconstruct_triangular(backend, grid, cell_values)
        rhs = Vector{SVector{3,Float64}}(undef, ncells)
        max_speed = VolumeFluxes.compute_triangular_fluxes!(
            backend, rhs, grid, eq, cell_values, gradients)

        dt = (max_speed > 0) ? cfl * h_min / max_speed : T_end - t
        dt = min(dt, T_end - t)

        for i in 1:ncells
            cell_values[i] = cell_values[i] + dt * rhs[i]
        end
        t += dt
    end

    h = [cell_values[i][1] for i in 1:ncells]
    centroids = grid.centroids
    return h, centroids
end

# --------------------------------------------------------------------------
# Cartesian simulation:  same equations, same initial condition.  Uses
# the standard `Simulator` pipeline with `ConservedSystem`.
# --------------------------------------------------------------------------
function run_cartesian(backend, N::Int, T_end::Float64)
    eq = VolumeFluxes.ShallowWaterEquationsPure()
    grid = VolumeFluxes.CartesianGrid(N, N; gc=2, boundary=VolumeFluxes.WallBC(),
                                      extent=[0.0 1.0; 0.0 1.0])

    reconstruction = LinearReconstruction()
    flux = CentralUpwind(eq)
    system = ConservedSystem(backend, reconstruction, flux, eq, grid)
    timestepper = VolumeFluxes.ForwardEulerStepper()
    simulator = VolumeFluxes.Simulator(backend, system, timestepper, grid)

    # Initial condition via a Volume
    init_volume = VolumeFluxes.Volume(backend, eq, grid)
    xs = VolumeFluxes.cell_centers(grid)
    CUDA.@allowscalar VolumeFluxes.InteriorVolume(init_volume)[1:end, 1:end] =
        [SVector{3, Float64}(_bump(xi[1], xi[2]), 0.0, 0.0) for xi in xs]
    VolumeFluxes.set_current_state!(simulator, init_volume)

    VolumeFluxes.simulate_to_time(simulator, T_end; maximum_timestep=1e-3)

    interior = VolumeFluxes.current_interior_state(simulator)
    h_field = collect(interior.h)

    # Return flattened h and centroids in the same column-major order as interior
    h = Vector{Float64}(undef, N * N)
    centroids = Vector{SVector{2,Float64}}(undef, N * N)
    for j in 1:N, i in 1:N
        idx = (j - 1) * N + i
        h[idx] = h_field[i, j]
        centroids[idx] = SVector((i - 0.5) / N, (j - 0.5) / N)
    end
    return h, centroids
end

# --------------------------------------------------------------------------
# Compare: for each triangular centroid, sample the Cartesian solution at
# the enclosing cell and take the RMS difference in water height.
# --------------------------------------------------------------------------
function rms_difference(h_tri, tri_centroids, h_cart, N_cart)
    diffs = similar(h_tri)
    for (k, c) in enumerate(tri_centroids)
        # Find enclosing Cartesian cell
        ci = clamp(Int(floor(c[1] * N_cart)) + 1, 1, N_cart)
        cj = clamp(Int(floor(c[2] * N_cart)) + 1, 1, N_cart)
        cart_idx = (cj - 1) * N_cart + ci
        diffs[k] = h_tri[k] - h_cart[cart_idx]
    end
    return sqrt(sum(d -> d^2, diffs) / length(diffs))
end

# --------------------------------------------------------------------------
# Run the convergence experiment: short time so BC effects are minimal.
# --------------------------------------------------------------------------
function test_triangular_vs_cartesian_convergence(backend)
    T_end = 0.02
    Ns = (8, 16, 32)
    errors = Float64[]

    for N in Ns
        h_tri,  tri_c  = run_triangular(backend, N, T_end)
        h_cart, _cart_c = run_cartesian(backend, N, T_end)
        e = rms_difference(h_tri, tri_c, h_cart, N)
        push!(errors, e)
    end

    # All errors should be finite and strictly decreasing
    for e in errors
        @test isfinite(e)
    end

    # Refining must reduce the inter-method discrepancy
    @test errors[end] < errors[1]
    @test errors[2]   < errors[1]

    # The finest error should be small
    @test errors[end] < 0.02
end

# Run on CPU backend only — convergence is a numerical correctness test
# independent of hardware.
test_triangular_vs_cartesian_convergence(make_cpu_backend())
