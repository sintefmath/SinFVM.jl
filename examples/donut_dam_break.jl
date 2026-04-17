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

# # Donut Dam Break — Cartesian vs. Triangular Grid Comparison
#
# A 2D shallow-water experiment over a smooth donut-shaped bottom topography.
#
# **Domain:** (−30, 30) × (−30, 30) m, wall boundary on all sides.
#
# **Bottom topography (donut):**
#   - Centre at the origin, ring radius R = 10 m, peak height 2 m,
#     smoothed with a Gaussian profile of half-width σ = 2 m.
#   - B(x,y) = 2 · exp(−((r − R)/σ)²),  where r = √(x² + y²).
#
# **Initial condition:**
#   - Water surface w = 3 m everywhere  →  depth h = w − B.
#   - Dam-break column: add 1 m of water surface for r < 3 m.
#
# The simulation is run with both a **Cartesian** and a **triangular** mesh.
# Results are compared via cross-section plots and 2D heat-map snapshots.

using CairoMakie
using StaticArrays
using VolumeFluxes
using DelaunayTriangulation

# ──────────────────────────────────────────────────────────────────
# Common parameters
# ──────────────────────────────────────────────────────────────────
const R_donut   = 10.0        # donut ring radius [m]
const σ_donut   = 2.0         # Gaussian half-width [m]
const B_peak    = 2.0         # peak height of the donut [m]
const w_base    = 3.0         # base water-surface elevation [m]
const dam_r     = 3.0         # dam-break radius [m]
const dam_dw    = 1.0         # extra surface elevation inside dam [m]
const g_        = 9.81
const L         = 30.0        # half-width of the square domain [m]
const T_end     = 3.0         # simulation end time [s]
const dt_fallback = 1e-3      # fallback time step when max_speed ≈ 0 [s]

"""
    donut_bottom(x, y)

Smooth donut-shaped bottom topography centred at the origin.
"""
donut_bottom(x, y) = B_peak * exp(-((sqrt(x^2 + y^2) - R_donut) / σ_donut)^2)

"""
    initial_w(x, y)

Initial water-surface elevation: flat at `w_base` plus a dam-break column.
"""
initial_w(x, y) = w_base + (sqrt(x^2 + y^2) < dam_r ? dam_dw : 0.0)

# ══════════════════════════════════════════════════════════════════
# Part 1 — Cartesian grid simulation
# ══════════════════════════════════════════════════════════════════
function run_cartesian(; nx=120, ny=120)
    backend = make_cpu_backend()

    grid = CartesianGrid(nx, ny;
        gc   = 2,
        boundary = VolumeFluxes.WallBC(),
        extent   = [-L L; -L L],
    )

    # Bottom topography on the node grid (size = (nx+2gc+1) × (ny+2gc+1))
    x_faces = collect(VolumeFluxes.cell_faces(grid, XDIR; interior=false))
    y_faces = collect(VolumeFluxes.cell_faces(grid, YDIR; interior=false))
    B_nodes = [donut_bottom(x, y) for x in x_faces, y in y_faces]
    bottom  = VolumeFluxes.BottomTopography2D(B_nodes, backend, grid)

    bottom_source = VolumeFluxes.SourceTermBottom()
    equation      = VolumeFluxes.ShallowWaterEquations(bottom; depth_cutoff=1e-4)
    reconstruction = LinearReconstruction(1.2)
    numericalflux  = CentralUpwind(equation)

    conserved_system = ConservedSystem(
        backend, reconstruction, numericalflux, equation, grid,
        [bottom_source],
    )

    timestepper = RungeKutta2()
    simulator   = Simulator(backend, conserved_system, timestepper, grid; cfl=0.25)

    # Initial condition: the Cartesian SWE stores w = h + B as the first
    # conserved variable (water-surface elevation).
    x_centers = VolumeFluxes.cell_centers(grid)
    initial   = map(c -> @SVector[initial_w(c[1], c[2]), 0.0, 0.0], x_centers)
    VolumeFluxes.set_current_state!(simulator, initial)

    # Collect snapshots
    snapshots_cart = []
    snapshot_times = Float64[]
    function save_snapshot(t, sim)
        state = VolumeFluxes.current_interior_state(sim)
        h_data = collect(state.h)
        B_cells = VolumeFluxes.collect_topography_cells(bottom, grid)
        push!(snapshots_cart, (t=t, w=h_data, B=B_cells))  # h stores w for ShallowWaterEquations
        push!(snapshot_times, t)
    end

    save_snapshot(0.0, simulator)
    cb = IntervalWriter(step=0.5, writer=save_snapshot)
    VolumeFluxes.simulate_to_time(simulator, T_end; callback=cb)
    save_snapshot(T_end, simulator)

    # Cell-centre coordinates for plotting
    xc = collect(VolumeFluxes.cell_centers(grid, XDIR))
    yc = collect(VolumeFluxes.cell_centers(grid, YDIR))

    return (xc=xc, yc=yc, snapshots=snapshots_cart, grid=grid, bottom=bottom)
end

# ══════════════════════════════════════════════════════════════════
# Part 2 — Triangular grid simulation  (manual time loop)
# ══════════════════════════════════════════════════════════════════

"""
    generate_triangular_mesh(h_target)

Create a Delaunay triangulation of the square domain [−L,L]² with
approximate element size `h_target`, and return TriangularGrid arrays.
"""
function generate_triangular_mesh(h_target)
    # Generate a regular point cloud, then triangulate
    xs = range(-L, L; step=h_target)
    ys = range(-L, L; step=h_target)
    points = Tuple{Float64,Float64}[]
    for y in ys, x in xs
        push!(points, (x, y))
    end

    tri = triangulate(points)

    # Extract nodes
    nodes = [SVector{2,Float64}(get_point(tri, i)...) for i in 1:num_points(tri)]

    # Extract triangles and build neighbour connectivity
    tris_raw = collect(each_solid_triangle(tri))
    ntri = length(tris_raw)
    triangles = Vector{SVector{3,Int}}(undef, ntri)
    for (idx, t) in enumerate(tris_raw)
        i, j, k = triangle_vertices(t)
        triangles[idx] = SVector{3,Int}(i, j, k)
    end

    # Build edge → triangle map for neighbour lookup
    edge_to_tri = Dict{Tuple{Int,Int},Int}()
    for (idx, tri_v) in enumerate(triangles)
        for e in 1:3
            v1 = tri_v[e]
            v2 = tri_v[e % 3 + 1]
            edge_to_tri[(v1, v2)] = idx
        end
    end
    neighbors = Vector{SVector{3,Int}}(undef, ntri)
    for (idx, tri_v) in enumerate(triangles)
        nb = MVector{3,Int}(0, 0, 0)
        for e in 1:3
            v1 = tri_v[e]
            v2 = tri_v[e % 3 + 1]
            # opposite edge
            opp = get(edge_to_tri, (v2, v1), 0)
            nb[e] = opp
        end
        neighbors[idx] = SVector{3,Int}(nb)
    end

    return nodes, triangles, neighbors
end

function run_triangular(; h_target=1.0)
    nodes, triangles, neighbors = generate_triangular_mesh(h_target)
    grid = TriangularGrid(nodes, triangles, neighbors;
                          boundary=TriangularWallBC())
    eq = ShallowWaterEquationsPure(1.0, g_)

    ncells = VolumeFluxes.number_of_cells(grid)
    centroids = VolumeFluxes.cell_centers(grid)

    # Bottom topography at cell centroids (for initial condition and source term)
    B_cell = [donut_bottom(centroids[i][1], centroids[i][2]) for i in 1:ncells]

    # Initial condition: h = w - B,  hu = hv = 0
    cell_values = [SVector{3,Float64}(
        initial_w(centroids[i][1], centroids[i][2]) - B_cell[i],
        0.0, 0.0) for i in 1:ncells]

    # Manual forward-Euler time loop with CFL
    cfl = 0.25
    t   = 0.0

    snapshots_tri = [(t=0.0, h=copy(cell_values), B=copy(B_cell))]

    next_snap = 0.5
    while t < T_end
        # Reconstruction
        gradients = VolumeFluxes.reconstruct_triangular(grid, cell_values)

        # Compute fluxes
        rhs = [SVector{3,Float64}(0.0, 0.0, 0.0) for _ in 1:ncells]
        max_speed = VolumeFluxes.compute_triangular_fluxes!(rhs, grid, eq, cell_values, gradients)

        # Add bottom-topography source term:
        # S = (0, -g h ∂B/∂x, -g h ∂B/∂y)  approximated via the
        # reconstructed gradient of B (constant per cell, finite-difference
        # from neighbour centroids).
        B_sv = [SVector{1,Float64}(B_cell[i]) for i in 1:ncells]
        B_grads = VolumeFluxes.reconstruct_triangular(grid, B_sv)
        for i in 1:ncells
            h_i = cell_values[i][1]
            dBdx = B_grads[i][1][1]
            dBdy = B_grads[i][1][2]
            rhs[i] += SVector{3,Float64}(0.0, -g_ * h_i * dBdx, -g_ * h_i * dBdy)
        end

        # CFL time step
        min_dx = minimum(minimum(grid.edge_lengths[i]) for i in 1:ncells)
        dt = max_speed > 0 ? cfl * min_dx / max_speed : dt_fallback
        dt = min(dt, T_end - t)

        # Update
        for i in 1:ncells
            cell_values[i] += dt .* rhs[i]
            # Ensure positivity of h
            if cell_values[i][1] < 0
                cell_values[i] = SVector{3,Float64}(0.0, 0.0, 0.0)
            end
        end
        t += dt

        if t >= next_snap - 1e-10
            push!(snapshots_tri, (t=t, h=copy(cell_values), B=copy(B_cell)))
            next_snap += 0.5
        end
    end
    push!(snapshots_tri, (t=t, h=copy(cell_values), B=copy(B_cell)))

    return (centroids=centroids, snapshots=snapshots_tri, grid=grid, B_cell=B_cell)
end

# ══════════════════════════════════════════════════════════════════
# Part 3 — Run both simulations
# ══════════════════════════════════════════════════════════════════
@info "Running Cartesian simulation…"
cart = run_cartesian(nx=120, ny=120)

@info "Running triangular simulation…"
tri  = run_triangular(h_target=1.0)

# ══════════════════════════════════════════════════════════════════
# Part 4 — Plotting
# ══════════════════════════════════════════════════════════════════
mkpath("figs/donut")

# ── 4a. Cross-section comparison (y ≈ 0 slice) ──────────────────
function plot_cross_sections()
    fig = Figure(size=(1400, 500), fontsize=18)
    ax1 = Axis(fig[1, 1]; xlabel="x [m]", ylabel="Water surface w [m]",
               title="Cross-section at y = 0  (t = $(T_end) s)")
    ax2 = Axis(fig[1, 2]; xlabel="x [m]", ylabel="Depth h [m]",
               title="Depth cross-section at y = 0  (t = $(T_end) s)")

    # Cartesian cross section: pick row closest to y=0
    xc = cart.xc
    j0 = argmin(abs.(cart.yc))
    snap_c = cart.snapshots[end]
    w_slice = snap_c.w[:, j0]
    B_slice = snap_c.B[:, j0]
    h_slice = w_slice .- B_slice
    lines!(ax1, xc, w_slice; label="Cartesian w", linewidth=2)
    lines!(ax1, xc, B_slice; label="Bottom B", color=:brown, linewidth=2, linestyle=:dash)
    lines!(ax2, xc, h_slice; label="Cartesian h", linewidth=2)

    # Triangular cross section: collect cells near y = 0  (|y| < h_target)
    centroids = tri.centroids
    tol = 1.5   # tolerance band around y=0
    idx = [i for i in 1:length(centroids) if abs(centroids[i][2]) < tol]
    xs_t = [centroids[i][1] for i in idx]
    snap_t = tri.snapshots[end]
    h_tri  = [snap_t.h[i][1] for i in idx]
    B_tri  = [tri.B_cell[i] for i in idx]
    w_tri  = h_tri .+ B_tri
    perm = sortperm(xs_t)
    lines!(ax1, xs_t[perm], w_tri[perm]; label="Triangular w", linewidth=2, linestyle=:dot, color=:red)
    lines!(ax2, xs_t[perm], h_tri[perm]; label="Triangular h", linewidth=2, linestyle=:dot, color=:red)

    axislegend(ax1; position=:lb)
    axislegend(ax2; position=:lt)
    save("figs/donut/cross_section.png", fig; px_per_unit=2)
    display(fig)
end
plot_cross_sections()

# ── 4b. 2D heat-map snapshots (Cartesian) ───────────────────────
function plot_heatmaps()
    snaps = cart.snapshots
    nsnap = length(snaps)
    ncols = min(nsnap, 4)
    fig = Figure(size=(400*ncols, 400), fontsize=14)

    for (k, snap) in enumerate(snaps[1:min(nsnap, ncols)])
        ax = Axis(fig[1, k]; title="t = $(round(snap.t; digits=2)) s",
                  xlabel="x", ylabel="y", aspect=DataAspect())
        w_data = snap.w
        hm = heatmap!(ax, cart.xc, cart.yc, w_data;
                       colorrange=(w_base - 0.5, w_base + dam_dw + 0.5))
        if k == ncols
            Colorbar(fig[1, ncols+1], hm; label="w [m]")
        end
    end
    save("figs/donut/heatmaps_cartesian.png", fig; px_per_unit=2)
    display(fig)
end
plot_heatmaps()

# ── 4c. 2D heat-map video (Cartesian) ───────────────────────────
function make_video()
    snaps = cart.snapshots
    fig  = Figure(size=(600, 550), fontsize=16)
    ax   = Axis(fig[1, 1]; xlabel="x [m]", ylabel="y [m]", aspect=DataAspect())
    crange = (w_base - 0.5, w_base + dam_dw + 0.5)

    w_obs = Observable(snaps[1].w)
    t_obs = Observable(snaps[1].t)
    hm = heatmap!(ax, cart.xc, cart.yc, w_obs; colorrange=crange, colormap=:blues)
    Colorbar(fig[1, 2], hm; label="w [m]")
    ax.title = lift(t -> "Water surface  t = $(round(t; digits=2)) s", t_obs)

    record(fig, "figs/donut/donut_cartesian.mp4", snaps; framerate=4) do snap
        w_obs[] = snap.w
        t_obs[] = snap.t
    end
    @info "Video saved to figs/donut/donut_cartesian.mp4"
end
make_video()

@info "All plots saved under figs/donut/"
