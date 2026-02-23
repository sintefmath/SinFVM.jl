using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Two-layer SWE 2D runner (EQUILIBRIUM STORAGE)
#   U = (h1, q1, p1, w, q2, p2)   where w = h2 + B
#
# This version enforces FREE SURFACE η = h1 + w = 0 everywhere:
#   w0 = -h10
# and requires h2 = w0 - B > 0  ->  -h10 - B > 0  ->  h10 < -max(B)
# ============================================================

# ----------------------------
# Bathymetry builders
# ----------------------------
function make_bottom_quadrant_step_2d(; Bll=0.45, Bother=0.55, backend, grid)
    x_faces = SinFVM.cell_faces(grid, SinFVM.XDIR; interior=false)
    y_faces = SinFVM.cell_faces(grid, SinFVM.YDIR; interior=false)

    nxg = length(x_faces)
    nyg = length(y_faces)

    x0 = 0.5 * (x_faces[1] + x_faces[end])
    y0 = 0.5 * (y_faces[1] + y_faces[end])

    Bint = Matrix{Float64}(undef, nxg, nyg)
    @inbounds for j in 1:nyg, i in 1:nxg
        Bint[i, j] = (x_faces[i] < x0 && y_faces[j] < y0) ? Bll : Bother
    end

    return SinFVM.BottomTopography2D(Bint, backend, grid)
end

function make_bottom_cos_sin_2d(; B0=-3.0,
    Ax=0.4, Ay=0.3, mx=1, my=1, φx=0.0, φy=0.0,
    backend, grid::SinFVM.CartesianGrid{2})

    x_faces = SinFVM.cell_faces(grid, SinFVM.XDIR; interior=false)
    y_faces = SinFVM.cell_faces(grid, SinFVM.YDIR; interior=false)

    nxg = length(x_faces)
    nyg = length(y_faces)

    x0 = SinFVM.start_extent(grid, SinFVM.XDIR)
    x1 = SinFVM.end_extent(grid,   SinFVM.XDIR)
    y0 = SinFVM.start_extent(grid, SinFVM.YDIR)
    y1 = SinFVM.end_extent(grid,   SinFVM.YDIR)

    Lx = x1 - x0
    Ly = y1 - y0

    B = Matrix{Float64}(undef, nxg, nyg)
    @inbounds for j in 1:nyg, i in 1:nxg
        xhat = (x_faces[i] - x0) / Lx
        yhat = (y_faces[j] - y0) / Ly
        xhat -= floor(xhat)
        yhat -= floor(yhat)

        B[i, j] = B0 +
                  Ax * cos(2π*mx*xhat + φx) +
                  Ay * sin(2π*my*yhat + φy)
    end

    return SinFVM.BottomTopography2D(B, backend, grid)
end

# ----------------------------
# Initial conditions: η = 0 equilibrium
# ----------------------------
function ic_equilibrium_eta0(; h10=1.0, min_h=1e-10)
    w0 = -h10
    return (xy, Bcell) -> begin
        h1 = max(h10, min_h)
        h2 = w0 - Bcell
        if h2 <= 0
            error("IC violates h2>0 at xy=$xy: h2=w0-Bcell=$w0-$Bcell=$h2. " *
                  "Need h10 < -max(Bcell).")
        end
        @SVector [h1, 0.0, 0.0, w0, 0.0, 0.0]
    end
end

function interior_fields(sim, eq, grid)
    st = SinFVM.current_interior_state(sim)
    Bcell = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    h1 = collect(st.h1)
    q1 = collect(st.q1); p1 = collect(st.p1)
    w  = collect(st.w)
    q2 = collect(st.q2); p2 = collect(st.p2)

    h2 = w .- Bcell
    η  = h1 .+ w

    u1 = SinFVM.desingularize.(Ref(eq), h1, q1)
    v1 = SinFVM.desingularize.(Ref(eq), h1, p1)
    u2 = SinFVM.desingularize.(Ref(eq), h2, q2)
    v2 = SinFVM.desingularize.(Ref(eq), h2, p2)

    return (; Bcell, h1,q1,p1,w,q2,p2,h2,η,u1,v1,u2,v2)
end

# ============================================================
# Main script
# ============================================================
backend = SinFVM.make_cpu_backend()
nx, ny = 64, 64
gc = 2
grid = SinFVM.CartesianGrid(nx, ny; gc=gc, boundary=SinFVM.PeriodicBC())

# bottom = make_bottom_quadrant_step_2d(; Bll=0.45, Bother=0.55, backend=backend, grid=grid)
bottom = make_bottom_cos_sin_2d(; B0=-3.0, Ax=0.4, Ay=0.3, mx=1, my=1, backend=backend, grid=grid)
# bottom = SinFVM.ConstantBottomTopography(-3.0)

equation = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=1.00, ρ2=1.02, g=9.81)
reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
numericalflux  = SinFVM.PathConservativeCentralUpwind(equation)

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()
cs  = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
# Paper Theorem 3.1: 2D non-split positivity bound is dt ≤ min(Δx/(4a), Δy/(4b)), i.e. cfl ≤ 0.25.
# In 1D the bound is cfl ≤ 0.5; the extra factor of 2 comes from both x and y sweeps.
sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.20)

# --- IC: enforce η=0
xy_int = SinFVM.cell_centers(grid; interior=true)
B_int  = SinFVM.collect_topography_cells(equation.B, grid; interior=true)
@assert size(xy_int) == size(B_int) == SinFVM.interior_size(grid)

println("---- bathymetry checks (interior) ----")
@show extrema(B_int)
println("For η=0 with w0=-h10, need h10 < -max(B). Here -max(B) = $(-maximum(B_int)).")

h10 = 1.0                      # choose so that h10 < -max(B_int)
ic  = ic_equilibrium_eta0(h10=h10)

initial = [ic(xy_int[I], B_int[I]) for I in eachindex(xy_int)]
SinFVM.set_current_state!(sim, initial)

# --- Initial diagnostics
fld0 = interior_fields(sim, equation, grid)
println("---- initial checks (interior) ----")
@show minimum(fld0.h1) maximum(fld0.h1)
@show minimum(fld0.w)  maximum(fld0.w)
@show minimum(fld0.h2) maximum(fld0.h2)
@show maximum(abs.(fld0.u1)) maximum(abs.(fld0.v1))
@show maximum(abs.(fld0.u2)) maximum(abs.(fld0.v2))
@show minimum(fld0.η) maximum(fld0.η)  # should be ~0

# --- Plot setup
Tshow = 2.0
title = "Equilibrium test (2D): η=0 (free surface), constant h1 and w=-h1"

f = Figure(size=(1600, 900), fontsize=18)
Label(f[0, 1:2], "$title | nx=$nx, ny=$ny, T=$Tshow", fontsize=22, padding=(0,0,10,0))

ax_B  = Axis(f[1, 1], title=L"B(x,y)")
ax_w  = Axis(f[1, 2], title=L"w = h_2 + B")
ax_η  = Axis(f[2, 1], title=L"\eta = h_1 + w \;\;(\approx 0)")
ax_v2 = Axis(f[2, 2], title=L"v_2")

B_obs  = Observable(fld0.Bcell)
w_obs  = Observable(fld0.w)
η_obs  = Observable(fld0.η)
v2_obs = Observable(fld0.v2)

hm_B  = heatmap!(ax_B,  B_obs);  Colorbar(f[1, 3], hm_B,  label=L"B")
hm_w  = heatmap!(ax_w,  w_obs);  Colorbar(f[1, 4], hm_w,  label=L"w")
hm_η  = heatmap!(ax_η,  η_obs);  Colorbar(f[2, 3], hm_η,  label=L"\eta")
hm_v2 = heatmap!(ax_v2, v2_obs); Colorbar(f[2, 4], hm_v2, label=L"v_2")

display(f)

# --- Micro-step sanity
println("---- micro-step sanity (t = 1e-4) ----")
SinFVM.simulate_to_time(sim, 1e-4)
fldm = interior_fields(sim, equation, grid)
@show any(isnan, fldm.h1) any(isnan, fldm.w) any(isnan, fldm.q1) any(isnan, fldm.q2)
@show minimum(fldm.h2)
@show minimum(fldm.η) maximum(fldm.η)

# reset to IC
SinFVM.set_current_state!(sim, initial)

# --- Run
println("---- run to Tshow ----")
@time SinFVM.simulate_to_time(sim, Tshow)

# --- Final diagnostics + update plot
fld = interior_fields(sim, equation, grid)
println("---- final checks (interior) ----")
@show minimum(fld.h1) maximum(fld.h1)
@show minimum(fld.w)  maximum(fld.w)
@show minimum(fld.h2) maximum(fld.h2)
@show maximum(abs.(fld.u1)) maximum(abs.(fld.v1))
@show maximum(abs.(fld.u2)) maximum(abs.(fld.v2))
@show minimum(fld.η) maximum(fld.η)

B_obs[]  = fld.Bcell
w_obs[]  = fld.w
η_obs[]  = fld.η
v2_obs[] = fld.v2
display(f)

f