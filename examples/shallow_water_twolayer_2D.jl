using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Two-layer SWE 2D runner (EQUILIBRIUM STORAGE)
#   U = (h1, q1, p1, w, q2, p2)   where w = h2 + B
# ============================================================

xwrap(x) = x - floor(x)

bottom_cosine_2d(; B0=-3.0, A=0.4, mx=1, my=1) =
    (x, y) -> (B0 + A*cos(2π*mx*x) * cos(2π*my*y))
function make_bottom_intersections_2d(Bfun, backend, grid)
    nxg, nyg = size(grid) .+ 1   # includes ghosts automatically

    x0 = SinFVM.start_extent(grid, SinFVM.XDIR)
    x1 = SinFVM.end_extent(grid,   SinFVM.XDIR)
    y0 = SinFVM.start_extent(grid, SinFVM.YDIR)
    y1 = SinFVM.end_extent(grid,   SinFVM.YDIR)

    Lx = x1 - x0
    Ly = y1 - y0

    dx = Lx / (size(grid)[1])
    dy = Ly / (size(grid)[2])

    gcx, gcy = grid.ghostcells

    Bint = Matrix{Float64}(undef, nxg, nyg)

    for j in 1:nyg
        for i in 1:nxg
            x = x0 + (i - gcx - 1)*dx
            y = y0 + (j - gcy - 1)*dy

            x̂ = xwrap((x - x0)/Lx)
            ŷ = xwrap((y - y0)/Ly)

            Bint[i,j] = Bfun(x̂, ŷ)
        end
    end

    return SinFVM.BottomTopography2D(Bint, backend, grid)
end



function ic_equilibrium_w(; h10=1.0, w0=-1.0, min_h=1e-10)
    return (xy, Bcell) -> begin
        h1 = max(h10, min_h)
        h2 = w0 - Bcell
        if h2 <= 0
            error("IC makes h2<=0 at xy=$xy: h2=w0-Bcell=$w0-$Bcell=$h2. Choose w0 > max(Bcell).")
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
nx, ny = 10, 10
gc = 2
grid = SinFVM.CartesianGrid(nx, ny; gc=gc, boundary=SinFVM.PeriodicBC())

# --- Bathymetry as before (intersections incl ghosts)
B0, A, mx, my = -3.0, 0.4, 1, 1
Bfun   = bottom_cosine_2d(B0=B0, A=A, mx=mx, my=my)

#bottom = make_bottom_intersections_2d(Bfun, backend, grid)
bottom = ConstantBottomTopography(B0, backend, grid)  # for testing

equation = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=1.00, ρ2=1.02, g=9.81)
numericalflux = SinFVM.CentralUpwind(equation)
reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()
cs  = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.6)

# --- IC MUST be interior-sized for set_current_state!
ic = ic_equilibrium_w(h10=1.0, w0=-1.0)

xy_int = SinFVM.cell_centers(grid; interior=true)
B_int  = SinFVM.collect_topography_cells(equation.B, grid; interior=true)

@assert size(xy_int) == size(B_int) == SinFVM.interior_size(grid)

initial = [ic(xy_int[I], B_int[I]) for I in eachindex(xy_int)]
SinFVM.set_current_state!(sim, initial)

# --- Initial diagnostics
fld0 = interior_fields(sim, equation, grid)
println("---- initial checks (interior) ----")
@show extrema(fld0.Bcell)
@show minimum(fld0.h1) maximum(fld0.h1)
@show minimum(fld0.w)  maximum(fld0.w)
@show minimum(fld0.h2) maximum(fld0.h2)
@show maximum(abs.(fld0.u1)) maximum(abs.(fld0.v1))
@show maximum(abs.(fld0.u2)) maximum(abs.(fld0.v2))
@show minimum(fld0.η) maximum(fld0.η)

# --- Plot setup (Observables so it updates)
Tshow = 0.1
title = "Equilibrium test (2D): constant h1 and constant w on cosine bathymetry"

f = Figure(size=(1600, 900), fontsize=18)
Label(f[0, 1:2], "$title | nx=$nx, ny=$ny, T=$Tshow", fontsize=22, padding=(0,0,10,0))

ax_B  = Axis(f[1, 1], title=L"B(x,y)")
ax_w  = Axis(f[1, 2], title=L"w = h_2 + B")
ax_η  = Axis(f[2, 1], title=L"\eta = h_1 + w")
ax_u2 = Axis(f[2, 2], title=L"u_2")

B_obs  = Observable(fld0.Bcell)
w_obs  = Observable(fld0.w)
η_obs  = Observable(fld0.η)
u2_obs = Observable(fld0.u2)

hm_B  = heatmap!(ax_B,  B_obs);  Colorbar(f[1, 3], hm_B,  label=L"B")
hm_w  = heatmap!(ax_w,  w_obs);  Colorbar(f[1, 4], hm_w,  label=L"w")
hm_η  = heatmap!(ax_η,  η_obs);  Colorbar(f[2, 3], hm_η,  label=L"\eta")
hm_u2 = heatmap!(ax_u2, u2_obs); Colorbar(f[2, 4], hm_u2, label=L"u_2")

display(f)

# --- Micro-step sanity
println("---- micro-step sanity (t = 1e-4) ----")
SinFVM.simulate_to_time(sim, 1e-4)
fldm = interior_fields(sim, equation, grid)
@show any(isnan, fldm.h1) any(isnan, fldm.w) any(isnan, fldm.q1) any(isnan, fldm.q2)
@show minimum(fldm.h2)

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
u2_obs[] = fld.u2
display(f)

f
