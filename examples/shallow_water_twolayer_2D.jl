using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Two-layer SWE 2D runner (PHYSICAL STORAGE)
#   U = (h1, q1, p1, h2, q2, p2)
# Reconstruction internally uses equilibrium variable ω = h2 + B
# but returns face states in PHYSICAL variables.
# ============================================================

backend = SinFVM.make_cpu_backend()
nx, ny = 32, 32
grid = SinFVM.CartesianGrid(nx, ny; gc=2, boundary=SinFVM.PeriodicBC())

B0 = -3.0
bottom = SinFVM.ConstantBottomTopography(B0)

equation = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=1.00, ρ2=1.02, g=9.81)
numericalflux = SinFVM.CentralUpwind(equation)

reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = SinFVM.ConservedSystem(
    backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src]
)

timestepper = SinFVM.RungeKutta2()
simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid; cfl=0.6)

# ------------------------------------------------------------
# Grid data (FULL grid incl. ghost cells) for initialization
# ------------------------------------------------------------
# IMPORTANT: keep x, Bvals, and initial consistent (same indexing space)
x_all    = SinFVM.cell_centers(grid)  # typically includes ghost cells when gc>0
B_all    = SinFVM.collect_topography_cells(equation.B, grid; interior=false)

ε0 = 0.0

u1fun(xy) = 0.0
v1fun(xy) = 0.0
u2fun(xy) = 0.0
v2fun(xy) = 0.0

h2fun(xy) = exp(-((xy[1] - 0.5)^2 + (xy[2] - 0.5)^2) / 0.02) + 1.5

u0 = (xy, Bi) -> begin
    h2 = h2fun(xy)
    h1 = ε0 - (Bi + h2)  # ε = B + h2 + h1

    # (q,p) are x- and y-momenta
    q1 = h1 * u1fun(xy)
    p1 = h1 * v1fun(xy)
    q2 = h2 * u2fun(xy)
    p2 = h2 * v2fun(xy)

    @SVector [h1, q1, p1, h2, q2, p2]
end

# Build initial state on the SAME index set as x_all and B_all
initial = [u0(x_all[I], B_all[I]) for I in eachindex(x_all)]
SinFVM.set_current_state!(simulator, initial)

# ------------------------------------------------------------
# Interior arrays for plotting
# ------------------------------------------------------------
B_int = SinFVM.collect_topography_cells(equation.B, grid; interior=true)

function interior_fields(simulator, Bvals_int)
    st = SinFVM.current_interior_state(simulator)

    h1 = collect(st.h1); q1 = collect(st.q1); p1 = collect(st.p1)
    h2 = collect(st.h2); q2 = collect(st.q2); p2 = collect(st.p2)

    ω = Bvals_int .+ h2
    ε = ω .+ h1

    u1 = q1 ./ max.(h1, 1e-8)
    v1 = p1 ./ max.(h1, 1e-8)
    u2 = q2 ./ max.(h2, 1e-8)
    v2 = p2 ./ max.(h2, 1e-8)

    return (; h1,q1,p1,h2,q2,p2, ω, ε, u1,v1,u2,v2)
end

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
Tshow = 10

f = Figure(size=(1600, 900), fontsize=18)
Label(f[0, 1:2], "Two-layer SWE 2D — nx=$(nx), ny=$(ny), T=$(Tshow)", fontsize=22, padding=(0, 0, 10, 0))

ax_ω  = Axis(f[1, 1], title=L"\omega = B + h_2", xlabel=L"x", ylabel=L"y")
ax_ε  = Axis(f[1, 2], title=L"\varepsilon = B + h_2 + h_1", xlabel=L"x", ylabel=L"y")
ax_u1 = Axis(f[2, 1], title=L"u_1 = q_1/h_1", xlabel=L"x", ylabel=L"y")
ax_u2 = Axis(f[2, 2], title=L"u_2 = q_2/h_2", xlabel=L"x", ylabel=L"y")

fld0 = interior_fields(simulator, B_int)

hm_ω0 = heatmap!(ax_ω, fld0.ω);  Colorbar(f[1, 3], hm_ω0, label=L"\omega")
hm_ε0 = heatmap!(ax_ε, fld0.ε);  Colorbar(f[1, 4], hm_ε0, label=L"\varepsilon")
hm_u10 = heatmap!(ax_u1, fld0.u1); Colorbar(f[2, 3], hm_u10, label=L"u_1")
hm_u20 = heatmap!(ax_u2, fld0.u2); Colorbar(f[2, 4], hm_u20, label=L"u_2")

println("---- initial checks ----")
@show minimum(fld0.h1) minimum(fld0.h2)
@show maximum(abs.(fld0.u1)) maximum(abs.(fld0.u2))
@show minimum(fld0.ε) maximum(fld0.ε)

display(f)

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
@time SinFVM.simulate_to_time(simulator, Tshow)

# ------------------------------------------------------------
# Final update
# ------------------------------------------------------------
fld = interior_fields(simulator, B_int)

println("---- final checks ----")
@show minimum(fld.h1) minimum(fld.h2)
@show maximum(abs.(fld.u1)) maximum(abs.(fld.u2))
@show minimum(fld.ε) maximum(fld.ε)

hm_ω0[3][]  = fld.ω
hm_ε0[3][]  = fld.ε
hm_u10[3][] = fld.u1
hm_u20[3][] = fld.u2

f
