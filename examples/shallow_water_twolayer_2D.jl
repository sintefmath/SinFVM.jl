using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Two-layer SWE 2D runner (PHYSICAL STORAGE)
#   U = (h1, q1, p1, h2, q2, p2)
# Reconstruction internally uses equilibrium variable
#   ω = h2 + B
# but returns face states in PHYSICAL variables.
# ============================================================

# ----------------------------
# Setup
# ----------------------------
backend = make_cpu_backend()
nx, ny = 32, 32
grid = CartesianGrid(nx, ny; gc=2, boundary=SinFVM.PeriodicBC())

B0 = -3.0
bottom = SinFVM.ConstantBottomTopography(B0)

equation = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=1.00, ρ2=1.02, g=9.81)
numericalflux = CentralUpwind(equation)

# Reconstruction: PHYSICAL storage in state:
#   input_conserved = (h1,q1,p1,h2,q2,p2)
# Internally uses ω = h2 + B (your 2D reconstruct! overload)
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = ConservedSystem(
    backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src]
)

timestepper = RungeKutta2()
simulator = Simulator(backend, conserved_system, timestepper, grid; cfl=0.99)

# ----------------------------
# Grid + bottom (interior)
# ----------------------------
x = SinFVM.cell_centers(grid)  # returns array of points/tuples
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)  # (nx,ny)

# ----------------------------
# Initial conditions (PHYSICAL conserved vars)
#   U = (h1, q1, p1, h2, q2, p2)
#   ε = B + h2 + h1
# ----------------------------
ε0 = 0.0

u1fun(xy) = 0.0
v1fun(xy) = 0.0
u2fun(xy) = 0.0
v2fun(xy) = 0.0

# A smooth h2 bump centered in the domain (works with periodic BC)
h2fun(xy) = exp(-((xy[1] - 0.5)^2 + (xy[2] - 0.5)^2) / 0.02) + 1.5

u0 = (xy, Bi) -> begin
    h2 = h2fun(xy)
    h1 = ε0 - (Bi + h2)     # ε = B + h2 + h1

    q1 = h1 * u1fun(xy)     # x-momentum layer 1
    p1 = h1 * v1fun(xy)     # y-momentum layer 1
    q2 = h2 * u2fun(xy)     # x-momentum layer 2
    p2 = h2 * v2fun(xy)     # y-momentum layer 2

    @SVector [h1, q1, p1, h2, q2, p2]
end

initial = [u0(x[I], Bvals[I]) for I in eachindex(x)]
SinFVM.set_current_state!(simulator, initial)

# ----------------------------
# Visualization
# ----------------------------
Tshow = 0.10

f = Figure(size=(1600, 900), fontsize=18)

Label(
    f[0, 1:2],
    "Two-layer SWE 2D — nx=$(nx), ny=$(ny), T=$(Tshow)",
    fontsize=22,
    padding=(0, 0, 10, 0)
)

ax_ω  = Axis(f[1, 1], title=L"\omega = B + h_2", xlabel=L"x", ylabel=L"y")
ax_ε  = Axis(f[1, 2], title=L"\varepsilon = B + h_2 + h_1", xlabel=L"x", ylabel=L"y")
ax_u1 = Axis(f[2, 1], title=L"u_1 = q_1/h_1", xlabel=L"x", ylabel=L"y")
ax_u2 = Axis(f[2, 2], title=L"u_2 = q_2/h_2", xlabel=L"x", ylabel=L"y")

# ----------------------------
# Helper for plotting interior fields
# ----------------------------
function interior_fields(simulator, Bvals)
    st = SinFVM.current_interior_state(simulator)

    h1 = collect(st.h1)
    q1 = collect(st.q1)
    p1 = collect(st.p1)
    h2 = collect(st.h2)
    q2 = collect(st.q2)
    p2 = collect(st.p2)

    ω = Bvals .+ h2
    ε = ω .+ h1

    u1 = q1 ./ max.(h1, 1e-8)
    v1 = p1 ./ max.(h1, 1e-8)
    u2 = q2 ./ max.(h2, 1e-8)
    v2 = p2 ./ max.(h2, 1e-8)

    return (; h1,q1,p1,h2,q2,p2, ω, ε, u1,v1,u2,v2)
end

# ----------------------------
# Initial plots
# ----------------------------
fld0 = interior_fields(simulator, Bvals)

hm_ω0 = heatmap!(ax_ω, fld0.ω)
Colorbar(f[1, 3], hm_ω0, label=L"\omega")

hm_ε0 = heatmap!(ax_ε, fld0.ε)
Colorbar(f[1, 4], hm_ε0, label=L"\varepsilon")

hm_u10 = heatmap!(ax_u1, fld0.u1)
Colorbar(f[2, 3], hm_u10, label=L"u_1")

hm_u20 = heatmap!(ax_u2, fld0.u2)
Colorbar(f[2, 4], hm_u20, label=L"u_2")

println("---- initial checks ----")
@show minimum(fld0.h1) minimum(fld0.h2)
@show maximum(abs.(fld0.u1)) maximum(abs.(fld0.u2))
@show minimum(fld0.ε) maximum(fld0.ε)

display(f)

# ----------------------------
# Run
# ----------------------------
T = Tshow
@time SinFVM.simulate_to_time(simulator, T)

# ----------------------------
# Final fields + update plots
# ----------------------------
fld = interior_fields(simulator, Bvals)

println("---- final checks ----")
@show minimum(fld.h1) minimum(fld.h2)
@show maximum(abs.(fld.u1)) maximum(abs.(fld.u2))
@show minimum(fld.ε) maximum(fld.ε)

# Update plots (replace heatmap data)
hm_ω0[3][]  = fld.ω
hm_ε0[3][]  = fld.ε
hm_u10[3][] = fld.u1
hm_u20[3][] = fld.u2

f
