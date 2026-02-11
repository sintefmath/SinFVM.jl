using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Setup
# ============================================================

backend = make_cpu_backend()
nx = 128
grid = CartesianGrid(nx; gc=2, boundary=SinFVM.PeriodicBC())

xwrap(x) = x - floor(x)

# ------------------------------------------------------------
# Bottom topography (constant here)
# ------------------------------------------------------------
B0 = -2.0
bottom = SinFVM.ConstantBottomTopography(B0)

# Paper parameters for §2.7.2: g=10, r=0.98 => ρ1/ρ2=0.98
equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=0.98, ρ2=1.0, g=10.0)
numericalflux = CentralUpwind(equation)

# Reconstruction: equilibrium storage in state (h1, q1, w, q2)
# but NOTE: your reconstruct! overwrites face buffers to physical h2 in slot 3 (Choice A, SWE-style)
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
timestepper = RungeKutta2()
simulator = Simulator(backend, conserved_system, timestepper, grid; cfl=0.60)

# Interior grid + cell-centered bottom
x     = SinFVM.cell_centers(grid)
Bvals = SinFVM.collect_topography_cells(bottom, grid; interior=true)  # all = B0

# ============================================================
# Initial conditions for §2.7.2 (EQUILIBRIUM storage)
# Stored state is U=(h1,q1,w,q2) with w = h2 + B
# ============================================================

bump = 0.001
h1fun(x) = (0.1 < xwrap(x) < 0.2) ? (1.0 + bump) : 1.0
wfun(x)  = -1.0

u0 = (xi) -> begin
    h1 = h1fun(xi)
    w  = wfun(xi)
    @SVector [h1, 0.0, w, 0.0]   # (h1,q1,w,q2)
end

initial = [u0(x[i]) for i in eachindex(x)]
SinFVM.set_current_state!(simulator, initial)

# ============================================================
# Visualization setup
# ============================================================

Tshow = 0.15
f = Figure(size=(1600, 600), fontsize=24)

ax_surf = Axis(
    f[1, 1],
    title="Two-layer SWE 1D (surfaces). nx=$(nx), T=$(Tshow)",
    ylabel="elevations",
    xlabel=L"x",
)

ax_vel = Axis(
    f[1, 2],
    title="Two-layer SWE 1D (velocities). nx=$(nx), T=$(Tshow)",
    ylabel="u",
    xlabel=L"x",
)

# ============================================================
# Initial state (EQUILIBRIUM storage; compute h2 diagnostically)
# ============================================================

st0 = SinFVM.current_interior_state(simulator)
h1_0 = st0.h1; q1_0 = st0.q1; w0 = st0.w; q2_0 = st0.q2

h2_0 = w0 .- Bvals                 # diagnostic physical h2
ε_0  = h1_0 .+ w0                  # ε = h1 + w = h1 + h2 + B

u1_0 = q1_0 ./ max.(h1_0, equation.depth_cutoff)
u2_0 = q2_0 ./ max.(h2_0, equation.depth_cutoff)

println("---- initial checks ----")
@show minimum(h1_0) minimum(h2_0)
@show maximum(abs.(u1_0)) maximum(abs.(u2_0))
@show minimum(ε_0) maximum(ε_0)

lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
lines!(ax_surf, x, w0,               label=L"w(x,0)=B+h_2")
lines!(ax_surf, x, ε_0,              label=L"\varepsilon(x,0)=h_1+w")

lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

# ============================================================
# Run
# ============================================================

@time SinFVM.simulate_to_time(simulator, Tshow)

# ============================================================
# Final state (EQUILIBRIUM storage; compute h2 diagnostically)
# ============================================================

st = SinFVM.current_interior_state(simulator)
h1 = st.h1; q1 = st.q1; w = st.w; q2 = st.q2

h2 = w .- Bvals
ε  = h1 .+ w

u1 = q1 ./ max.(h1, equation.depth_cutoff)
u2 = q2 ./ max.(h2, equation.depth_cutoff)

println("---- final checks ----")
@show minimum(h1) minimum(h2)
@show maximum(abs.(u1)) maximum(abs.(u2))
@show minimum(ε) maximum(ε)

lines!(ax_surf, x, w, linestyle=:dot, linewidth=5, label=L"w(x,t)=B+h_2")
lines!(ax_surf, x, ε, linestyle=:dot, linewidth=5, label=L"\varepsilon(x,t)=h_1+w")

lines!(ax_vel, x, u1, linestyle=:dashdot, linewidth=5, label=L"u_1(x,t)")
lines!(ax_vel, x, u2, linestyle=:dashdot, linewidth=5, label=L"u_2(x,t)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

f
