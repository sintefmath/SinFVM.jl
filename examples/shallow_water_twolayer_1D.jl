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
# Bottom topography on faces (intersections), including ghosts
# Eq. (2.35):
# B(x) = 0.25[cos(10π(x-0.5)) + 1] - 2,   if 0.4<x<0.6
#      = -2,                               otherwise
# ------------------------------------------------------------
"""
xF = SinFVM.cell_faces(grid; interior=false)
Bint = similar(xF)
@inbounds for i in eachindex(xF)
    x = xwrap(xF[i])
    Bint[i] = (0.4 < x < 0.6) ? (0.25*(cos(10π*(x - 0.5)) + 1.0) - 2.0) : -2.0
end
bottom = SinFVM.BottomTopography1D(Bint, backend, grid)
"""

B0 = -2.0
bottom = SinFVM.ConstantBottomTopography(B0)
equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=0.98, ρ2=1.0, g=10.0)
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)  # all = B0

# Paper parameters for §2.7.2: g=10, r=0.98 => ρ1/ρ2=0.98
equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=0.98, ρ2=1.0, g=10.0)
numericalflux = CentralUpwind(equation)

# Reconstruction: STORE physical (h1,q1,h2,q2) but reconstruct using w=h2+B internally
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

@show bottom_src
conserved_system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
timestepper = RungeKutta2()
simulator = Simulator(backend, conserved_system, timestepper, grid; cfl=0.60)

# Interior grid + cell-centered bottom
x     = SinFVM.cell_centers(grid)
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)

# ============================================================
# Initial conditions for §2.7.2 (PHYSICAL storage)
# Paper gives w(x,0) = h2 + B = -1, q1=q2=0, and h1 has a small bump
# Stored state must be U=(h1,q1,h2,q2) with h2 = w - B_cell
# ============================================================

bump = 1e-5
h1fun(x) = (0.1 < xwrap(x) < 0.2) ? (1.0 + bump) : 1.0
wfun(x)  = -1.0

u0 = (xi, Bi) -> begin
    h1 = h1fun(xi)
    w  = wfun(xi)
    h2 = w - Bi                 # physical h2 from w=h2+B
    @SVector [h1, 0.0, h2, 0.0] # (h1,q1,h2,q2) PHYSICAL STORAGE
end

initial = [u0(x[i], Bvals[i]) for i in eachindex(x)]
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
# Initial state (PHYSICAL storage; compute w, ε diagnostically)
# ============================================================

st0 = SinFVM.current_interior_state(simulator)

# Be robust to naming differences: read as vectors of SVectors
U0 = collect(st0)  # Vector{SVector{4}}
h1_0 = st0.h1
q1_0 = st0.q1
h2_0 = st0.h2     # PHYSICAL h2
q2_0 = st0.q2

w0  = h2_0 .+ Bvals         # w = h2 + B
ε_0 = h1_0 .+ w0            # ε = h1 + h2 + B

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
# Final state (PHYSICAL storage; compute w, ε diagnostically)
# ============================================================

st = SinFVM.current_interior_state(simulator)
U = collect(st)

h1 = st.h1
q1 = st.q1
h2 = st.h2     # PHYSICAL h2
q2 = st.q2

w  = h2 .+ Bvals            # w = h2 + B
ε  = h1 .+ w                # ε = h1 + h2 + B

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

