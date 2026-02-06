using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Setup
# ============================================================

backend = make_cpu_backend()
nx = 128
grid = CartesianGrid(nx; gc=2, boundary=SinFVM.PeriodicBC())

B0 = -3.0
bottom = SinFVM.ConstantBottomTopography(B0)

equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1 = 1.00, ρ2 = 1.02, g = 9.81)
numericalflux = CentralUpwind(equation)

# Reconstruction:
#   input_conserved  = (h1, q1, h2, q2)  [PHYSICAL STORAGE]
#   internally uses ω = h2 + B for limiting and reconstructing,
#   outputs faces    = (h1, q1, h2, q2)
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
timestepper = RungeKutta2()
simulator = Simulator(backend, conserved_system, timestepper, grid; cfl = 0.99)

# Grid + sampled bottom (interior)
x = SinFVM.cell_centers(grid)
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)

# ============================================================
# Initial conditions (PHYSICAL conserved variables): (h1, q1, h2, q2)
# ε = B + h2 + h1
# ============================================================

ε0 = 0.0
u1fun(x) = 0.0
u2fun(x) = 0.0
h2fun(x) = exp(-(x - 0.5)^2 / 0.05) + 1.5

u0 = (xi, Bi) -> begin
    h2 = h2fun(xi)
    h1 = ε0 - (Bi + h2)      # ε = B + h2 + h1

    q1 = h1 * u1fun(xi)
    q2 = h2 * u2fun(xi)

    @SVector [h1, q1, h2, q2]   # PHYSICAL STORAGE
end

initial = [u0(x[i], Bvals[i]) for i in eachindex(x)]
SinFVM.set_current_state!(simulator, initial)

# ============================================================
# Visualization setup
# ============================================================

Tshow = 10000.0
f = Figure(size=(1600, 600), fontsize=24)

ax_surf = Axis(
    f[1, 1],
    title="Two-layer SWE 1D (surfaces). nx=$(nx), T=$(Tshow)",
    ylabel="elevations",
    xlabel=L"x",
)
ax_vel  = Axis(
    f[1, 2],
    title="Two-layer SWE 1D (velocities). nx=$(nx), T=$(Tshow)",
    ylabel="u",
    xlabel=L"x",
)

# ============================================================
# Initial state (now slot 3 is truly h2)
# ============================================================

st0 = SinFVM.current_interior_state(simulator)
@show SinFVM.variable_names(typeof(st0))  # (:h1,:q1,:h2,:q2)

h1_0 = collect(st0.h1)
q1_0 = collect(st0.q1)
h2_0 = collect(st0.h2)   # PHYSICAL h2
q2_0 = collect(st0.q2)

ω0  = Bvals .+ h2_0
ε_0 = ω0 .+ h1_0

u1_0 = q1_0 ./ max.(h1_0, 1e-5)
u2_0 = q2_0 ./ max.(h2_0, 1e-5)

println("---- initial checks ----")
@show minimum(h1_0) minimum(h2_0)
@show maximum(abs.(u1_0)) maximum(abs.(u2_0))
@show minimum(ε_0) maximum(ε_0)

lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
lines!(ax_surf, x, ω0,                label=L"\omega(x,0)=B+h_2")
lines!(ax_surf, x, ε_0,               label=L"\varepsilon(x,0)=B+h_2+h_1")

lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

# ============================================================
# Run
# ============================================================

T = 10000.0
@time SinFVM.simulate_to_time(simulator, T)

# ============================================================
# Final state (slot 3 is h2)
# ============================================================

st = SinFVM.current_interior_state(simulator)

h1 = collect(st.h1)
q1 = collect(st.q1)
h2 = collect(st.h2)      # PHYSICAL h2
q2 = collect(st.q2)

ω = Bvals .+ h2
ε = ω .+ h1

u1 = q1 ./ max.(h1, 1e-5)
u2 = q2 ./ max.(h2, 1e-5)

println("---- final checks ----")
@show minimum(h1) minimum(h2)
@show maximum(abs.(u1)) maximum(abs.(u2))
@show minimum(ε) maximum(ε)

lines!(ax_surf, x, ω, linestyle=:dot, linewidth=5, label=L"\omega(x,t)")
lines!(ax_surf, x, ε, linestyle=:dot, linewidth=5, label=L"\varepsilon(x,t)")

lines!(ax_vel, x, u1, linestyle=:dashdot, linewidth=5, label=L"u_1(x,t)")
lines!(ax_vel, x, u2, linestyle=:dashdot, linewidth=5, label=L"u_2(x,t)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

f
