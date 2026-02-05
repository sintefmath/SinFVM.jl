using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# Setup
# ============================================================

backend = make_cpu_backend()
nx = 1024
grid = CartesianGrid(nx; gc=2, boundary=SinFVM.PeriodicBC())

B0 = -3.0
bottom = SinFVM.ConstantBottomTopography(B0)

equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom)
numericalflux = CentralUpwind(equation)

# IMPORTANT: this assumes your TwoLayer limiter-reconstruction is the ω-well-balanced one:
#   input_conserved = (h1, q1, ω, q2)
#   outputs faces    = (h1, q1, h2, q2)
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = ConservedSystem(
    backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src]
)

timestepper = RungeKutta2()
T = 0.05
simulator = Simulator(backend, conserved_system, timestepper, grid)

# Grid + sampled bottom (interior)
x     = SinFVM.cell_centers(grid)
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)

# ============================================================
# Initial conditions (WELL-BALANCED storage): (h1, q1, ω, q2)
# ε = ω + h1,   ω = B + h2
# ============================================================

ε0 = 0.0
u1fun(x) = 0.0
u2fun(x) = 0.0
h2fun(x) = exp(-(x - 0.5)^2 / 0.01) + 1.0

u0 = (xi, Bi) -> begin
    h2 = h2fun(xi)
    ω  = Bi + h2            # store ω in slot 3
    h1 = ε0 - ω             # ε = ω + h1

    q1 = h1 * u1fun(xi)
    q2 = h2 * u2fun(xi)

    @SVector [h1, q1, ω, q2]  # (h1, q1, ω, q2)
end

initial = [u0(x[i], Bvals[i]) for i in eachindex(x)]
SinFVM.set_current_state!(simulator, initial)

# ============================================================
# Visualization setup
# ============================================================

f = Figure(size=(1600, 600), fontsize=24)
ax_surf = Axis(
    f[1, 1],
    title="Two-layer SWE 1D (surfaces). nx=$(nx), T=$(T)",
    ylabel="elevations",
    xlabel=L"x",
)
ax_vel  = Axis(
    f[1, 2],
    title="Two-layer SWE 1D (velocities). nx=$(nx), T=$(T)",
    ylabel="u",
    xlabel=L"x",
)

# ============================================================
# Initial state (slot 3 is ω, even though variable name says :h2)
# ============================================================

st0 = SinFVM.current_interior_state(simulator)
@show SinFVM.variable_names(typeof(st0))  # will print (:h1,:q1,:h2,:q2)

h1_0 = collect(st0.h1)
q1_0 = collect(st0.q1)
ω0   = collect(st0.h2)           # slot 3 == ω
q2_0 = collect(st0.q2)

h2_0 = ω0 .- Bvals               # recover physical h2
ε_0  = ω0 .+ h1_0                # ε = ω + h1

u1_0 = q1_0 ./ max.(h1_0, 1e-12)
u2_0 = q2_0 ./ max.(h2_0, 1e-12)

println("---- initial checks ----")
@show minimum(h1_0) minimum(h2_0)
@show maximum(abs.(u1_0)) maximum(abs.(u2_0))
@show minimum(ε_0) maximum(ε_0)

lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
lines!(ax_surf, x, ω0,                  label=L"\omega(x,0)=B+h_2")
lines!(ax_surf, x, ε_0,                 label=L"\varepsilon(x,0)=B+h_2+h_1")

lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

# ============================================================
# Run
# ============================================================

@time SinFVM.simulate_to_time(simulator, T)

# ============================================================
# Final state (slot 3 is ω)
# ============================================================

st = SinFVM.current_interior_state(simulator)

h1 = collect(st.h1)
q1 = collect(st.q1)
ω  = collect(st.h2)              # slot 3 == ω
q2 = collect(st.q2)

h2 = ω .- Bvals
ε  = ω .+ h1

u1 = q1 ./ max.(h1, 1e-12)
u2 = q2 ./ max.(h2, 1e-12)

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
