using CairoMakie
using StaticArrays
using SinFVM

backend = make_cpu_backend()
nx = 1024
grid = CartesianGrid(nx; gc=2)

# --- constant negative bottom so free surface ε=0 is possible with positive h1,h2 ---
B0 = -3.0
bottom = SinFVM.ConstantBottomTopography(B0)

# --- Setup the simulation ---
equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom)
reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())
numericalflux = CentralUpwind(equation)

# Source terms: bottom slope and non-conservative term
bottom_src = SinFVM.SourceTermBottom()
ncp_src    = SinFVM.SourceTermNonConservative()

conserved_system = ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
timestepper = RungeKutta2()
T = 0.05
simulator = Simulator(backend, conserved_system, timestepper, grid)

# --- Grid and bottom (cell-centered, interior) ---
x = SinFVM.cell_centers(grid)
Bvals = SinFVM.collect_topography_cells(equation.B, grid; interior=true)



# --- Initial conditions in Path A variables: (h1, q1, h2, q2) ---
ε0 = 0.0  # equilibrium free surface: ε = B + h2 + h1
h1fun(x) = 2.0
u1fun(x) = 0.0
h2fun(x) = exp(-(x - 0.5)^2 / 0.01) + 1.0
u2fun(x) = 0.0


u0 = (xi, Bi) -> begin
    h2 = h2fun(xi)
    h1 = ε0 - Bi - h2fun(xi)  
    q1 = h1 * u1fun(xi)
    q2 = h2 * u2fun(xi)
    @SVector[h1, q1, h2, q2]
end

initial = [u0(x[i], Bvals[i]) for i in eachindex(x)]
SinFVM.set_current_state!(simulator, initial)

# --- Visualization setup ---
f = Figure(size=(1600, 600), fontsize=24)
ax_surf = Axis(f[1, 1], title="Two-layer SWE 1D (surfaces). nx=$(nx), T=$(T)", ylabel="elevations", xlabel=L"x")
ax_vel  = Axis(f[1, 2], title="Two-layer SWE 1D (velocities). nx=$(nx), T=$(T)", ylabel="u", xlabel=L"x")

# --- Initial state ---
st0 = SinFVM.current_interior_state(simulator)
@show SinFVM.variable_names(typeof(st0))  # (:h1,:q1,:h2,:q2)

h1_0 = collect(st0.h1)
q1_0 = collect(st0.q1)
h2_0 = collect(st0.h2)
q2_0 = collect(st0.q2)

ω = Bvals .+ h2_0                 # interface elevation = B + h2

u1_0 = q1_0 ./ max.(h1_0, 1e-12)
u2_0 = q2_0 ./ max.(h2_0, 1e-12)

println("---- initial checks ----")
@show minimum(h1_0) minimum(h2_0)
@show maximum(abs.(u1_0)) maximum(abs.(u2_0))
@show minimum(ε_0) maximum(ε_0)   # should be ~0 and ~0 if equilibrium holds

# Plot initial surfaces
lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
lines!(ax_surf, x, ω, label=L"\omega(x,0)=B+h_2")
lines!(ax_surf, x, ε_0,  label=L"\varepsilon(x,0)=B+h_2+h_1")

# Plot initial velocities
lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)

# --- Run simulation ---
@time SinFVM.simulate_to_time(simulator, T)

# --- Final state ---
st = SinFVM.current_interior_state(simulator)
h1 = collect(st.h1)
q1 = collect(st.q1)
h2 = collect(st.h2)
q2 = collect(st.q2)

ω2 = Bvals .+ h2
ε  = ω2 .+ h1

u1 = q1 ./ max.(h1, 1e-12)
u2 = q2 ./ max.(h2, 1e-12)

println("---- final checks ----")
@show minimum(h1) minimum(h2)
@show maximum(abs.(u1)) maximum(abs.(u2))
@show minimum(ε) maximum(ε)

# Overlay final surfaces
lines!(ax_surf, x, ω2, linestyle=:dot, linewidth=5, label=L"\omega(x,t)")
lines!(ax_surf, x, ε,  linestyle=:dot, linewidth=5, label=L"\varepsilon(x,t)")

# Overlay final velocities
lines!(ax_vel, x, u1, linestyle=:dashdot, linewidth=5, label=L"u_1(x,t)")
lines!(ax_vel, x, u2, linestyle=:dashdot, linewidth=5, label=L"u_2(x,t)")

axislegend(ax_surf, position=:lt)
axislegend(ax_vel, position=:lt)
f
