# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# # Optimization of two-layer interface using AD
#
# In this example we use the two-layer shallow water equations in 1D to pose
# an inverse / optimal-control problem: given a dam-break scenario, find the
# initial interface elevation `w0` and upper-layer thickness `h10` (on the
# left, "wet" side of the dam) that minimise a compound cost function that
# penalises large upper-layer water volume at a downstream measurement cell
# while simultaneously encouraging a physically meaningful stratification.
#
# The state vector stored by the solver is
#
#   V = (h1, q1, w, q2)
#
# where h1 is the upper-layer depth, q1 = h1·u1 is the upper-layer discharge,
# w = h2 + B is the interface elevation (equilibrium-storage variable), and
# q2 = h2·u2 is the lower-layer discharge.  The physical lower-layer depth is
# recovered as  h2 = w - B.
#
# Gradients are computed automatically via ForwardDiff.  The optimisation is
# carried out with Optim.jl using a bounded L-BFGS method (Fminbox).

using SinFVM
using StaticArrays
using ForwardDiff
using Optim
using Parameters
using CairoMakie

# ---------------------------------------------------------------------------
# Callback: accumulates total upper-layer water volume at a single cell
# ---------------------------------------------------------------------------

"""
Mutable callback that accumulates the time-integrated upper-layer water volume

    ∫₀ᵀ  h1(t, xᵢ) · Δx  dt

at interior cell index `cell_index`.  A cutoff avoids contributions from
numerically dry states.  The type parameters allow ForwardDiff Dual numbers
to flow through the accumulator.
"""
@with_kw mutable struct TotalUpperLayerAtCell{IndexType, AccumType, AreaType, CutType}
    cell_index::IndexType
    total_h1::AccumType      = 0.0
    area_of_cell::AreaType   # = Δx in 1D
    cutoff::CutType          = 1e-6
end

function (cb::TotalUpperLayerAtCell)(time, simulator)
    h1_i = SinFVM.current_interior_state(simulator).h1[cb.cell_index]
    if h1_i > cb.cutoff
        dt = ForwardDiff.value(SinFVM.current_timestep(simulator))
        cb.total_h1 += dt * cb.area_of_cell * h1_i
        # Safety: bail out on NaN gradients
        if cb.total_h1 isa ForwardDiff.Dual && any(isnan.(cb.total_h1.partials))
            println("Aborting: NaN in ForwardDiff partials")
            exit()
        end
    end
end

# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

"""
Build a two-layer 1D dam-break simulator.

Parameters
- `w0`        : initial interface elevation (= h2 + B) on the LEFT (wet) side.
                On the right side we use a slightly deeper interface so
                the lower layer fills the nearly-dry right domain.
- `h10`       : upper-layer depth on the LEFT side of the dam.
- `backend`   : SinFVM backend (use `make_cpu_backend(ADType)` for AD).

Domain: [0, 100],  flat bottom B = 0,  WallBC.
Dam at x = 50:
  LEFT  ( x < 50 ): h1 = h10,  w = w0
  RIGHT ( x ≥ 50 ): h1 = ε,    w = ε  (near-dry thin lower layer only)
"""
function setup_twolayer_simulator(;backend = SinFVM.make_cpu_backend(),w0::T, h10::T,
    ) where {T}
    nx   = 64
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 100.0])
    x    = SinFVM.cell_centers(grid)

    # Flat bottom at B = 0
    B = SinFVM.ConstantBottomTopography(zero(T))

    # Density ratio: fresh water (ρ1) over salt water (ρ2)
    ρ1 = T(0.98)
    ρ2 = T(1.00)
    eq  = SinFVM.TwoLayerShallowWaterEquations1D(B; ρ1=ρ1, ρ2=ρ2, g=T(9.81),
                                                  depth_cutoff      = T(1e-4),
                                                  desingularizing_kappa = T(1e-4))
    rec  = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    flux = SinFVM.CentralUpwind(eq)
    bst  = SinFVM.SourceTermBottom()
    ncp  = SinFVM.SourceTermNonConservative()

    cs          = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [bst, ncp])
    timestepper = SinFVM.RungeKutta2()
    simulator   = SinFVM.Simulator(backend, cs, timestepper, grid; cfl=0.1)

    # --- Initial conditions (dam break) ---
    ε_h  = T(1e-4)   # thin dry layer for near-dry states
    x_dam = T(50.0)

    initial = map(x) do xi
        if xi < x_dam
            # wet side: lower layer fills from B=0 up to w0,  upper layer on top
            q1 = zero(T)
            q2 = zero(T)
            @SVector [h10, q1, w0, q2]
        else
            # dry side: thin near-dry state in both layers, w just above B=0
            h1_dry = ε_h
            w_dry  = ε_h          # h2_dry = w_dry - B = ε_h
            q1 = zero(T)
            q2 = zero(T)
            @SVector [h1_dry, q1, w_dry, q2]
        end
    end

    SinFVM.set_current_state!(simulator, initial)
    return simulator
end

# ---------------------------------------------------------------------------
# Forward model: run to time T and return total upper-layer volume at probe
# ---------------------------------------------------------------------------

"""
Run the two-layer dam-break model to time `T` for given interface elevation
`w0` and upper-layer thickness `h10`.  Returns the time-integrated upper-
layer volume at the probe cell (just downstream of the dam break).

This function is differentiable with ForwardDiff.
"""
function twolayer_dambreak_optim(; T, w0, h10)
    ADType  = promote_type(eltype(w0), eltype(h10))
    backend = SinFVM.make_cpu_backend(ADType)

    simulator = setup_twolayer_simulator(; backend, w0=ADType(w0), h10=ADType(h10))

    nx         = SinFVM.number_of_interior_cells(simulator.grid)
    # Probe just downstream of the dam (dam is at x=50, domain [0,100], nx=64 cells).
    # Cell nx÷2 + 4  ≈  x = 56 m  — always reached quickly so the gradient is nonzero.
    cell_index = nx÷2 + 4
    Δx         = SinFVM.compute_dx(simulator.grid)

    callback = TotalUpperLayerAtCell(
        cell_index   = cell_index,
        area_of_cell = Δx,
        total_h1     = ADType(0.0),
    )

    SinFVM.simulate_to_time(simulator, T; callback=callback)
    return callback.total_h1
end

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

"""
Compound cost function for the optimiser.

Physical scenario
-----------------
We are designing the initial stratification (interface depth w0, upper-layer
thickness h10) for a two-layer estuary release event.  The upper layer carries
a tracer (e.g. fresh water or sediment) that we want to observe at a sensor
located just downstream of the release point.

Competing objectives:
- `match_cost`   : We want exactly `h1_target` units of cumulative upper-layer
                   volume to arrive at the sensor.  Both too little and too much
                   are penalised quadratically.  This term links the cost to the
                   *simulation output* and provides the AD gradient.
- `release_cost` : We prefer a thin initial upper layer (h10 small) — fewer
                   resources / less release mass.  This competes with match_cost
                   because a smaller h10 reduces the upper-layer signal at the
                   sensor.

There is no trivial analytical minimum: to match the target with minimal h10
the optimizer must find the w0 that makes the two-layer wave dynamics most
efficient at transporting upper-layer mass to the sensor.
"""
const H1_TARGET = 8.0   # desired cumulative upper-layer volume at probe [m²·s]

function cost_function(params)
    w0, h10 = params

    if params isa Vector{Float64}
        println("Optimisation step:  w0 = $(round(w0; digits=4)),  " *
                "h10 = $(round(h10; digits=4))")
    end

    # Forward simulation — provides gradient through h1(t, x_probe)
    total_h1 = twolayer_dambreak_optim(; T=20.0, w0=w0, h10=h10)

    # Cost components (competing objectives)
    match_cost   = 1000.0 * (total_h1 - H1_TARGET)^2   # match sensor target
    release_cost =  400.0 * h10^2                        # minimise initial release

    total_cost = match_cost + release_cost

    if params isa Vector{Float64}
        println("  total_h1=$(round(ForwardDiff.value(total_h1); digits=4))  " *
                "match=$(round(ForwardDiff.value(match_cost); digits=2))  " *
                "release=$(round(ForwardDiff.value(release_cost); digits=2))")
    end

    if total_cost isa ForwardDiff.Dual
        @assert all(.!isnan.(total_cost.partials)) "NaN in cost partials"
    end

    return total_cost
end

# Gradient via ForwardDiff
function grad!(storage, params)
    ForwardDiff.gradient!(storage, cost_function, params)
end

# ---------------------------------------------------------------------------
# Optimisation setup
# ---------------------------------------------------------------------------

# Parameter bounds
#   w0  ∈ [0.5, 3.0]   (lower-layer depth; must stay above flat bottom B=0)
#   h10 ∈ [0.1, 2.5]   (upper-layer thickness; must be positive)
lower_bound = [0.5, 0.1]
upper_bound = [3.0, 2.5]

# Initial guess — deliberately away from the expected optimum so convergence is visible.
# With h10=2.0 the release_cost is large; w0=1.0 gives a shallow lower layer.
initial_guess = [1.0, 2.0]   # [w0, h10]

# Optimiser  (bounded L-BFGS via Fminbox)
# - iterations:       max outer Fminbox iterations (each involves several simulations)
# - outer_iterations: hard cap on the Fminbox barrier-method outer loops
# - show_trace:       print cost + gradient norm each outer step so you can see progress
# - g_tol / f_tol:   stop early once gradient or cost change is small enough
opts = Optim.Options(
    store_trace      = true,
    show_trace       = true,
    iterations       = 30,        # ≈ 30 outer steps before giving up
    outer_iterations = 30,
    g_tol            = 1e-3,
    f_tol            = 1e-4,
)
result = optimize(
    cost_function, grad!,
    lower_bound, upper_bound,
    initial_guess,
    Fminbox(LBFGS(; m=5)),
    opts,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

optimal = result.minimizer
optimal_h1 = twolayer_dambreak_optim(; T=20.0, w0=optimal[1], h10=optimal[2])
println("\n=== Optimisation complete ===")
println("  Optimal w0  = $(round(optimal[1]; digits=4))  (lower-layer depth)")
println("  Optimal h10 = $(round(optimal[2]; digits=4))  (upper-layer thickness)")
println("  Simulated total_h1 at probe = $(round(optimal_h1; digits=4))  (target = $H1_TARGET)")
println("  Minimum cost = $(round(result.minimum; digits=4))")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# --- Convergence history ---
cost_vals = let tr = Optim.trace(result)
    outer = map(os -> os.iteration, tr) .== 0
    map(os -> os.value, tr)[outer]
end

fig = Figure(size=(1400, 900), fontsize=22)

ax_conv = Axis(fig[1, 1:2],
    title  = "Optimisation convergence (two-layer 1D interface)",
    xlabel = "outer iteration",
    ylabel = "cost value",
)
lines!(ax_conv,  1:length(cost_vals), cost_vals, linewidth=2)
scatter!(ax_conv, 1:length(cost_vals), cost_vals, markersize=8)

# --- Compare initial and optimal flow fields at T=20 ---
function snapshot(w0, h10; label)
    sim = setup_twolayer_simulator(;
        backend = SinFVM.make_cpu_backend(),
        w0      = Float64(w0),
        h10     = Float64(h10),
    )
    grid = sim.grid
    x    = collect(SinFVM.cell_centers(grid))

    # store the initial state for plotting
    st0  = SinFVM.current_interior_state(sim)
    h1_0 = collect(st0.h1)
    w_0  = collect(st0.w)
    η_0  = h1_0 .+ w_0

    SinFVM.simulate_to_time(sim, 20.0)
    st   = SinFVM.current_interior_state(sim)
    h1   = collect(st.h1)
    w    = collect(st.w)
    η    = h1 .+ w
    q1   = collect(st.q1)
    q2   = collect(st.q2)
    h2   = collect(st.w)           # B = 0, so h2 = w
    u1   = q1 ./ max.(h1, 1e-8)
    u2   = q2 ./ max.(h2, 1e-8)

    return (; x, h1_0, w_0, η_0, h1, w, η, u1, u2, label)
end

snap_init = snapshot(initial_guess[1], initial_guess[2]; label="initial guess")
snap_opt  = snapshot(optimal[1],       optimal[2];       label="optimal")

for (row, sn) in enumerate([snap_init, snap_opt])
    ax_surf = Axis(fig[row+1, 1],
        title  = "$(sn.label): surface elevations at T=20",
        xlabel = "x",
        ylabel = "elevation",
    )
    ax_vel = Axis(fig[row+1, 2],
        title  = "$(sn.label): layer velocities at T=20",
        xlabel = "x",
        ylabel = "u",
    )

    lines!(ax_surf, sn.x, zeros(length(sn.x)), linestyle=:dash,
           color=:black, label="B(x) = 0")
    lines!(ax_surf, sn.x, sn.w,  linewidth=2, label=L"w = h_2+B  \;(interface)")
    lines!(ax_surf, sn.x, sn.η,  linewidth=2, label=L"\eta = h_1+w  \;(surface)")
    axislegend(ax_surf, position=:lt)

    lines!(ax_vel, sn.x, sn.u1, linewidth=2, label=L"u_1  \;(upper)")
    lines!(ax_vel, sn.x, sn.u2, linewidth=2, label=L"u_2  \;(lower)")
    axislegend(ax_vel, position=:lt)
end

display(fig)
