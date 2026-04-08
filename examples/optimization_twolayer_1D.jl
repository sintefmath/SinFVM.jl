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

# ---------------------------------------------------------------------------
# Twin experiment: identify unknown interface parameter w0 from observable data
#
# Observable data used in cost:
#   eta = h1 + w           (free surface)
#   u1  = q1 / h1          (upper-layer velocity)
#   u2  = q2 / (w - B(x))  (lower-layer velocity)
#
# Control variable:
#   w0 = initial interface elevation on left side of dam
#
# Everything else is fixed, in particular h10.
#
# We first generate synthetic "exact" data from a known W0_TRUE, and then
# optimize from a wrong initial guess to recover w0.
# ---------------------------------------------------------------------------

using SinFVM
using StaticArrays
using ForwardDiff
using Optim
using Parameters
using CairoMakie

# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

function setup_twolayer_simulator(; backend = SinFVM.make_cpu_backend(), w0::T, h10::T) where {T}
    nx   = 64
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 100.0])
    x    = SinFVM.cell_centers(grid)

    # Bathymetry; here flat, but the code below is written so you can later
    # replace this with a non-flat B(x) if desired.
    B = SinFVM.ConstantBottomTopography(zero(T))

    ρ1 = T(0.98)
    ρ2 = T(1.00)

    eq = SinFVM.TwoLayerShallowWaterEquations1D(
        B;
        ρ1 = ρ1,
        ρ2 = ρ2,
        g = T(9.81),
        depth_cutoff = T(1e-4),
        desingularizing_kappa = T(1e-4),
    )

    rec  = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    flux = SinFVM.CentralUpwind(eq)
    bst  = SinFVM.SourceTermBottom()
    ncp  = SinFVM.SourceTermNonConservative()

    cs          = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [bst, ncp])
    timestepper = SinFVM.RungeKutta2()
    simulator   = SinFVM.Simulator(backend, cs, timestepper, grid; cfl=0.1)

    ε_h   = T(1e-4)
    x_dam = T(50.0)

    initial = map(x) do xi
        if xi < x_dam
            # left / wet side
            @SVector [h10, zero(T), w0, zero(T)]
        else
            # right / nearly dry side
            @SVector [ε_h, zero(T), ε_h, zero(T)]
        end
    end

    SinFVM.set_current_state!(simulator, initial)
    return simulator
end

# ---------------------------------------------------------------------------
# Observable extraction
# ---------------------------------------------------------------------------

"""
Evaluate bathymetry B(x) at cell centers.

For the current flat-bottom setup B(x)=0. This is written as a separate
function so you can later replace it by the actual bathymetry profile.
"""
function bathymetry_values(simulator, ::Type{T}) where {T}
    x = SinFVM.cell_centers(simulator.grid)
    return fill(zero(T), length(x))
end

"""
Return observable fields from the simulator interior state:

  eta = h1 + w
  u1  = q1 / h1
  u2  = q2 / h2,  h2 = w - B(x)

A small epsilon protects against division by very small depths.
"""
function observable_fields(simulator; vel_eps = nothing)
    st = SinFVM.current_interior_state(simulator)
    Tstate = eltype(st.h1)

    epsT = vel_eps === nothing ? Tstate(1e-6) : Tstate(vel_eps)

    Bvals = bathymetry_values(simulator, Tstate)

    h1 = st.h1
    q1 = st.q1
    w  = st.w
    q2 = st.q2

    h2  = w .- Bvals
    eta = h1 .+ w
    u1  = q1 ./ max.(h1, epsT)
    u2  = q2 ./ max.(h2, epsT)

    return (; eta, u1, u2)
end

# ---------------------------------------------------------------------------
# Observation callback
# ---------------------------------------------------------------------------

"""
Record observable data [eta, u1, u2] at selected times and cells.

Stored order:
  [eta(t1,x1), u1(t1,x1), u2(t1,x1), eta(t1,x2), u1(t1,x2), u2(t1,x2), ...]
"""
@with_kw mutable struct ObservableRecorder{VT, IT, OT}
    obs_times::VT
    cell_indices::IT
    next_obs::Int = 1
    data::Vector{OT} = OT[]
end

function (cb::ObservableRecorder)(time, simulator)
    t = ForwardDiff.value(time)

    while cb.next_obs <= length(cb.obs_times) && t + 1e-12 >= cb.obs_times[cb.next_obs]
        obs = observable_fields(simulator)

        for i in cb.cell_indices
            push!(cb.data, obs.eta[i])
            push!(cb.data, obs.u1[i])
            push!(cb.data, obs.u2[i])
        end

        cb.next_obs += 1
    end
end

# ---------------------------------------------------------------------------
# Forward solve returning observable data vector
# ---------------------------------------------------------------------------

function simulate_observations(; T, w0, h10, obs_times, cell_indices)
    ADType  = promote_type(typeof(w0), typeof(h10))
    backend = SinFVM.make_cpu_backend(ADType)

    sim = setup_twolayer_simulator(; backend, w0 = ADType(w0), h10 = ADType(h10))

    recorder = ObservableRecorder(
        obs_times    = obs_times,
        cell_indices = cell_indices,
        data         = ADType[],
    )

    SinFVM.simulate_to_time(sim, T; callback = recorder)

    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"

    return recorder.data
end

# ---------------------------------------------------------------------------
# Twin experiment setup
# ---------------------------------------------------------------------------

const T_END = 20.0

# Fixed, known upper-layer thickness
const H10_FIXED = 0.75

# True interface parameter used to create synthetic exact observations
const W0_TRUE = 1.85

# Pick several cells downstream of the dam
const NX = 64
const CELL_INDICES = collect((NX ÷ 2) .+ [2, 4, 6, 8, 10])

# Pick several observation times
const OBS_TIMES = collect(range(4.0, T_END; length = 12))

# Synthetic exact data
const EXACT_OBS = simulate_observations(
    T            = T_END,
    w0           = W0_TRUE,
    h10          = H10_FIXED,
    obs_times    = OBS_TIMES,
    cell_indices = CELL_INDICES,
)

println("Generated synthetic exact observations:")
println("  W0_TRUE       = $W0_TRUE")
println("  H10_FIXED     = $H10_FIXED")
println("  n_observables = $(length(EXACT_OBS))")

# ---------------------------------------------------------------------------
# Cost function
# ---------------------------------------------------------------------------

# Weights for [eta, u1, u2] contributions
const W_ETA = 1.0
const W_U1  = 1.0
const W_U2  = 1.0

"""
Least-squares observable misfit.

The optimizer only sees w0 as unknown. Data come from the measurable fields:
  eta, u1, u2
at several times/cells.
"""
function cost_function(wvec)
    w0 = wvec[1]

    pred = simulate_observations(
        T            = T_END,
        w0           = w0,
        h10          = H10_FIXED,
        obs_times    = OBS_TIMES,
        cell_indices = CELL_INDICES,
    )

    # Data ordering is [eta, u1, u2, eta, u1, u2, ...]
    J = zero(eltype(pred))

    @inbounds for k in 1:3:length(pred)
        dη  = pred[k]   - EXACT_OBS[k]
        du1 = pred[k+1] - EXACT_OBS[k+1]
        du2 = pred[k+2] - EXACT_OBS[k+2]

        J += 0.5 * (W_ETA * dη^2 + W_U1 * du1^2 + W_U2 * du2^2)
    end

    if wvec isa Vector{Float64}
        println("w0 = $(round(w0; digits=8)),  J = $(round(ForwardDiff.value(J); digits=12))")
    end

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

function grad!(storage, wvec)
    ForwardDiff.gradient!(storage, cost_function, wvec)
end

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

lower_bound   = [0.5]
upper_bound   = [3.0]
initial_guess = [1.0]   # deliberately wrong

opts = Optim.Options(
    store_trace      = true,
    show_trace       = true,
    iterations       = 50,
    outer_iterations = 50,
    g_tol            = 1e-10,
    f_tol            = 1e-12,
)

result = optimize(
    cost_function, grad!,
    lower_bound, upper_bound,
    initial_guess,
    Fminbox(LBFGS(; m = 5)),
    opts,
)

w_opt = result.minimizer[1]
final_cost = result.minimum

println("\n=== Optimization complete ===")
println("True w0      = $(round(W0_TRUE; digits=10))")
println("Recovered w0 = $(round(w_opt; digits=10))")
println("Absolute error = $(round(abs(w_opt - W0_TRUE); digits=12))")
println("Final cost     = $(round(final_cost; digits=16))")

# ---------------------------------------------------------------------------
# Diagnostic snapshots for plotting
# ---------------------------------------------------------------------------

function snapshot(w0; h10 = H10_FIXED, label = "")
    sim = setup_twolayer_simulator(
        backend = SinFVM.make_cpu_backend(),
        w0      = Float64(w0),
        h10     = Float64(h10),
    )

    x = collect(SinFVM.cell_centers(sim.grid))

    SinFVM.simulate_to_time(sim, T_END)

    st  = SinFVM.current_interior_state(sim)
    obs = observable_fields(sim)

    h1 = collect(st.h1)
    q1 = collect(st.q1)
    w  = collect(st.w)
    q2 = collect(st.q2)
    η  = collect(obs.eta)
    u1 = collect(obs.u1)
    u2 = collect(obs.u2)

    return (; x, h1, q1, w, q2, η, u1, u2, label)
end

snap_true = snapshot(W0_TRUE; label = "truth")
snap_init = snapshot(initial_guess[1]; label = "initial guess")
snap_opt  = snapshot(w_opt; label = "optimized")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

cost_vals = let tr = Optim.trace(result)
    outer = map(os -> os.iteration, tr) .== 0
    map(os -> os.value, tr)[outer]
end

fig = Figure(size = (1500, 1000), fontsize = 22)

ax_conv = Axis(
    fig[1, 1:2],
    title  = "Twin-experiment convergence: recovering unknown interface parameter w0",
    xlabel = "outer iteration",
    ylabel = "cost",
)
lines!(ax_conv, 1:length(cost_vals), cost_vals, linewidth = 2)
scatter!(ax_conv, 1:length(cost_vals), cost_vals, markersize = 8)

for (row, sn) in enumerate([snap_true, snap_init, snap_opt])
    ax_eta = Axis(
        fig[row+1, 1],
        title  = "$(sn.label): free surface and interface at T=$(T_END)",
        xlabel = "x",
        ylabel = "elevation",
    )
    ax_u = Axis(
        fig[row+1, 2],
        title  = "$(sn.label): measurable velocities at T=$(T_END)",
        xlabel = "x",
        ylabel = "velocity",
    )

    lines!(ax_eta, sn.x, sn.w, linewidth = 2, label = "w (interface)")
    lines!(ax_eta, sn.x, sn.η, linewidth = 2, label = "η (free surface)")
    axislegend(ax_eta, position = :lt)

    lines!(ax_u, sn.x, sn.u1, linewidth = 2, label = "u1")
    lines!(ax_u, sn.x, sn.u2, linewidth = 2, label = "u2")
    axislegend(ax_u, position = :lt)
end

display(fig)