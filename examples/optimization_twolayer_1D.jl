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

using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie

# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

const DESING_KAPPA = 1e-4

function setup_twolayer_simulator(; backend=SinFVM.make_cpu_backend(), w0::T, h10::T) where {T}
    nx = 64
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 100.0])
    x = SinFVM.cell_centers(grid)
    B = SinFVM.ConstantBottomTopography(zero(T))
    eq = SinFVM.TwoLayerShallowWaterEquations1D(B; ρ1=T(0.98), ρ2=T(1.00), g=T(9.81), depth_cutoff=T(1e-4), desingularizing_kappa=T(DESING_KAPPA))
    rec, flux = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1)), SinFVM.PathConservativeCentralUpwind(eq)
    cs = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.1)
    ε_h, x_dam = T(1e-4), T(50.0)
    initial = map(x) do xi
        xi < x_dam ? @SVector([h10, zero(T), w0, zero(T)]) : @SVector([ε_h, zero(T), ε_h, zero(T)])
    end
    SinFVM.set_current_state!(sim, initial)
    sim
end

# ---------------------------------------------------------------------------
# Observables and reconstruction
# ---------------------------------------------------------------------------

bathymetry_values(simulator, ::Type{T}) where {T} = fill(zero(T), length(SinFVM.cell_centers(simulator.grid)))

function desingularize(h, κ)
    copysign(one(h), h) * max(abs(h), min(h^2 / (2*κ) + κ / 2, κ))
end

desingularize(h, momentum, κ) = momentum / desingularize(h, κ)

function observable_fields(simulator)
    st = SinFVM.current_interior_state(simulator)
    Tstate, κ = eltype(st.h1), eltype(st.h1)(DESING_KAPPA)
    Bvals = bathymetry_values(simulator, Tstate)
    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2, eta = w .- Bvals, h1 .+ w
    u1, u2 = desingularize.(h1, q1, κ), desingularize.(h2, q2, κ)
    (; eta, u1, u2, w, Bvals)
end

reconstruct_from_observables(eta, u1, u2, w, B) = (; h1 = eta .- w, h2 = w .- B, q1 = (eta .- w) .* u1, q2 = (w .- B) .* u2)

# ---------------------------------------------------------------------------
# Observation callback
# ---------------------------------------------------------------------------

@with_kw mutable struct ObservableRecorder{VT,IT,OT}
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
            push!(cb.data, obs.eta[i]); push!(cb.data, obs.u1[i]); push!(cb.data, obs.u2[i])
        end
        cb.next_obs += 1
    end
end

# ---------------------------------------------------------------------------
# Forward solve returning observable data vector
# ---------------------------------------------------------------------------

function simulate_observations(; T, w0, h10, obs_times, cell_indices)
    ADType = promote_type(typeof(w0), typeof(h10))
    sim = setup_twolayer_simulator(; backend=SinFVM.make_cpu_backend(ADType), w0=ADType(w0), h10=ADType(h10))
    recorder = ObservableRecorder(obs_times=obs_times, cell_indices=cell_indices, data=ADType[])
    SinFVM.simulate_to_time(sim, T; callback=recorder)
    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"
    recorder.data
end

# ---------------------------------------------------------------------------
# Twin experiment setup
# ---------------------------------------------------------------------------

const T_END, H10_FIXED, W0_TRUE = 20.0, 0.75, 1.85
const NX = 64
const CELL_INDICES = collect((NX ÷ 2) .+ [2, 4, 6, 8, 10])
const OBS_TIMES = [8.0, 12.0, 16.0, 20.0]
const EXACT_OBS = simulate_observations(T=T_END, w0=W0_TRUE, h10=H10_FIXED, obs_times=OBS_TIMES, cell_indices=CELL_INDICES)

println("Generated synthetic exact observations:")
println("  W0_TRUE       = $W0_TRUE")
println("  H10_FIXED     = $H10_FIXED")
println("  n_observables = $(length(EXACT_OBS))")

# ---------------------------------------------------------------------------
# Cost function: only observables in the misfit
# ---------------------------------------------------------------------------

const W_ETA, W_U1, W_U2, W_REG = 1.0, 1.0, 1.0, 1e-10

function raw_cost(w0)
    pred = simulate_observations(T=T_END, w0=w0, h10=H10_FIXED, obs_times=OBS_TIMES, cell_indices=CELL_INDICES)
    J = zero(eltype(pred))
    @inbounds for k in 1:3:length(pred)
        dη, du1, du2 = pred[k] - EXACT_OBS[k], pred[k+1] - EXACT_OBS[k+1], pred[k+2] - EXACT_OBS[k+2]
        J += 0.5 * (W_ETA*dη^2 + W_U1*du1^2 + W_U2*du2^2)
    end
    J + 0.5 * W_REG * w0^2
end

const W0_MIN, W0_MAX = 0.5, 3.0
σ(z) = inv(one(z) + exp(-z))
to_w0(z) = W0_MIN + (W0_MAX - W0_MIN) * σ(z)
from_w0(w) = log((w - W0_MIN) / (W0_MAX - w))

function cost_function(zvec)
    w0 = to_w0(zvec[1])
    J = raw_cost(w0)
    J isa ForwardDiff.Dual && @assert all(.!isnan.(J.partials)) "NaN in gradient"
    J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)

# ---------------------------------------------------------------------------
# Optimization: LBFGS on unconstrained variable z
# ---------------------------------------------------------------------------

initial_guess_w0 = 1.0
initial_guess = [from_w0(initial_guess_w0)]

opts = Optim.Options(
    store_trace=true, show_trace=true, show_every=1,
    iterations=8, g_tol=1e-6, f_abstol=1e-8, x_abstol=1e-8,
    allow_f_increases=false,
)

result = optimize(cost_function, grad!, initial_guess, LBFGS(; m=3), opts)

z_opt = Optim.minimizer(result)[1]
w_opt = to_w0(z_opt)
final_cost = raw_cost(w_opt)

println("\n=== Optimization complete ===")
println("True w0        = $(round(W0_TRUE; digits=10))")
println("Recovered w0   = $(round(w_opt; digits=10))")
println("Absolute error = $(round(abs(w_opt - W0_TRUE); digits=12))")
println("Final raw cost = $(round(final_cost; digits=16))")

# ---------------------------------------------------------------------------
# Diagnostic snapshots for plotting
# ---------------------------------------------------------------------------

function snapshot(w0; h10=H10_FIXED, label="")
    sim = setup_twolayer_simulator(backend=SinFVM.make_cpu_backend(), w0=Float64(w0), h10=Float64(h10))
    x = collect(SinFVM.cell_centers(sim.grid))
    SinFVM.simulate_to_time(sim, T_END)
    obs = observable_fields(sim)
    η, u1, u2, w, B = collect(obs.eta), collect(obs.u1), collect(obs.u2), collect(obs.w), collect(obs.Bvals)
    rec = reconstruct_from_observables(η, u1, u2, w, B)
    h1, h2, q1, q2 = collect(rec.h1), collect(rec.h2), collect(rec.q1), collect(rec.q2)
    (; x, B, w, η, u1, u2, h1, h2, q1, q2, label)
end

snap_true = snapshot(W0_TRUE; label="truth")
snap_init = snapshot(initial_guess_w0; label="initial guess")
snap_opt  = snapshot(w_opt; label="optimized")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

cost_vals = try [t.value for t in Optim.trace(result) if hasproperty(t, :value)] catch; Float64[] end

fig = Figure(size=(1500, 1000), fontsize=22)

if !isempty(cost_vals)
    ax_conv = Axis(fig[1, 1:2], title="Twin-experiment convergence: recovering interface parameter w0", xlabel="iteration", ylabel="objective")
    lines!(ax_conv, 1:length(cost_vals), cost_vals, linewidth=2)
    scatter!(ax_conv, 1:length(cost_vals), cost_vals, markersize=8)
else
    Label(fig[1, 1:2], "Twin experiment: true = $(round(W0_TRUE; digits=6)), recovered = $(round(w_opt; digits=6)), cost = $(round(final_cost; digits=10))", fontsize=22)
end

for (row, sn) in enumerate([snap_true, snap_init, snap_opt])
    ax_eta = Axis(fig[row+1, 1], title="$(sn.label): free surface, interface and bathymetry at T=$(T_END)", xlabel="x", ylabel="elevation")
    ax_u   = Axis(fig[row+1, 2], title="$(sn.label): measured velocities at T=$(T_END)", xlabel="x", ylabel="velocity")
    lines!(ax_eta, sn.x, sn.B, linewidth=2, linestyle=:dash, label="B")
    lines!(ax_eta, sn.x, sn.w, linewidth=2, label="w")
    lines!(ax_eta, sn.x, sn.η, linewidth=2, label="η")
    axislegend(ax_eta, position=:lt)
    lines!(ax_u, sn.x, sn.u1, linewidth=2, label="u1")
    lines!(ax_u, sn.x, sn.u2, linewidth=2, label="u2")
    axislegend(ax_u, position=:lt)
end

display(fig)