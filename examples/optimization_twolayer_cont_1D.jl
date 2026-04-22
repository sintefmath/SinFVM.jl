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
using LinearAlgebra

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

const DESING_KAPPA = 1e-4

const NX = 64
const XMIN = 0.0
const XMAX = 100.0

const T_END = 20.0
const OBS_TIMES = [1.0, 2.0, 3.0, 4.0, 10, 20]
const CELL_INDICES = collect(1:NX)

# The TRUE initial free surface is prescribed as epsilon_true = w_true + 0.75
const H1_CONST_ABOVE_INTERFACE = 0.75

const W0_LEFT_TRUE = 1.85
const W0_RIGHT_TRUE = 0.10

const W0_MIN = 0.05
const W0_MAX = 3.00

const X_DAM = 50.0
const TRANSITION_WIDTH_W_TRUE = 10.0

# Constant low initial guess for the control
const W0_INIT_CONST = 0.20

const W_EPS = 10.0
const W_U1  = 10.0
const W_U2  = 10.0
const W_REG_H1 = 1e-1

const LBFGS_M = 20
const LBFGS_MAX_ITERS = 200
const LBFGS_G_SWITCH = 1e-7

const GN_MAX_ITERS = 25
const GN_G_FINAL = 1e-6
const GN_DAMPING0 = 1e-4
const GN_ARMIJO_C1 = 1e-4
const GN_BACKTRACK = 0.5
const GN_MIN_STEP = 1e-6

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

# ---------------------------------------------------------------------------
# Utility: reference grid and smooth profiles
# ---------------------------------------------------------------------------

function make_reference_grid(; nx=NX, xmin=XMIN, xmax=XMAX)
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[xmin xmax])
    x = collect(SinFVM.cell_centers(grid))
    dx = x[2] - x[1]
    return x, dx
end

const X_GRID, DX = make_reference_grid()

function smooth_step_profile(x; left, right, center, width)
    T = promote_type(eltype(x), typeof(left), typeof(right), typeof(center), typeof(width))
    leftT   = T(left)
    rightT  = T(right)
    centerT = T(center)
    widthT  = T(width)

    out = Vector{T}(undef, length(x))
    @inbounds for i in eachindex(x)
        s = (one(T) + tanh((centerT - T(x[i])) / widthT)) / 2
        out[i] = rightT + (leftT - rightT) * s
    end
    out
end

# True interface profile
const W0_TRUE_PROFILE = smooth_step_profile(
    X_GRID;
    left=W0_LEFT_TRUE,
    right=W0_RIGHT_TRUE,
    center=X_DAM,
    width=TRANSITION_WIDTH_W_TRUE,
)

# True initial free surface stays fixed in the inversion
const EPS_TRUE_PROFILE = W0_TRUE_PROFILE .+ H1_CONST_ABOVE_INTERFACE

# Constant low initial guess for interface
const W0_INIT_PROFILE = fill(W0_INIT_CONST, NX)

# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

function setup_twolayer_simulator(;
    backend=SinFVM.make_cpu_backend(),
    ε_profile,
    w0_profile,
)
    @assert length(ε_profile) == NX "ε_profile must have length NX"
    @assert length(w0_profile) == NX "w0_profile must have length NX"

    TT = promote_type(eltype(ε_profile), eltype(w0_profile))

    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])

    B = SinFVM.ConstantBottomTopography(zero(TT))
    eq = SinFVM.TwoLayerShallowWaterEquations1D(
        B;
        ρ1=TT(0.98),
        ρ2=TT(1.00),
        g=TT(9.81),
        depth_cutoff=TT(1e-4),
        desingularizing_kappa=TT(DESING_KAPPA),
    )

    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    flux = SinFVM.PathConservativeCentralUpwind(eq)

    cs = SinFVM.ConservedSystem(
        backend,
        rec,
        flux,
        eq,
        grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()],
    )

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.1)

    ε_cut = TT(1e-4)
    εv = TT.(ε_profile)
    w0v = TT.(w0_profile)

    initial = map(1:NX) do i
        w = max(w0v[i], ε_cut)
        h1 = max(εv[i] - w, ε_cut)
        @SVector([h1, zero(TT), w, zero(TT)])
    end

    SinFVM.set_current_state!(sim, initial)
    sim
end

# ---------------------------------------------------------------------------
# Observables and reconstruction
# ---------------------------------------------------------------------------

bathymetry_values(simulator, ::Type{T}) where {T} =
    fill(zero(T), length(SinFVM.cell_centers(simulator.grid)))

function smooth_positive(h, κ)
    0.5 * (h + sqrt(h^2 + κ^2))
end

smooth_velocity(h, momentum, κ) = momentum / smooth_positive(h, κ)

function observable_fields(simulator)
    st = SinFVM.current_interior_state(simulator)
    Tstate = eltype(st.h1)
    κ = Tstate(DESING_KAPPA)
    Bvals = bathymetry_values(simulator, Tstate)

    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2 = w .- Bvals
    ε = h1 .+ w
    u1 = smooth_velocity.(h1, q1, κ)
    u2 = smooth_velocity.(h2, q2, κ)

    (; ε, u1, u2, w, Bvals)
end

reconstruct_from_observables(ε, u1, u2, w, B) = (;
    h1 = ε .- w,
    h2 = w .- B,
    q1 = (ε .- w) .* u1,
    q2 = (w .- B) .* u2,
)

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
            push!(cb.data, obs.ε[i])
            push!(cb.data, obs.u1[i])
            push!(cb.data, obs.u2[i])
        end
        cb.next_obs += 1
    end
end

# ---------------------------------------------------------------------------
# Forward solve returning observable data vector
# ---------------------------------------------------------------------------

function simulate_observations(; t_end, w0_profile, ε_profile, obs_times, cell_indices)
    ADType = promote_type(eltype(w0_profile), eltype(ε_profile))

    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(ADType),
        ε_profile=ADType.(ε_profile),
        w0_profile=ADType.(w0_profile),
    )

    recorder = ObservableRecorder(obs_times=obs_times, cell_indices=cell_indices, data=ADType[])
    SinFVM.simulate_to_time(sim, t_end; callback=recorder)

    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"
    recorder.data
end

# ---------------------------------------------------------------------------
# Twin experiment setup
# ---------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    t_end=T_END,
    w0_profile=W0_TRUE_PROFILE,
    ε_profile=EPS_TRUE_PROFILE,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

println("Generated synthetic exact observations:")
println("  n_cells          = $NX")
println("  n_obs_times      = $(length(OBS_TIMES))")
println("  n_observables    = $(length(EXACT_OBS))")
println("  LBFGS memory m   = $LBFGS_M")
println("  H1 reg weight    = $W_REG_H1")

# ---------------------------------------------------------------------------
# Control transform: unconstrained z -> bounded physical w0 profile
# ---------------------------------------------------------------------------

σ(z) = inv(one(z) + exp(-z))
to_w0(z) = W0_MIN + (W0_MAX - W0_MIN) * σ(z)
from_w0(w) = log((w - W0_MIN) / (W0_MAX - w))

# ---------------------------------------------------------------------------
# Residual vector and cost
# ---------------------------------------------------------------------------

const N_TRIPLES = length(EXACT_OBS) ÷ 3
const MISFIT_SCALE_EPS = sqrt(W_EPS / N_TRIPLES)
const MISFIT_SCALE_U1  = sqrt(W_U1  / N_TRIPLES)
const MISFIT_SCALE_U2  = sqrt(W_U2  / N_TRIPLES)
const REG_SCALE = sqrt(W_REG_H1 / DX)

function residual_vector(zvec)
    w0_profile = to_w0.(zvec)

    pred = simulate_observations(
        t_end=T_END,
        w0_profile=w0_profile,
        ε_profile=EPS_TRUE_PROFILE,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)
    nmis = length(pred)
    nreg = length(w0_profile) - 1
    r = Vector{T}(undef, nmis + nreg)

    @inbounds for k in 1:3:nmis
        r[k]   = MISFIT_SCALE_EPS * (pred[k]   - EXACT_OBS[k])
        r[k+1] = MISFIT_SCALE_U1  * (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = MISFIT_SCALE_U2  * (pred[k+2] - EXACT_OBS[k+2])
    end

    off = nmis
    @inbounds for i in 1:nreg
        r[off + i] = REG_SCALE * (w0_profile[i+1] - w0_profile[i])
    end

    return r
end

function cost_function(zvec)
    r = residual_vector(zvec)
    J = 0.5 * dot(r, r)
    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end
    J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)
raw_cost(w0_profile) = cost_function(from_w0.(w0_profile))

# ---------------------------------------------------------------------------
# Optimization history
# ---------------------------------------------------------------------------

@with_kw mutable struct OptimizationHistory
    iter::Vector{Int} = Int[]
    phase::Vector{String} = String[]
    J::Vector{Float64} = Float64[]
    gnorm::Vector{Float64} = Float64[]
    w0_profiles::Vector{Vector{Float64}} = Vector{Float64}[]
end

function push_history!(history::OptimizationHistory, iter::Int, phase::String, zvec, J::Real, gnorm::Real)
    push!(history.iter, iter)
    push!(history.phase, phase)
    push!(history.J, Float64(J))
    push!(history.gnorm, Float64(gnorm))
    push!(history.w0_profiles, copy(Float64.(to_w0.(zvec))))
    return history
end

function optim_iteration(state)
    if hasproperty(state, :iteration)
        return Int(getproperty(state, :iteration))
    elseif hasproperty(state, :pseudo_iteration)
        return Int(getproperty(state, :pseudo_iteration))
    else
        return 0
    end
end

function optim_state_x(state)
    hasproperty(state, :x) || error("Could not find parameter vector in Optim state.")
    return getproperty(state, :x)
end

function optim_state_value(state, zvec)
    if hasproperty(state, :value)
        return Float64(getproperty(state, :value))
    elseif hasproperty(state, :f_x)
        return Float64(getproperty(state, :f_x))
    else
        return Float64(cost_function(zvec))
    end
end

function make_history_callback(history::OptimizationHistory)
    last_iter = Ref(-1)

    function cb(state)
        k = optim_iteration(state)
        if k == last_iter[]
            return false
        end
        last_iter[] = k

        z = copy(optim_state_x(state))
        J = optim_state_value(state, z)
        g = ForwardDiff.gradient(cost_function, z)
        gnorm = norm(g)
        push_history!(history, k, "LBFGS", z, J, gnorm)

        return false
    end

    return cb
end

# ---------------------------------------------------------------------------
# Gauss-Newton / Levenberg-Marquardt phase
# ---------------------------------------------------------------------------

function gauss_newton_phase(
    z0;
    history::OptimizationHistory,
    max_iters::Int=GN_MAX_ITERS,
    damping0::Float64=GN_DAMPING0,
    c1::Float64=GN_ARMIJO_C1,
    backtrack::Float64=GN_BACKTRACK,
    min_step::Float64=GN_MIN_STEP,
    g_tol::Float64=GN_G_FINAL,
)
    z = copy(z0)
    μ = damping0

    for k in 1:max_iters
        r = residual_vector(z)
        J = 0.5 * dot(r, r)

        Jr = ForwardDiff.jacobian(residual_vector, z)
        g = Jr' * r
        gnorm = norm(g)

        if gnorm < g_tol
            next_iter = isempty(history.iter) ? 1 : history.iter[end] + 1
            push_history!(history, next_iter, "GN", z, J, gnorm)
            println("GN phase converged on gradient norm.")
            break
        end

        Hgn = Jr' * Jr + μ * I
        δ = -(Hgn \ g)

        α = 1.0
        accepted = false
        Jtrial = J
        ztrial = z

        while α >= min_step
            zcand = z .+ α .* δ
            rcand = residual_vector(zcand)
            Jcand = 0.5 * dot(rcand, rcand)

            if Jcand <= J + c1 * α * dot(g, δ)
                ztrial = zcand
                Jtrial = Jcand
                accepted = true
                break
            end

            α *= backtrack
        end

        if !accepted
            println("GN iter $(k): line search failed, stopping.")
            break
        end

        z = ztrial
        next_iter = isempty(history.iter) ? 1 : history.iter[end] + 1
        push_history!(history, next_iter, "GN", z, Jtrial, gnorm)

        println("GN iter $(k): J = $(round(Jtrial; digits=12)), ‖g‖ = $(round(gnorm; digits=8)), α = $(round(α; digits=6)), μ = $(round(μ; digits=8))")

        if Jtrial < J
            μ = max(μ * 0.5, 1e-8)
        else
            μ = min(μ * 10.0, 1e4)
        end
    end

    return z
end

# ---------------------------------------------------------------------------
# Optimization: LBFGS until switch tolerance, then Gauss-Newton
# ---------------------------------------------------------------------------

initial_guess_profile = copy(W0_INIT_PROFILE)
initial_guess = from_w0.(initial_guess_profile)

initial_grad = ForwardDiff.gradient(cost_function, initial_guess)
history = OptimizationHistory()
push_history!(history, 0, "INIT", initial_guess, cost_function(initial_guess), norm(initial_grad))

opts = Optim.Options(
    store_trace=true,
    show_trace=true,
    show_every=1,
    iterations=LBFGS_MAX_ITERS,
    g_tol=LBFGS_G_SWITCH,
    f_abstol=0.0,
    x_abstol=0.0,
    allow_f_increases=true,
    successive_f_tol=5,
    callback=make_history_callback(history),
)

result_lbfgs = optimize(cost_function, grad!, initial_guess, LBFGS(; m=LBFGS_M), opts)

z_lbfgs = copy(Optim.minimizer(result_lbfgs))
J_lbfgs = cost_function(z_lbfgs)
g_lbfgs = ForwardDiff.gradient(cost_function, z_lbfgs)
gnorm_lbfgs = norm(g_lbfgs)

println("\n=== After LBFGS phase ===")
println("Cost after LBFGS          = $(round(J_lbfgs; digits=12))")
println("Gradient norm after LBFGS = $(round(gnorm_lbfgs; digits=12))")

z_opt = gauss_newton_phase(
    z_lbfgs;
    history=history,
    max_iters=GN_MAX_ITERS,
    g_tol=GN_G_FINAL,
)

w0_opt_profile = to_w0.(z_opt)
final_cost = cost_function(z_opt)
final_grad = ForwardDiff.gradient(cost_function, z_opt)
final_gnorm = norm(final_grad)

println("\n=== Optimization complete ===")
println("Final raw cost   = $(round(final_cost; digits=12))")
println("Final grad norm  = $(round(final_gnorm; digits=12))")
println("Profile L2 error = $(round(norm(w0_opt_profile .- W0_TRUE_PROFILE) / sqrt(length(w0_opt_profile)); digits=12))")

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

function plot_elevation_fields!(
    ax,
    x,
    ε,
    w,
    B;
    legend_position=:lt,
    legend_orientation=:vertical,
    legend_labelsize=16,
)
    lines!(ax, x, ε, linewidth=2, label="ε")
    lines!(ax, x, w, linewidth=2, linestyle=:dash, label="w")
    lines!(ax, x, B, linewidth=2, label="B")

    axislegend(
        ax;
        position=legend_position,
        orientation=legend_orientation,
        labelsize=legend_labelsize,
        patchsize=(18, 10),
        rowgap=4,
        colgap=8,
        framevisible=true,
    )
end

function plot_velocity_fields!(
    ax,
    x,
    u1,
    u2;
    legend_position=:lt,
    legend_orientation=:vertical,
    legend_labelsize=16,
)
    lines!(ax, x, u1, linewidth=2, label="u1")
    lines!(ax, x, u2, linewidth=2, label="u2")

    axislegend(
        ax;
        position=legend_position,
        orientation=legend_orientation,
        labelsize=legend_labelsize,
        patchsize=(18, 10),
        rowgap=4,
        colgap=8,
        framevisible=true,
    )
end

# ---------------------------------------------------------------------------
# Diagnostic snapshots
# ---------------------------------------------------------------------------

function snapshot(w0_profile; ε_profile=EPS_TRUE_PROFILE, label="", t_end=T_END)
    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(),
        ε_profile=Float64.(ε_profile),
        w0_profile=Float64.(w0_profile),
    )

    x = collect(SinFVM.cell_centers(sim.grid))
    SinFVM.simulate_to_time(sim, t_end)
    obs = observable_fields(sim)

    ε  = collect(obs.ε)
    u1 = collect(obs.u1)
    u2 = collect(obs.u2)
    w  = collect(obs.w)
    B  = collect(obs.Bvals)

    rec = reconstruct_from_observables(ε, u1, u2, w, B)
    h1 = collect(rec.h1)
    h2 = collect(rec.h2)
    q1 = collect(rec.q1)
    q2 = collect(rec.q2)

    (; x, B, w, ε, u1, u2, h1, h2, q1, q2, w0_profile=collect(w0_profile), label, t_end)
end

snap_ic  = snapshot(W0_INIT_PROFILE; label="initial condition", t_end=0.0)
snap_syn = snapshot(W0_TRUE_PROFILE; label="synthetic", t_end=T_END)
snap_opt = snapshot(w0_opt_profile; label="optimized", t_end=T_END)

# ---------------------------------------------------------------------------
# Animation similar to the 1D constant case
# ---------------------------------------------------------------------------

function animate_iteration_updates(
    history::OptimizationHistory;
    filename="optimization_profile_iterations.mp4",
    framerate=2,
)
    snaps = [snapshot(wprof; label="$(ph) iter $(k)", t_end=T_END) for (k, ph, wprof) in zip(history.iter, history.phase, history.w0_profiles)]

    x = snaps[1].x

    prof_min = minimum(vcat(W0_TRUE_PROFILE, W0_INIT_PROFILE, reduce(vcat, history.w0_profiles)))
    prof_max = maximum(vcat(W0_TRUE_PROFILE, W0_INIT_PROFILE, reduce(vcat, history.w0_profiles)))
    pad_prof = 0.08 * max(prof_max - prof_min, 1e-8)

    elev_min = minimum(vcat([vcat(sn.ε, sn.w, sn.B) for sn in snaps]...))
    elev_max = maximum(vcat([vcat(sn.ε, sn.w, sn.B) for sn in snaps]...))
    vel_min  = minimum(vcat([vcat(sn.u1, sn.u2) for sn in snaps]...))
    vel_max  = maximum(vcat([vcat(sn.u1, sn.u2) for sn in snaps]...))

    pad_elev = 0.05 * max(elev_max - elev_min, 1e-8)
    pad_vel  = 0.05 * max(vel_max - vel_min, 1e-8)

    prof_obs = Observable(snaps[1].w0_profile)
    ε_obs    = Observable(snaps[1].ε)
    w_obs    = Observable(snaps[1].w)
    B_obs    = Observable(snaps[1].B)
    u1_obs   = Observable(snaps[1].u1)
    u2_obs   = Observable(snaps[1].u2)

    current_iter_obs  = Observable(history.iter[1])
    current_phase_obs = Observable(history.phase[1])
    current_J_obs     = Observable(history.J[1])

    fig = Figure(size=(1500, 900), fontsize=22)

    title_text = @lift "Phase: $current_phase_obs   |   iteration $current_iter_obs   |   J = $(round($current_J_obs; digits=10))"
    Label(fig[1, 1:2], title_text, fontsize=24)

    ax_prof = Axis(
        fig[2, 1:2],
        title="Recovered initial interface profile w₀(x)",
        xlabel="x",
        ylabel="w₀",
    )

    lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true w₀")
    lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial guess")
    lines!(ax_prof, X_GRID, prof_obs, linewidth=3, label="current iterate")
    axislegend(ax_prof, position=:rb)
    ylims!(ax_prof, prof_min - pad_prof, prof_max + pad_prof)

    ax_eps = Axis(
        fig[3, 1],
        title="Free surface, interface and bathymetry at t=$(T_END)",
        xlabel="x",
        ylabel="elevation",
    )
    ax_u = Axis(
        fig[3, 2],
        title="Velocities at t=$(T_END)",
        xlabel="x",
        ylabel="velocity",
    )

    lines!(ax_eps, x, ε_obs, linewidth=2, label="ε")
    lines!(ax_eps, x, w_obs, linewidth=2, linestyle=:dash, label="w")
    lines!(ax_eps, x, B_obs, linewidth=2, label="B")
    axislegend(ax_eps, position=:lt)
    ylims!(ax_eps, elev_min - pad_elev, elev_max + pad_elev)

    lines!(ax_u, x, u1_obs, linewidth=2, label="u1")
    lines!(ax_u, x, u2_obs, linewidth=2, label="u2")
    axislegend(ax_u, position=:lt)
    ylims!(ax_u, vel_min - pad_vel, vel_max + pad_vel)

    record(fig, filename, eachindex(snaps); framerate=framerate) do i
        sn = snaps[i]
        prof_obs[] = sn.w0_profile
        ε_obs[] = sn.ε
        w_obs[] = sn.w
        B_obs[] = sn.B
        u1_obs[] = sn.u1
        u2_obs[] = sn.u2

        current_iter_obs[] = history.iter[i]
        current_phase_obs[] = history.phase[i]
        current_J_obs[] = history.J[i]
    end

    return filename
end

# ---------------------------------------------------------------------------
# Static plot with same structure as the constant-interface script
# ---------------------------------------------------------------------------

fig = Figure(size=(1500, 1300), fontsize=22)

ax_prof = Axis(
    fig[1, 1:2],
    title="Initial interface profiles",
    xlabel="x",
    ylabel="w₀",
)

lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true w₀")
lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial guess")
lines!(ax_prof, X_GRID, w0_opt_profile, linewidth=3, label="optimized w₀")
axislegend(ax_prof, position=:rb)

prof_min = minimum(vcat(W0_TRUE_PROFILE, W0_INIT_PROFILE, w0_opt_profile))
prof_max = maximum(vcat(W0_TRUE_PROFILE, W0_INIT_PROFILE, w0_opt_profile))
pad_prof = 0.08 * max(prof_max - prof_min, 1e-8)
ylims!(ax_prof, prof_min - pad_prof, prof_max + pad_prof)

states = [snap_ic, snap_syn, snap_opt]

for (row, sn) in enumerate(states)
    ax_eps = Axis(
        fig[row + 1, 1],
        title="$(sn.label): free surface, interface and bathymetry at t=$(sn.t_end)",
        xlabel="x",
        ylabel="elevation",
    )
    ax_u = Axis(
        fig[row + 1, 2],
        title="$(sn.label): velocities at t=$(sn.t_end)",
        xlabel="x",
        ylabel="velocity",
    )

    if sn.label == "initial condition"
        plot_elevation_fields!(
            ax_eps, sn.x, sn.ε, sn.w, sn.B;
            legend_position=:rt,
            legend_orientation=:vertical,
            legend_labelsize=16,
        )
        plot_velocity_fields!(
            ax_u, sn.x, sn.u1, sn.u2;
            legend_position=:rt,
            legend_orientation=:vertical,
            legend_labelsize=16,
        )
    else
        plot_elevation_fields!(
            ax_eps, sn.x, sn.ε, sn.w, sn.B;
            legend_position=:lt,
            legend_orientation=:horizontal,
            legend_labelsize=14,
        )
        plot_velocity_fields!(
            ax_u, sn.x, sn.u1, sn.u2;
            legend_position=:lt,
            legend_orientation=:horizontal,
            legend_labelsize=14,
        )
    end
end

display(fig)

# ---------------------------------------------------------------------------
# Save figure and animation
# ---------------------------------------------------------------------------

fig_file = joinpath(SAVE_DIR, "optimization_profile_result.png")
save(fig_file, fig)
println("Saved figure to: $fig_file")

anim_file = animate_iteration_updates(
    history;
    filename=joinpath(SAVE_DIR, "optimization_profile_iterations.mp4"),
    framerate=2,
)
println("Saved animation to: $anim_file")