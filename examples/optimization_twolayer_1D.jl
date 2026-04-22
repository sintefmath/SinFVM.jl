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
# Global settings
# ---------------------------------------------------------------------------

const DESING_KAPPA = 1e-4

save_dir = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(save_dir)

# ---------------------------------------------------------------------------
# Configuration:
# sharp dam-break in h1, constant interface parameter w0 everywhere
# ---------------------------------------------------------------------------

const NX = 64
const DOMAIN_XMIN = 0.0
const DOMAIN_XMAX = 100.0
const X_DAM = 50.0

const T_END = 20.0
const H1_LEFT = 1.0
const H1_RIGHT = 0.20
const W0_TRUE = 0.10

# Observe most/all of the domain
const CELL_INDICES = collect(1:NX)
const OBS_TIMES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

# Weighted misfit only
const W_EPS, W_U1, W_U2 = 0.1, 0.01, 0.01

# Scalar bounds for optimization
const W0_MIN, W0_MAX = 1e-4, 1.0

# ---------------------------------------------------------------------------
# Simulator factory
# ---------------------------------------------------------------------------

function setup_twolayer_simulator(;
    backend=SinFVM.make_cpu_backend(),
    w0,
    h10,
    init_type::Symbol = :sharp_dam,
    x_dam = X_DAM,
    transition_width = 6.0,
    h1_right = H1_RIGHT,
    w_right = w0,   # constant interface everywhere by default
)
    TT = promote_type(typeof(w0), typeof(h10), typeof(x_dam),
                      typeof(transition_width), typeof(h1_right), typeof(w_right))

    w0 = TT(w0)
    h10 = TT(h10)
    x_dam = TT(x_dam)
    transition_width = TT(transition_width)
    h1_right = TT(h1_right)
    w_right = TT(w_right)

    grid = SinFVM.CartesianGrid(
        NX;
        gc=2,
        boundary=SinFVM.WallBC(),
        extent=[DOMAIN_XMIN DOMAIN_XMAX],
    )

    x = SinFVM.cell_centers(grid)
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

    initial = map(x) do xi
        if init_type == :sharp_dam
            h1 = xi < x_dam ? h10 : max(h1_right, ε_cut)
            w  = xi < x_dam ? w0  : max(w_right, ε_cut)

        elseif init_type == :smooth_dam
            s = (one(TT) + tanh((x_dam - xi) / transition_width)) / 2
            h1 = h1_right + (h10 - h1_right) * s
            w  = w_right  + (w0  - w_right)  * s
            h1 = max(h1, ε_cut)
            w  = max(w, ε_cut)

        else
            error("Unknown init_type = $init_type. Use :sharp_dam or :smooth_dam")
        end

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

function desingularize(h, κ)
    copysign(one(h), h) * max(abs(h), min(h^2 / (2 * κ) + κ / 2, κ))
end

desingularize(h, momentum, κ) = momentum / desingularize(h, κ)

function observable_fields(simulator)
    st = SinFVM.current_interior_state(simulator)
    Tstate = eltype(st.h1)
    κ = Tstate(DESING_KAPPA)
    Bvals = bathymetry_values(simulator, Tstate)

    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2 = w .- Bvals
    ε = h1 .+ w

    u1 = desingularize.(h1, q1, κ)
    u2 = desingularize.(h2, q2, κ)

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

function simulate_observations(; t_end, w0, h10, obs_times, cell_indices)
    ADType = promote_type(typeof(w0), typeof(h10))

    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(ADType),
        w0=ADType(w0),
        h10=ADType(h10),
        init_type=:sharp_dam,
        x_dam=ADType(X_DAM),
        transition_width=ADType(6.0),
        h1_right=ADType(H1_RIGHT),
        w_right=ADType(w0),   # constant interface everywhere
    )

    recorder = ObservableRecorder(
        obs_times=obs_times,
        cell_indices=cell_indices,
        data=ADType[],
    )

    SinFVM.simulate_to_time(sim, t_end; callback=recorder)
    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"
    recorder.data
end

# ---------------------------------------------------------------------------
# Twin experiment setup
# ---------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    t_end=T_END,
    w0=W0_TRUE,
    h10=H1_LEFT,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

println("Generated synthetic exact observations:")
println("  W0_TRUE       = $W0_TRUE")
println("  H1_LEFT       = $H1_LEFT")
println("  H1_RIGHT      = $H1_RIGHT")
println("  n_cells_obs   = $(length(CELL_INDICES))")
println("  n_observables = $(length(EXACT_OBS))")

# ---------------------------------------------------------------------------
# Cost function: weighted misfit only
# ---------------------------------------------------------------------------

function raw_cost(w0)
    pred = simulate_observations(
        t_end=T_END,
        w0=w0,
        h10=H1_LEFT,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    J = zero(eltype(pred))
    @inbounds for k in 1:3:length(pred)
        dε  = pred[k]   - EXACT_OBS[k]
        du1 = pred[k+1] - EXACT_OBS[k+1]
        du2 = pred[k+2] - EXACT_OBS[k+2]
        J += 0.5 * (W_EPS * dε^2 + W_U1 * du1^2 + W_U2 * du2^2)
    end
    return J
end

σ(z) = inv(one(z) + exp(-z))
to_w0(z) = W0_MIN + (W0_MAX - W0_MIN) * σ(z)
from_w0(w) = log((w - W0_MIN) / (W0_MAX - w))

function cost_function(zvec)
    w0 = to_w0(zvec[1])
    J = raw_cost(w0)

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)

# ---------------------------------------------------------------------------
# Optimization history storage
# ---------------------------------------------------------------------------

@with_kw mutable struct OptimizationHistory
    iter::Vector{Int} = Int[]
    z::Vector{Float64} = Float64[]
    w0::Vector{Float64} = Float64[]
    J::Vector{Float64} = Float64[]
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
    if hasproperty(state, :x)
        return getproperty(state, :x)
    else
        error("Could not find parameter vector in Optim state.")
    end
end

function optim_state_value(state, w0)
    if hasproperty(state, :value)
        return Float64(getproperty(state, :value))
    elseif hasproperty(state, :f_x)
        return Float64(getproperty(state, :f_x))
    else
        return Float64(raw_cost(w0))
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

        xstate = optim_state_x(state)
        z = Float64(xstate[1])
        w0 = Float64(to_w0(z))
        J = optim_state_value(state, w0)

        push!(history.iter, k)
        push!(history.z, z)
        push!(history.w0, w0)
        push!(history.J, J)

        return false
    end

    return cb
end

# ---------------------------------------------------------------------------
# Optimization: LBFGS on unconstrained variable z
# ---------------------------------------------------------------------------

initial_guess_w0 = 0.40
initial_guess = [from_w0(initial_guess_w0)]

history = OptimizationHistory()
push!(history.iter, 0)
push!(history.z, Float64(initial_guess[1]))
push!(history.w0, Float64(initial_guess_w0))
push!(history.J, Float64(raw_cost(initial_guess_w0)))

opts = Optim.Options(
    store_trace=true,
    show_trace=true,
    show_every=1,
    iterations=20,
    g_tol=1e-8,
    f_abstol=1e-10,
    x_abstol=1e-10,
    allow_f_increases=false,
    callback=make_history_callback(history),
)

result = optimize(cost_function, grad!, initial_guess,  LBFGS(; m=5), opts)

z_opt = Optim.minimizer(result)[1]
w_opt = to_w0(z_opt)
final_cost = raw_cost(w_opt)

println("\n=== Optimization complete ===")
println("True w0        = $(round(W0_TRUE; digits=10))")
println("Recovered w0   = $(round(w_opt; digits=10))")
println("Absolute error = $(round(abs(w_opt - W0_TRUE); digits=12))")
println("Final raw cost = $(round(final_cost; digits=16))")

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
    lines!(ax, x, B, linewidth=2, label="B")   # solid line for B

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
# Diagnostic snapshots for plotting
# ---------------------------------------------------------------------------

function snapshot(w0; h10=H1_LEFT, label="", t_end=T_END)
    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(),
        w0=Float64(w0),
        h10=Float64(h10),
        init_type=:sharp_dam,
        x_dam=Float64(X_DAM),
        transition_width=6.0,
        h1_right=Float64(H1_RIGHT),
        w_right=Float64(w0),   # constant interface everywhere
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

    (; x, B, w, ε, u1, u2, h1, h2, q1, q2, label, t_end)
end

snap_ic  = snapshot(W0_TRUE; label="initial condition", t_end=0.0)
snap_syn = snapshot(W0_TRUE; label="synthetic", t_end=T_END)
snap_opt = snapshot(w_opt;   label="optimized", t_end=T_END)

# ---------------------------------------------------------------------------
# Animation of optimization iterates
# ---------------------------------------------------------------------------

function animate_iteration_updates(
    history::OptimizationHistory;
    t_end=T_END,
    filename="optimization_iterations.mp4",
    framerate=2,
)
    snaps = [
        snapshot(w0; label="iter $(k)", t_end=t_end)
        for (k, w0) in zip(history.iter, history.w0)
    ]

    x = snaps[1].x

    elev_min = minimum(vcat([vcat(sn.ε, sn.w, sn.B) for sn in snaps]...))
    elev_max = maximum(vcat([vcat(sn.ε, sn.w, sn.B) for sn in snaps]...))
    vel_min  = minimum(vcat([vcat(sn.u1, sn.u2) for sn in snaps]...))
    vel_max  = maximum(vcat([vcat(sn.u1, sn.u2) for sn in snaps]...))

    pad_elev = 0.05 * max(elev_max - elev_min, 1e-8)
    pad_vel  = 0.05 * max(vel_max - vel_min, 1e-8)

    ε_obs  = Observable(snaps[1].ε)
    w_obs  = Observable(snaps[1].w)
    B_obs  = Observable(snaps[1].B)
    u1_obs = Observable(snaps[1].u1)
    u2_obs = Observable(snaps[1].u2)

    current_iter_obs = Observable(history.iter[1])
    current_w0_obs   = Observable(history.w0[1])
    current_k_idx    = Observable(1)

    fig = Figure(size=(1500, 900), fontsize=22)

    title_text = @lift "Optimization iteration $current_iter_obs   |   recovered w0 = $(round($current_w0_obs; digits=8))   |   true w0 = $(round(W0_TRUE; digits=8))"
    Label(fig[1, 1:2], title_text, fontsize=24)

    ax_eps = Axis(
        fig[2, 1],
        title="Free surface, interface and bathymetry at t=$(t_end)",
        xlabel="x",
        ylabel="elevation",
    )
    ax_u = Axis(
        fig[2, 2],
        title="Measured velocities at t=$(t_end)",
        xlabel="x",
        ylabel="velocity",
    )
    ax_w0 = Axis(
        fig[3, 1:2],
        title="Convergence of w0",
        xlabel="iteration",
        ylabel="w0",
    )

    lines!(ax_w0, history.iter, history.w0, linewidth=2, label="recovered w0")
    scatter!(ax_w0, history.iter, history.w0, markersize=10)
    lines!(ax_w0, history.iter, fill(W0_TRUE, length(history.iter)), linewidth=2, linestyle=:dash, label="true w0")

    marker_x = @lift [history.iter[$current_k_idx]]
    marker_y = @lift [history.w0[$current_k_idx]]
    scatter!(ax_w0, marker_x, marker_y, markersize=18, label="current iterate")

    axislegend(ax_w0, position=:rb)

    ylo = min(minimum(history.w0), W0_TRUE)
    yhi = max(maximum(history.w0), W0_TRUE)
    pady = 0.08 * max(yhi - ylo, 1e-8)
    ylims!(ax_w0, ylo - pady, yhi + pady)

    lines!(ax_eps, x, ε_obs, linewidth=2, label="ε")
    lines!(ax_eps, x, w_obs, linewidth=2, linestyle=:dash, label="w")
    lines!(ax_eps, x, B_obs, linewidth=2, label="B")
    axislegend(
        ax_eps;
        position=:lt,
        orientation=:horizontal,
        labelsize=15,
        patchsize=(18, 10),
        rowgap=4,
        colgap=8,
        framevisible=true,
    )
    ylims!(ax_eps, elev_min - pad_elev, elev_max + pad_elev)

    lines!(ax_u, x, u1_obs, linewidth=2, label="u1")
    lines!(ax_u, x, u2_obs, linewidth=2, label="u2")
    axislegend(
        ax_u;
        position=:lt,
        orientation=:horizontal,
        labelsize=15,
        patchsize=(18, 10),
        rowgap=4,
        colgap=8,
        framevisible=true,
    )
    ylims!(ax_u, vel_min - pad_vel, vel_max + pad_vel)

    record(fig, filename, eachindex(snaps); framerate=framerate) do i
        sn = snaps[i]
        ε_obs[] = sn.ε
        w_obs[] = sn.w
        B_obs[] = sn.B
        u1_obs[] = sn.u1
        u2_obs[] = sn.u2

        current_iter_obs[] = history.iter[i]
        current_w0_obs[] = history.w0[i]
        current_k_idx[] = i
    end

    return filename
end

# ---------------------------------------------------------------------------
# Plot: iteration history + initial/synthetic/optimized states only
# ---------------------------------------------------------------------------

fig = Figure(size=(1500, 1100), fontsize=22)

ax_w0 = Axis(
    fig[1, 1:2],
    title="Twin-experiment convergence: recovering constant interface parameter w0",
    xlabel="iteration",
    ylabel="w0",
)

lines!(ax_w0, history.iter, history.w0, linewidth=2, label="recovered w0")
scatter!(ax_w0, history.iter, history.w0, markersize=8)
lines!(ax_w0, history.iter, fill(W0_TRUE, length(history.iter)), linewidth=2, linestyle=:dash, label="true w0 = $W0_TRUE")
axislegend(ax_w0, position=:rb)

ylo = min(minimum(history.w0), W0_TRUE)
yhi = max(maximum(history.w0), W0_TRUE)
pady = 0.08 * max(yhi - ylo, 1e-8)
ylims!(ax_w0, ylo - pady, yhi + pady)

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
save(joinpath(save_dir, "optimization_iterations.png"), fig)

# ---------------------------------------------------------------------------
# Create animation
# ---------------------------------------------------------------------------

anim_file = animate_iteration_updates(
    history;
    filename=joinpath(save_dir, "optimization_1D.mp4"),
    framerate=2,
)

println("Saved optimization figure to: $(joinpath(save_dir, "optimization_iterations.png"))")
println("Saved animation to: $anim_file")