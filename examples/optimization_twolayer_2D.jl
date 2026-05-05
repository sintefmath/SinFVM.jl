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
using LinearAlgebra, Statistics, Printf

# -----------------------------------------------------------------------------
# 2D twin experiment:
# recover a spatially varying initial interface w0(x,y).
#
# Important modeling choice in this version:
#   - The initial free-surface elevation η0 is fixed and is NOT optimized.
#   - The optimizer changes only the interface position w0.
#   - Therefore h1 = η0 - w0 changes when w0 changes.
#   - The initial guess for w0 is constant over the wet half.
#
# Observables used in the cost:
#   η, u1, v1, u2, v2
#
# Internal state:
#   U = (h1, q1, p1, w, q2, p2), where w = h2 + B.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

const NX, NY, GC = 64, 64, 2
const XMIN, XMAX = 0.0, 100.0
const YMIN, YMAX = 0.0, 50.0
const X_DAM = 50.0
const Y_MID = 25.0

const CFL_2D = 0.2
const DEPTH_CUT = 1e-4

const T_END = 6.0
const OBS_TIMES = [2.0, 4.0, 6.0]

# Initial interface bounds for the wet-half coefficient parameterization.
const W_MIN, W_MAX = 0.5, 3.0

# Constant initial guess for the interface over the wet half.
const W0_INIT_CONST = 1.0

# Fixed initial free surface. This is not optimized.
# Choose this high enough so that η0 - w0 remains positive for all admissible w0.
const ETA0_FIXED = 3.25

# True 4-parameter initial interface field on wet half:
# [bottom-left, top-left, bottom-right, top-right] within x < X_DAM.
const WCOEFFS_TRUE = [1.80, 2.10, 1.65, 1.95]

# Observation cells. Avoid boundary cells initially; add them later if desired.
const CELL_INDICES = [(4, 4), (8, 8), (12, 12), (20, 20), (30, 30)]

# Original objective weights.
const W_ETA_ORIG = 1.0
const W_U1_ORIG  = 0.5
const W_V1_ORIG  = 0.5
const W_U2_ORIG  = 0.5
const W_V2_ORIG  = 0.5
const W_REG_ORIG = 1e-8

# Weight clipping for adaptive sensitivity-based weighting.
const WEIGHT_MIN_CLIP = 1e-2
const WEIGHT_MAX_CLIP = 1e6

# Optimizer settings.
const LBFGS_M = 5
const LBFGS_MAX_ITERS = 40
const LBFGS_G_TOL = 1e-8

# -----------------------------------------------------------------------------
# Bounded transform for coefficients
# -----------------------------------------------------------------------------

σ(z) = inv(one(z) + exp(-z))
to_w(z) = W_MIN + (W_MAX - W_MIN) * σ(z)
from_w(w) = log((w - W_MIN) / (W_MAX - w))

# -----------------------------------------------------------------------------
# Bathymetry builders
# -----------------------------------------------------------------------------

function make_bottom_cos_sin_2d(;
    B0=-3.0,
    Ax=0.4,
    Ay=0.3,
    mx=1,
    my=1,
    φx=0.0,
    φy=0.0,
    backend,
    grid::SinFVM.CartesianGrid{2},
)
    x_faces = SinFVM.cell_faces(grid, SinFVM.XDIR; interior=false)
    y_faces = SinFVM.cell_faces(grid, SinFVM.YDIR; interior=false)

    nxg, nyg = length(x_faces), length(y_faces)

    x0, x1 = SinFVM.start_extent(grid, SinFVM.XDIR), SinFVM.end_extent(grid, SinFVM.XDIR)
    y0, y1 = SinFVM.start_extent(grid, SinFVM.YDIR), SinFVM.end_extent(grid, SinFVM.YDIR)

    Lx, Ly = x1 - x0, y1 - y0
    B = Matrix{Float64}(undef, nxg, nyg)

    @inbounds for j in 1:nyg, i in 1:nxg
        xhat = (x_faces[i] - x0) / Lx
        yhat = (y_faces[j] - y0) / Ly
        xhat -= floor(xhat)
        yhat -= floor(yhat)
        B[i, j] = B0 + Ax * cos(2π * mx * xhat + φx) + Ay * sin(2π * my * yhat + φy)
    end

    return SinFVM.BottomTopography2D(B, backend, grid)
end

# -----------------------------------------------------------------------------
# 2x2 block parameterization of w0(x,y) on wet half
# coeffs = [bottom-left, top-left, bottom-right, top-right] within x < X_DAM.
# -----------------------------------------------------------------------------

function w0_from_coeffs(xy, coeffs; x_dam=X_DAM, y_mid=Y_MID, dry_eps=DEPTH_CUT)
    x, y = xy

    if x >= x_dam
        return eltype(coeffs)(dry_eps)
    end

    left_right = x < x_dam / 2 ? 0 : 1
    bot_top = y < y_mid ? 0 : 1
    idx = 1 + left_right + 2 * bot_top

    return coeffs[idx]
end

function eta0_profile(xy, ::Type{T}) where {T}
    x, y = xy
    if x < X_DAM
        return T(ETA0_FIXED)
    else
        return T(DEPTH_CUT)
    end
end

# -----------------------------------------------------------------------------
# Simulator factory
# -----------------------------------------------------------------------------

function setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wcoeffs)
    T = eltype(wcoeffs)

    grid = SinFVM.CartesianGrid(
        NX,
        NY;
        gc=GC,
        boundary=SinFVM.PeriodicBC(),
        extent=[XMIN XMAX; YMIN YMAX],
    )

    bottom = make_bottom_cos_sin_2d(;
        B0=-3.0,
        Ax=0.4,
        Ay=0.3,
        mx=1,
        my=1,
        backend=backend,
        grid=grid,
    )

    eq = SinFVM.TwoLayerShallowWaterEquations2D(
        bottom;
        ρ1=T(1.00),
        ρ2=T(1.02),
        g=T(9.81),
        depth_cutoff=T(DEPTH_CUT),
    )

    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)

    cs = SinFVM.ConservedSystem(
        backend,
        rec,
        flux,
        eq,
        grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()],
    )

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=CFL_2D)

    xy_int = SinFVM.cell_centers(grid; interior=true)
    B_int = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    εh = T(DEPTH_CUT)

    initial = [begin
        xy = xy_int[I]
        x = xy[1]

        if x < X_DAM
            w0_raw = T(w0_from_coeffs(xy, wcoeffs; x_dam=X_DAM, y_mid=Y_MID, dry_eps=εh))
            η0 = eta0_profile(xy, T)

            # Keep both layers wet and do not change η0.
            # This means changing w0 changes h1 and h2, while η0 stays fixed.
            lower_w = T(B_int[I]) + εh
            upper_w = η0 - εh
            w0 = clamp(w0_raw, lower_w, upper_w)

            h1 = η0 - w0
            h2 = w0 - T(B_int[I])

            h1 <= εh / 2 && error("Initial w0 makes h1 too small at xy=$xy: h1=$h1")
            h2 <= εh / 2 && error("Initial w0 makes h2 too small at xy=$xy: h2=$h2")

            @SVector [h1, zero(T), zero(T), w0, zero(T), zero(T)]
        else
            # Dry/right side. Keep a thin positive layer for numerical safety.
            wdry = max(T(B_int[I]) + εh, εh)
            @SVector [εh, zero(T), zero(T), wdry, zero(T), zero(T)]
        end
    end for I in eachindex(xy_int)]

    SinFVM.set_current_state!(sim, initial)

    return sim, eq, grid
end

# -----------------------------------------------------------------------------
# Observables and reconstruction
# -----------------------------------------------------------------------------

function observable_fields(sim, eq, grid)
    st = SinFVM.current_interior_state(sim)
    Bcell = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    h1 = st.h1
    q1 = st.q1
    p1 = st.p1
    w  = st.w
    q2 = st.q2
    p2 = st.p2

    h2 = w .- Bcell
    η  = h1 .+ w

    u1 = SinFVM.desingularize.(Ref(eq), h1, q1)
    v1 = SinFVM.desingularize.(Ref(eq), h1, p1)
    u2 = SinFVM.desingularize.(Ref(eq), h2, q2)
    v2 = SinFVM.desingularize.(Ref(eq), h2, p2)

    return (; Bcell, h1, q1, p1, w, q2, p2, h2, η, u1, v1, u2, v2)
end

reconstruct_from_observables(η, u1, v1, u2, v2, w, B) = (;
    h1 = η .- w,
    h2 = w .- B,
    q1 = (η .- w) .* u1,
    p1 = (η .- w) .* v1,
    q2 = (w .- B) .* u2,
    p2 = (w .- B) .* v2,
)

# -----------------------------------------------------------------------------
# Observation callback
# cell_indices are tuples like (i,j).
# -----------------------------------------------------------------------------

@with_kw mutable struct ObservableRecorder{VT,IT,OT}
    obs_times::VT
    cell_indices::IT
    next_obs::Int = 1
    data::Vector{OT} = OT[]
end

function (cb::ObservableRecorder)(time, simulator)
    t = ForwardDiff.value(time)
    sim = simulator
    eq = sim.system.equation
    grid = sim.grid

    while cb.next_obs <= length(cb.obs_times) && t + 1e-12 >= cb.obs_times[cb.next_obs]
        obs = observable_fields(sim, eq, grid)

        for (i, j) in cb.cell_indices
            push!(cb.data, obs.η[i, j])
            push!(cb.data, obs.u1[i, j])
            push!(cb.data, obs.v1[i, j])
            push!(cb.data, obs.u2[i, j])
            push!(cb.data, obs.v2[i, j])
        end

        cb.next_obs += 1
    end
end

# -----------------------------------------------------------------------------
# Forward solve
# -----------------------------------------------------------------------------

function simulate_observations(; T_end, wcoeffs, obs_times, cell_indices)
    ADType = eltype(wcoeffs)

    sim, _, _ = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(ADType),
        wcoeffs=ADType.(wcoeffs),
    )

    recorder = ObservableRecorder(
        obs_times=obs_times,
        cell_indices=cell_indices,
        data=ADType[],
    )

    SinFVM.simulate_to_time(sim, T_end; callback=recorder)

    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"

    return recorder.data
end

# -----------------------------------------------------------------------------
# Synthetic observations
# -----------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    T_end=T_END,
    wcoeffs=WCOEFFS_TRUE,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

println("Generated synthetic exact observations:")
println("  true coeffs   = $(round.(WCOEFFS_TRUE; digits=4))")
println("  constant η0   = $ETA0_FIXED")
println("  initial guess = $(fill(W0_INIT_CONST, 4))")
println("  CFL_2D        = $CFL_2D")
println("  n_observables = $(length(EXACT_OBS))")

# -----------------------------------------------------------------------------
# Adaptive weights config
# -----------------------------------------------------------------------------

const N_OBS_GROUPS = length(EXACT_OBS) ÷ 5

const ADAPTIVE_WEIGHTS_CONFIG_2D = Dict{Symbol,Any}(
    :use_adaptive => false,
    :W_eta => W_ETA_ORIG,
    :W_u1  => W_U1_ORIG,
    :W_v1  => W_V1_ORIG,
    :W_u2  => W_U2_ORIG,
    :W_v2  => W_V2_ORIG,
    :W_reg => W_REG_ORIG,
)

function reset_to_original_weights_2d!()
    ADAPTIVE_WEIGHTS_CONFIG_2D[:use_adaptive] = false
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_eta] = W_ETA_ORIG
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u1]  = W_U1_ORIG
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v1]  = W_V1_ORIG
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u2]  = W_U2_ORIG
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v2]  = W_V2_ORIG
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_reg] = W_REG_ORIG
    return nothing
end

function set_adaptive_weights_2d!(W_eta, W_u1, W_v1, W_u2, W_v2; W_reg=W_REG_ORIG)
    ADAPTIVE_WEIGHTS_CONFIG_2D[:use_adaptive] = true
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_eta] = W_eta
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u1]  = W_u1
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v1]  = W_v1
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u2]  = W_u2
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v2]  = W_v2
    ADAPTIVE_WEIGHTS_CONFIG_2D[:W_reg] = W_reg
    return nothing
end

function current_weights_2d()
    return (
        W_eta = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_eta],
        W_u1  = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u1],
        W_v1  = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v1],
        W_u2  = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_u2],
        W_v2  = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_v2],
        W_reg = ADAPTIVE_WEIGHTS_CONFIG_2D[:W_reg],
    )
end

# -----------------------------------------------------------------------------
# Residual and cost
# -----------------------------------------------------------------------------

function residual_vector_wcoeffs(wcoeffs)
    pred = simulate_observations(
        T_end=T_END,
        wcoeffs=wcoeffs,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    T = eltype(pred)
    weights = current_weights_2d()

    nmis = length(pred)
    nreg = length(wcoeffs)
    r = Vector{T}(undef, nmis + nreg)

    # Use sqrt(W / N_OBS_GROUPS) to keep cost magnitude roughly independent of
    # the number of observations.
    sη  = sqrt(T(weights.W_eta) / T(N_OBS_GROUPS))
    su1 = sqrt(T(weights.W_u1)  / T(N_OBS_GROUPS))
    sv1 = sqrt(T(weights.W_v1)  / T(N_OBS_GROUPS))
    su2 = sqrt(T(weights.W_u2)  / T(N_OBS_GROUPS))
    sv2 = sqrt(T(weights.W_v2)  / T(N_OBS_GROUPS))

    @inbounds for k in 1:5:nmis
        r[k]   = sη  * (pred[k]   - EXACT_OBS[k])
        r[k+1] = su1 * (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = sv1 * (pred[k+2] - EXACT_OBS[k+2])
        r[k+3] = su2 * (pred[k+3] - EXACT_OBS[k+3])
        r[k+4] = sv2 * (pred[k+4] - EXACT_OBS[k+4])
    end

    sreg = sqrt(T(weights.W_reg))
    @inbounds for i in eachindex(wcoeffs)
        r[nmis + i] = sreg * wcoeffs[i]
    end

    return r
end

raw_cost(wcoeffs) = 0.5 * dot(residual_vector_wcoeffs(wcoeffs), residual_vector_wcoeffs(wcoeffs))

function cost_function(zvec)
    wcoeffs = to_w.(zvec)
    J = raw_cost(wcoeffs)

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)

# -----------------------------------------------------------------------------
# Sensitivity-based adaptive weighting
# -----------------------------------------------------------------------------

function observable_jacobian_sensitivities_2d(wcoeffs_ref)
    obs_fun = w -> simulate_observations(
        T_end=T_END,
        wcoeffs=w,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    Jobs = ForwardDiff.jacobian(obs_fun, wcoeffs_ref)

    η_rows  = 1:5:size(Jobs, 1)
    u1_rows = 2:5:size(Jobs, 1)
    v1_rows = 3:5:size(Jobs, 1)
    u2_rows = 4:5:size(Jobs, 1)
    v2_rows = 5:5:size(Jobs, 1)

    sens_η  = norm(Jobs[η_rows,  :]) / sqrt(length(η_rows))
    sens_u1 = norm(Jobs[u1_rows, :]) / sqrt(length(u1_rows))
    sens_v1 = norm(Jobs[v1_rows, :]) / sqrt(length(v1_rows))
    sens_u2 = norm(Jobs[u2_rows, :]) / sqrt(length(u2_rows))
    sens_v2 = norm(Jobs[v2_rows, :]) / sqrt(length(v2_rows))

    return (; sens_η, sens_u1, sens_v1, sens_u2, sens_v2, Jobs)
end

function adaptive_weights_from_observable_sensitivities_2d(wcoeffs_ref; min_clip=WEIGHT_MIN_CLIP, max_clip=WEIGHT_MAX_CLIP)
    sens = observable_jacobian_sensitivities_2d(wcoeffs_ref)
    vals = [sens.sens_η, sens.sens_u1, sens.sens_v1, sens.sens_u2, sens.sens_v2]

    ref = maximum(vals)
    safe = max(1e-14 * ref, eps(Float64))

    # Since residuals use sqrt(W), equalizing Jacobian magnitudes uses W ≈ (ref/sens)^2.
    Wη  = (ref / max(sens.sens_η,  safe))^2
    Wu1 = (ref / max(sens.sens_u1, safe))^2
    Wv1 = (ref / max(sens.sens_v1, safe))^2
    Wu2 = (ref / max(sens.sens_u2, safe))^2
    Wv2 = (ref / max(sens.sens_v2, safe))^2

    return (
        clamp(Wη,  min_clip, max_clip),
        clamp(Wu1, min_clip, max_clip),
        clamp(Wv1, min_clip, max_clip),
        clamp(Wu2, min_clip, max_clip),
        clamp(Wv2, min_clip, max_clip),
        sens,
    )
end

function plot_sensitivity_diagnostics_2d(wcoeffs_ref, filename)
    reset_to_original_weights_2d!()

    sens = observable_jacobian_sensitivities_2d(wcoeffs_ref)
    Jobs = sens.Jobs

    col_norms = vec(sqrt.(sum(abs2, Jobs; dims=1)))
    row_norms = vec(sqrt.(sum(abs2, Jobs; dims=2)))

    nmis = length(EXACT_OBS)
    row_η  = row_norms[1:5:nmis]
    row_u1 = row_norms[2:5:nmis]
    row_v1 = row_norms[3:5:nmis]
    row_u2 = row_norms[4:5:nmis]
    row_v2 = row_norms[5:5:nmis]

    fig = Figure(size=(1400, 900), fontsize=18)

    ax1 = Axis(fig[1, 1], title="Coefficient sensitivities", xlabel="Coefficient index", ylabel="Jacobian column norm")
    lines!(ax1, 1:length(col_norms), col_norms, linewidth=2)
    scatter!(ax1, 1:length(col_norms), col_norms, markersize=10)

    ax2 = Axis(fig[1, 2], title="Observation row sensitivities", xlabel="Observation index per type", ylabel="Row norm")
    lines!(ax2, 1:length(row_η),  row_η,  label="η", linewidth=2)
    lines!(ax2, 1:length(row_u1), row_u1, label="u1", linewidth=2)
    lines!(ax2, 1:length(row_v1), row_v1, label="v1", linewidth=2)
    lines!(ax2, 1:length(row_u2), row_u2, label="u2", linewidth=2)
    lines!(ax2, 1:length(row_v2), row_v2, label="v2", linewidth=2)
    axislegend(ax2, position=:rb)

    ax3 = Axis(fig[2, 1:2], title="Observable block sensitivities", xlabel="Observable", ylabel="Sensitivity", yscale=log10)
    labels = ["η", "u1", "v1", "u2", "v2"]
    vals = [sens.sens_η, sens.sens_u1, sens.sens_v1, sens.sens_u2, sens.sens_v2]
    xpos = 1:5
    barplot!(ax3, xpos, vals)
    ax3.xticks = (xpos, labels)

    save(filename, fig)
    println("Saved sensitivity diagnostics to: $filename")

    return fig
end

# -----------------------------------------------------------------------------
# Optimization helpers
# -----------------------------------------------------------------------------

function run_optimization(label; initial_guess_coeffs, iterations=LBFGS_MAX_ITERS)
    initial_guess = from_w.(initial_guess_coeffs)

    opts = Optim.Options(
        store_trace=true,
        show_trace=true,
        show_every=1,
        iterations=iterations,
        g_tol=LBFGS_G_TOL,
        f_abstol=1e-10,
        x_abstol=1e-10,
        allow_f_increases=false,
    )

    println("\n" * "="^70)
    println("Optimization: $label")
    println("="^70)

    result = optimize(cost_function, grad!, initial_guess, LBFGS(; m=LBFGS_M), opts)

    z_opt = Optim.minimizer(result)
    wcoeffs_opt = to_w.(z_opt)
    final_cost = raw_cost(wcoeffs_opt)
    grad_final = ForwardDiff.gradient(cost_function, z_opt)
    coeff_error = norm(wcoeffs_opt .- WCOEFFS_TRUE)

    println("\n=== Optimization complete: $label ===")
    println("True coeffs      = $(round.(WCOEFFS_TRUE; digits=6))")
    println("Recovered coeffs = $(round.(wcoeffs_opt; digits=6))")
    println("Coeff error norm = $(round(coeff_error; digits=10))")
    println("Final raw cost   = $(round(final_cost; digits=12))")
    println("Final ‖∇z J‖     = $(round(norm(grad_final); digits=12))")

    return (; result, z_opt, wcoeffs_opt, final_cost, grad_final, coeff_error)
end

# -----------------------------------------------------------------------------
# Diagnostics and plotting helpers
# -----------------------------------------------------------------------------

function initial_w_field(grid, wcoeffs)
    xy_int = SinFVM.cell_centers(grid; interior=true)
    nx_int, ny_int = SinFVM.interior_size(grid)

    vals = [
        Float64(w0_from_coeffs(xy_int[I], Float64.(wcoeffs); x_dam=X_DAM, y_mid=Y_MID, dry_eps=DEPTH_CUT))
        for I in eachindex(xy_int)
    ]

    return reshape(vals, nx_int, ny_int)
end

function initial_eta_field(grid)
    xy_int = SinFVM.cell_centers(grid; interior=true)
    nx_int, ny_int = SinFVM.interior_size(grid)

    vals = [Float64(eta0_profile(xy_int[I], Float64)) for I in eachindex(xy_int)]
    return reshape(vals, nx_int, ny_int)
end

function snapshot(wcoeffs; label="")
    sim, eq, grid = setup_twolayer_simulator_2d(
        backend=SinFVM.make_cpu_backend(),
        wcoeffs=Float64.(wcoeffs),
    )

    η0_init = initial_eta_field(grid)
    w0_init = initial_w_field(grid, wcoeffs)

    SinFVM.simulate_to_time(sim, T_END)

    obs = observable_fields(sim, eq, grid)
    rec = reconstruct_from_observables(obs.η, obs.u1, obs.v1, obs.u2, obs.v2, obs.w, obs.Bcell)

    return (;
        η0_init = Array(η0_init),
        w0_init = Array(w0_init),
        η = Array(obs.η),
        w = Array(obs.w),
        B = Array(obs.Bcell),
        u1 = Array(obs.u1),
        v1 = Array(obs.v1),
        u2 = Array(obs.u2),
        v2 = Array(obs.v2),
        h1 = Array(rec.h1),
        h2 = Array(rec.h2),
        label,
    )
end

function cost_trace_values(result)
    return try
        [t.value for t in Optim.trace(result) if hasproperty(t, :value)]
    catch
        Float64[]
    end
end

# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

initial_guess_coeffs = fill(W0_INIT_CONST, 4)

# Baseline with original weights.
reset_to_original_weights_2d!()
baseline = run_optimization("original weights"; initial_guess_coeffs=initial_guess_coeffs, iterations=LBFGS_MAX_ITERS)

# Sensitivity diagnostics and adaptive weights.
sens_file = joinpath(pwd(), "sensitivity_diagnostics_2d.png")
plot_sensitivity_diagnostics_2d(baseline.wcoeffs_opt, sens_file)

Wη_ad, Wu1_ad, Wv1_ad, Wu2_ad, Wv2_ad, obs_sens =
    adaptive_weights_from_observable_sensitivities_2d(baseline.wcoeffs_opt)

println("\nObservable sensitivities at baseline solution:")
println("  η  = $(@sprintf "%.6e" obs_sens.sens_η)")
println("  u1 = $(@sprintf "%.6e" obs_sens.sens_u1)")
println("  v1 = $(@sprintf "%.6e" obs_sens.sens_v1)")
println("  u2 = $(@sprintf "%.6e" obs_sens.sens_u2)")
println("  v2 = $(@sprintf "%.6e" obs_sens.sens_v2)")

println("\nAdaptive weights:")
println("  W_ETA = $(@sprintf "%.6f" Wη_ad)")
println("  W_U1  = $(@sprintf "%.6f" Wu1_ad)")
println("  W_V1  = $(@sprintf "%.6f" Wv1_ad)")
println("  W_U2  = $(@sprintf "%.6f" Wu2_ad)")
println("  W_V2  = $(@sprintf "%.6f" Wv2_ad)")

# Adaptive run from the same constant interface initial guess.
set_adaptive_weights_2d!(Wη_ad, Wu1_ad, Wv1_ad, Wu2_ad, Wv2_ad; W_reg=W_REG_ORIG)
adaptive = run_optimization("adaptive observable weights"; initial_guess_coeffs=initial_guess_coeffs, iterations=LBFGS_MAX_ITERS)

# Comparison.
println("\n" * "="^70)
println("Comparison")
println("="^70)
println("Baseline coeff error = $(round(baseline.coeff_error; digits=10))")
println("Adaptive coeff error = $(round(adaptive.coeff_error; digits=10))")
println("Baseline final cost  = $(round(baseline.final_cost; digits=12))")
println("Adaptive final cost  = $(round(adaptive.final_cost; digits=12))")

# Use adaptive result as final.
wcoeffs_opt = adaptive.wcoeffs_opt

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

reset_to_original_weights_2d!()

snap_true = snapshot(WCOEFFS_TRUE; label="truth")
snap_init = snapshot(initial_guess_coeffs; label="constant initial guess")
snap_base = snapshot(baseline.wcoeffs_opt; label="optimized original weights")
snap_adap = snapshot(adaptive.wcoeffs_opt; label="optimized adaptive weights")

cost_vals_baseline = cost_trace_values(baseline.result)
cost_vals_adaptive = cost_trace_values(adaptive.result)

fig = Figure(size=(1600, 1400), fontsize=18)

ax_conv = Axis(fig[1, 1:2], title="2D twin-experiment convergence", xlabel="iteration", ylabel="objective", yscale=log10)
if !isempty(cost_vals_baseline)
    lines!(ax_conv, 1:length(cost_vals_baseline), cost_vals_baseline, linewidth=2, label="original")
    scatter!(ax_conv, 1:length(cost_vals_baseline), cost_vals_baseline, markersize=6)
end
if !isempty(cost_vals_adaptive)
    lines!(ax_conv, 1:length(cost_vals_adaptive), cost_vals_adaptive, linewidth=2, label="adaptive")
    scatter!(ax_conv, 1:length(cost_vals_adaptive), cost_vals_adaptive, markersize=6)
end
axislegend(ax_conv, position=:rt)

snaps = [snap_true, snap_init, snap_base, snap_adap]

for (row, sn) in enumerate(snaps)
    ax1 = Axis(fig[row + 1, 1], title="$(sn.label): initial w₀(x,y)")
    ax2 = Axis(fig[row + 1, 2], title="$(sn.label): final η(x,y) at T=$(T_END)")
    heatmap!(ax1, sn.w0_init)
    heatmap!(ax2, sn.η)
end

display(fig)

println("\n=== Final selected result ===")
println("True coeffs      = $(round.(WCOEFFS_TRUE; digits=6))")
println("Recovered coeffs = $(round.(wcoeffs_opt; digits=6))")
println("Coeff error norm = $(round(norm(wcoeffs_opt .- WCOEFFS_TRUE); digits=10))")
println("Sensitivity diagnostics saved to: $sens_file")
