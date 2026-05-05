using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie
using LinearAlgebra, JLD2, Printf, Statistics

const DESING_KAPPA = 1e-4
const EPS_CUT = 1e-4

const NX = 64
const XMIN, XMAX = 0.0, 100.0
const T_END = 20.0
const OBS_TIMES = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0]
const CELL_INDICES = collect(1:NX)

const H1_CONST_ABOVE_INTERFACE = 0.75
const W0_LEFT_TRUE, W0_RIGHT_TRUE = 1.85, 0.10
const X_DAM = 50.0
const TRANSITION_WIDTH_W_TRUE = 10.0
const W0_INIT_CONST = 0.5

# ---------------------------------------------------------------------------
# Bathymetry setup
# ---------------------------------------------------------------------------

const B_AMP = 1.0
const B_CENTER = 50
const B_WIDTH = 10
const B_BACKGROUND = -1.0

# ---------------------------------------------------------------------------
# Optimization tuning
# ---------------------------------------------------------------------------

const W_EPS, W_U1, W_U2 = 1.0, 100.0, 100.0
const W_REG_H1 = 0.0001
const OBJ_SCALE = 1.0

const LBFGS_M = 10
const LBFGS_MAX_ITERS = 200
const LBFGS_G_SWITCH = 1e-2

const GN_MAX_ITERS = 50
const GN_G_FINAL = 1e-7
const GN_DAMPING0 = 1e-8
const GN_ARMIJO_C1 = 1e-6
const GN_BACKTRACK = 0.8
const GN_MIN_STEP = 1e-2

const SAVE_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Optimization"
mkpath(SAVE_DIR)

const RESULTS_DIR = joinpath(SAVE_DIR, "results")
mkpath(RESULTS_DIR)

# ---------------------------------------------------------------------------
# Boundary condition settings
# ---------------------------------------------------------------------------

const BOUNDARY_CONDITION_TYPE = "WallBC"  # "WallBC", "OutflowBC", or "PeriodicBC"
const USE_ALTERNATIVE_BC = false

function make_bc()
    if USE_ALTERNATIVE_BC && BOUNDARY_CONDITION_TYPE == "OutflowBC"
        return SinFVM.OutflowBC()
    elseif USE_ALTERNATIVE_BC && BOUNDARY_CONDITION_TYPE == "PeriodicBC"
        return SinFVM.PeriodicBC()
    else
        return SinFVM.WallBC()
    end
end

# ---------------------------------------------------------------------------
# Grid and profiles
# ---------------------------------------------------------------------------

function make_reference_grid()
    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=make_bc(), extent=[XMIN XMAX])
    x = collect(SinFVM.cell_centers(grid))
    return x, x[2] - x[1]
end

const X_GRID, DX = make_reference_grid()

function smooth_step_profile(x; left, right, center, width)
    T = promote_type(eltype(x), typeof(left), typeof(right), typeof(center), typeof(width))
    out = Vector{T}(undef, length(x))

    @inbounds for i in eachindex(x)
        s = (one(T) + tanh((T(center) - T(x[i])) / T(width))) / 2
        out[i] = T(right) + (T(left) - T(right)) * s
    end

    return out
end

function smooth_bathymetry_profile(x; amp, center, width, background=0.0)
    T = promote_type(eltype(x), typeof(amp), typeof(center), typeof(width), typeof(background))
    out = Vector{T}(undef, length(x))

    @inbounds for i in eachindex(x)
        ξ = (T(x[i]) - T(center)) / T(width)
        out[i] = T(background) + T(amp) * exp(-ξ^2)
    end

    return out
end

const B_PROFILE = smooth_bathymetry_profile(
    X_GRID;
    amp=B_AMP,
    center=B_CENTER,
    width=B_WIDTH,
    background=B_BACKGROUND,
)

const W0_TRUE_PROFILE = smooth_step_profile(
    X_GRID;
    left=W0_LEFT_TRUE,
    right=W0_RIGHT_TRUE,
    center=X_DAM,
    width=TRANSITION_WIDTH_W_TRUE,
)

# ε is the free surface. Since w is the interface elevation and B is bottom,
# h2 = w - B and h1 = ε - w.
const EPS_TRUE_PROFILE = W0_TRUE_PROFILE .+ H1_CONST_ABOVE_INTERFACE

const LOWER_W0_PROFILE = B_PROFILE .+ EPS_CUT
const UPPER_W0_PROFILE = EPS_TRUE_PROFILE .- EPS_CUT

const W0_INIT_PROFILE = clamp.(
    fill(W0_INIT_CONST, NX),
    LOWER_W0_PROFILE,
    UPPER_W0_PROFILE,
)

const B_CELL_CENTERS = smooth_bathymetry_profile(
    X_GRID;
    amp=B_AMP,
    center=B_CENTER,
    width=B_WIDTH,
    background=B_BACKGROUND,
)

project_w0(w0) = clamp.(w0, LOWER_W0_PROFILE, UPPER_W0_PROFILE)

# ---------------------------------------------------------------------------
# Bathymetry object helper
# ---------------------------------------------------------------------------

function make_bathymetry(backend, grid, ::Type{T}) where {T}
    B_cells = T.(B_CELL_CENTERS)
    x_faces = SinFVM.cell_faces(grid; interior=false)
    B_face = similar(x_faces, T)

    gc = grid.ghostcells[1]
    n_interior = NX

    B_padded = similar(B_face)
    B_padded[1:gc] .= B_cells[1]
    B_padded[gc+1:gc+n_interior] = B_cells[:]
    B_padded[gc+n_interior+1:end] .= B_cells[end]

    for i in eachindex(B_face)
        if i < length(B_face)
            B_face[i] = T(0.5) * (B_padded[i] + B_padded[i+1])
        else
            B_face[i] = B_padded[i]
        end
    end

    return SinFVM.BottomTopography1D(B_face, backend, grid)
end

function compute_bathymetry_values(grid, ::Type{T}) where {T}
    return T.(B_CELL_CENTERS)
end

bathymetry_cell_values(simulator, ::Type{T}) where {T} = compute_bathymetry_values(simulator.grid, T)

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

function setup_twolayer_simulator(; backend=SinFVM.make_cpu_backend(), ε_profile, w0_profile)
    TT = promote_type(eltype(ε_profile), eltype(w0_profile))

    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=make_bc(), extent=[XMIN XMAX])
    B = make_bathymetry(backend, grid, TT)

    eq = SinFVM.TwoLayerShallowWaterEquations1D(
        B;
        ρ1=TT(0.98),
        ρ2=TT(1.00),
        g=TT(9.81),
        depth_cutoff=TT(EPS_CUT),
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

    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=0.4)

    εv = TT.(ε_profile)
    wv = TT.(w0_profile)
    Bv = compute_bathymetry_values(grid, TT)
    ε_cut = TT(EPS_CUT)

    initial = map(1:NX) do i
        w = max(wv[i], Bv[i] + ε_cut)
        h1 = max(εv[i] - w, ε_cut)
        @SVector([h1, zero(TT), w, zero(TT)])
    end

    SinFVM.set_current_state!(sim, initial)
    return sim
end

smooth_positive(h, κ) = 0.5 * (h + sqrt(h^2 + κ^2))
smooth_velocity(h, q, κ) = q / smooth_positive(h, κ)

function observable_fields(simulator)
    st = SinFVM.current_interior_state(simulator)
    T = eltype(st.h1)
    κ = T(DESING_KAPPA)
    B = bathymetry_cell_values(simulator, T)

    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2
    h2 = w .- B
    ε = h1 .+ w

    u1 = smooth_velocity.(h1, q1, κ)
    u2 = smooth_velocity.(h2, q2, κ)

    return (; ε, u1, u2, w, Bvals=B)
end

reconstruct_from_observables(ε, u1, u2, w, B) = (;
    h1 = ε .- w,
    h2 = w .- B,
    q1 = (ε .- w) .* u1,
    q2 = (w .- B) .* u2,
)

# ---------------------------------------------------------------------------
# Observation machinery
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

function simulate_observations(; t_end, w0_profile, ε_profile, obs_times, cell_indices)
    ADType = promote_type(eltype(w0_profile), eltype(ε_profile))

    sim = setup_twolayer_simulator(
        backend=SinFVM.make_cpu_backend(ADType),
        ε_profile=ADType.(ε_profile),
        w0_profile=ADType.(w0_profile),
    )

    recorder = ObservableRecorder(
        obs_times=obs_times,
        cell_indices=cell_indices,
        data=ADType[],
    )

    SinFVM.simulate_to_time(sim, t_end; callback=recorder)

    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"

    return recorder.data
end

# ---------------------------------------------------------------------------
# Synthetic observations
# ---------------------------------------------------------------------------

const EXACT_OBS = simulate_observations(
    t_end=T_END,
    w0_profile=W0_TRUE_PROFILE,
    ε_profile=EPS_TRUE_PROFILE,
    obs_times=OBS_TIMES,
    cell_indices=CELL_INDICES,
)

println("Generated synthetic exact observations:")
println("  n_cells       = $NX")
println("  n_obs_times   = $(length(OBS_TIMES))")
println("  n_observables = $(length(EXACT_OBS))")
println("  LBFGS m       = $LBFGS_M")
println("  H1 reg weight = $W_REG_H1")
println("  bathymetry    = Gaussian bump, amp=$B_AMP, center=$B_CENTER, width=$B_WIDTH")

# ---------------------------------------------------------------------------
# Residual and cost
# ---------------------------------------------------------------------------

const N_TRIPLES = length(EXACT_OBS) ÷ 3
const MISFIT_SCALE_EPS_ORIG = sqrt(W_EPS / N_TRIPLES)
const MISFIT_SCALE_U1_ORIG  = sqrt(W_U1  / N_TRIPLES)
const MISFIT_SCALE_U2_ORIG  = sqrt(W_U2  / N_TRIPLES)
const REG_SCALE = sqrt(W_REG_H1 / DX)

const ADAPTIVE_WEIGHTS_CONFIG = Dict{Symbol,Any}(
    :use_adaptive => false,
    :use_boundary_taper => false,
    :w_eps => MISFIT_SCALE_EPS_ORIG,
    :w_u1 => MISFIT_SCALE_U1_ORIG,
    :w_u2 => MISFIT_SCALE_U2_ORIG,
    :boundary_taper => ones(Float64, NX),
)

function compute_boundary_taper(nx; taper_fraction=0.15)
    n_taper = max(1, Int(ceil(taper_fraction * nx)))
    taper = ones(Float64, nx)

    for i in 1:n_taper
        taper[i] = (i - 1) / n_taper
    end

    for i in 1:n_taper
        taper[end - i + 1] = (i - 1) / n_taper
    end

    return taper
end

function set_adaptive_weights!(w_eps, w_u1, w_u2; use_taper=false)
    ADAPTIVE_WEIGHTS_CONFIG[:w_eps] = sqrt(w_eps / N_TRIPLES)
    ADAPTIVE_WEIGHTS_CONFIG[:w_u1] = sqrt(w_u1 / N_TRIPLES)
    ADAPTIVE_WEIGHTS_CONFIG[:w_u2] = sqrt(w_u2 / N_TRIPLES)
    ADAPTIVE_WEIGHTS_CONFIG[:use_adaptive] = true

    if use_taper
        ADAPTIVE_WEIGHTS_CONFIG[:boundary_taper] = compute_boundary_taper(NX; taper_fraction=0.15)
        ADAPTIVE_WEIGHTS_CONFIG[:use_boundary_taper] = true
    else
        ADAPTIVE_WEIGHTS_CONFIG[:boundary_taper] = ones(Float64, NX)
        ADAPTIVE_WEIGHTS_CONFIG[:use_boundary_taper] = false
    end

    return nothing
end

function reset_to_original_weights!()
    ADAPTIVE_WEIGHTS_CONFIG[:use_adaptive] = false
    ADAPTIVE_WEIGHTS_CONFIG[:use_boundary_taper] = false
    ADAPTIVE_WEIGHTS_CONFIG[:w_eps] = MISFIT_SCALE_EPS_ORIG
    ADAPTIVE_WEIGHTS_CONFIG[:w_u1] = MISFIT_SCALE_U1_ORIG
    ADAPTIVE_WEIGHTS_CONFIG[:w_u2] = MISFIT_SCALE_U2_ORIG
    ADAPTIVE_WEIGHTS_CONFIG[:boundary_taper] = ones(Float64, NX)
    return nothing
end

function get_current_scales()
    if ADAPTIVE_WEIGHTS_CONFIG[:use_adaptive]
        return (
            ADAPTIVE_WEIGHTS_CONFIG[:w_eps],
            ADAPTIVE_WEIGHTS_CONFIG[:w_u1],
            ADAPTIVE_WEIGHTS_CONFIG[:w_u2],
        )
    else
        return (MISFIT_SCALE_EPS_ORIG, MISFIT_SCALE_U1_ORIG, MISFIT_SCALE_U2_ORIG)
    end
end

function get_boundary_taper()
    if ADAPTIVE_WEIGHTS_CONFIG[:use_boundary_taper]
        return ADAPTIVE_WEIGHTS_CONFIG[:boundary_taper]
    else
        return ones(Float64, NX)
    end
end

function residual_vector(w0_vec)
    # Fminbox handles optimization bounds, but this projection also protects manual calls
    # and the Gauss-Newton phase. If many parameters are exactly at bounds, this can flatten
    # gradients locally; remove this line for pure Fminbox-only testing if desired.
    w0_profile = project_w0(w0_vec)

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

    scale_eps, scale_u1, scale_u2 = get_current_scales()
    boundary_taper = get_boundary_taper()

    n_cells_obs = length(CELL_INDICES)

    @inbounds for k in 1:3:nmis
        # obs_idx goes 1:(length(OBS_TIMES) * length(CELL_INDICES)).
        # Map it back to the cell index because boundary_taper only has NX entries.
        obs_idx = div(k - 1, 3) + 1
        local_cell_pos = mod1(obs_idx, n_cells_obs)
        cell_idx = CELL_INDICES[local_cell_pos]
        taper_weight = T(boundary_taper[cell_idx])

        r[k]   = taper_weight * T(scale_eps) * (pred[k]   - EXACT_OBS[k])
        r[k+1] = taper_weight * T(scale_u1)  * (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = taper_weight * T(scale_u2)  * (pred[k+2] - EXACT_OBS[k+2])
    end

    @inbounds for i in 1:nreg
        r[nmis + i] = T(REG_SCALE) * (w0_profile[i+1] - w0_profile[i])
    end

    return r
end

function cost_function(w0_vec)
    r = residual_vector(w0_vec)
    J = OBJ_SCALE * 0.5 * dot(r, r)

    if J isa ForwardDiff.Dual
        @assert all(.!isnan.(J.partials)) "NaN in gradient"
    end

    return J
end

grad!(storage, w0_vec) = ForwardDiff.gradient!(storage, cost_function, w0_vec)
raw_cost(w0_profile) = cost_function(w0_profile)

# ---------------------------------------------------------------------------
# Sensitivity-based adaptive weighting
# ---------------------------------------------------------------------------

function compute_jacobian_column_sensitivities(w0_ref)
    Jr = ForwardDiff.jacobian(residual_vector, w0_ref)
    col_norms = vec(sqrt.(sum(abs2, Jr; dims=1)))
    row_norms = vec(sqrt.(sum(abs2, Jr; dims=2)))

    return (;
        col_norms,
        row_norms,
        mean_col_norm=mean(col_norms),
    )
end

function component_cost_gradient_norms(w0_ref)
    pred0 = simulate_observations(
        t_end=T_END,
        w0_profile=w0_ref,
        ε_profile=EPS_TRUE_PROFILE,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    function component_residuals(w, component::Int)
        pred = simulate_observations(
            t_end=T_END,
            w0_profile=project_w0(w),
            ε_profile=EPS_TRUE_PROFILE,
            obs_times=OBS_TIMES,
            cell_indices=CELL_INDICES,
        )
        return pred[component:3:end] .- EXACT_OBS[component:3:end]
    end

    cost_eps = w -> 0.5 * dot(component_residuals(w, 1), component_residuals(w, 1))
    cost_u1  = w -> 0.5 * dot(component_residuals(w, 2), component_residuals(w, 2))
    cost_u2  = w -> 0.5 * dot(component_residuals(w, 3), component_residuals(w, 3))

    g_eps = ForwardDiff.gradient(cost_eps, w0_ref)
    g_u1  = ForwardDiff.gradient(cost_u1,  w0_ref)
    g_u2  = ForwardDiff.gradient(cost_u2,  w0_ref)

    return (;
        norm_g_eps = norm(g_eps),
        norm_g_u1 = norm(g_u1),
        norm_g_u2 = norm(g_u2),
    )
end

function observable_jacobian_sensitivities(w0_ref)
    obs_fun = w -> simulate_observations(
        t_end=T_END,
        w0_profile=project_w0(w),
        ε_profile=EPS_TRUE_PROFILE,
        obs_times=OBS_TIMES,
        cell_indices=CELL_INDICES,
    )

    Jobs = ForwardDiff.jacobian(obs_fun, w0_ref)

    eps_rows = 1:3:size(Jobs, 1)
    u1_rows  = 2:3:size(Jobs, 1)
    u2_rows  = 3:3:size(Jobs, 1)

    sens_eps = norm(Jobs[eps_rows, :]) / sqrt(length(eps_rows))
    sens_u1  = norm(Jobs[u1_rows,  :]) / sqrt(length(u1_rows))
    sens_u2  = norm(Jobs[u2_rows,  :]) / sqrt(length(u2_rows))

    return (; sens_eps, sens_u1, sens_u2)
end

function adaptive_weights_from_observable_sensitivities(w0_ref; min_clip=1e-2, max_clip=1e6)
    sens = observable_jacobian_sensitivities(w0_ref)
    vals = [sens.sens_eps, sens.sens_u1, sens.sens_u2]
    ref = maximum(vals)
    safe = max(1e-14 * ref, eps(Float64))

    # Because residuals are multiplied by sqrt(W), equalizing Jacobian magnitudes
    # requires W_i ≈ (ref / sens_i)^2.
    w_eps = (ref / max(sens.sens_eps, safe))^2
    w_u1  = (ref / max(sens.sens_u1,  safe))^2
    w_u2  = (ref / max(sens.sens_u2,  safe))^2

    return (
        clamp(w_eps, min_clip, max_clip),
        clamp(w_u1,  min_clip, max_clip),
        clamp(w_u2,  min_clip, max_clip),
        sens,
    )
end

function plot_sensitivity_diagnostics(w0_ref, filename)
    Jr = ForwardDiff.jacobian(residual_vector, w0_ref)
    col_norms = vec(sqrt.(sum(abs2, Jr; dims=1)))
    row_norms = vec(sqrt.(sum(abs2, Jr; dims=2)))

    nmis = length(EXACT_OBS)
    row_norms_eps = row_norms[1:3:nmis]
    row_norms_u1  = row_norms[2:3:nmis]
    row_norms_u2  = row_norms[3:3:nmis]

    obs_sens = observable_jacobian_sensitivities(w0_ref)
    grad_sens = component_cost_gradient_norms(w0_ref)

    fig = Figure(size=(1400, 1000), fontsize=18)

    ax1 = Axis(fig[1, 1], title="Parameter sensitivities: residual Jacobian column norms",
               xlabel="Parameter index", ylabel="Column norm")
    lines!(ax1, 1:length(col_norms), col_norms, linewidth=2)
    scatter!(ax1, 1:length(col_norms), col_norms, markersize=4)

    ax2 = Axis(fig[1, 2], title="Observation sensitivities by residual row",
               xlabel="Observation row index per type", ylabel="Row norm")
    lines!(ax2, 1:length(row_norms_eps), row_norms_eps, label="ε", linewidth=2)
    lines!(ax2, 1:length(row_norms_u1),  row_norms_u1,  label="u1", linewidth=2)
    lines!(ax2, 1:length(row_norms_u2),  row_norms_u2,  label="u2", linewidth=2)
    axislegend(ax2, position=:rb)

    ax3 = Axis(fig[2, 1], title="Observable Jacobian sensitivities",
               xlabel="Observable", ylabel="Sensitivity", yscale=log10)
    comps = ["ε", "u1", "u2"]
    vals = [obs_sens.sens_eps, obs_sens.sens_u1, obs_sens.sens_u2]
    x_pos = 1:3
    barplot!(ax3, x_pos, vals)
    ax3.xticks = (x_pos, comps)

    ax4 = Axis(fig[2, 2], title="Unweighted component cost gradient norms",
               xlabel="Observable", ylabel="Gradient norm", yscale=log10)
    gvals = [grad_sens.norm_g_eps, grad_sens.norm_g_u1, grad_sens.norm_g_u2]
    barplot!(ax4, x_pos, gvals)
    ax4.xticks = (x_pos, comps)

    save(filename, fig)
    println("Saved sensitivity diagnostics to: $filename")
    return fig
end

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@with_kw mutable struct OptimizationHistory
    iter::Vector{Int} = Int[]
    phase::Vector{String} = String[]
    J::Vector{Float64} = Float64[]
    gnorm::Vector{Float64} = Float64[]
    w0_profiles::Vector{Vector{Float64}} = Vector{Float64}[]
end

function push_history!(history, iter, phase, w0_vec, J, gnorm)
    wprof = project_w0(w0_vec)

    push!(history.iter, iter)
    push!(history.phase, phase)
    push!(history.J, Float64(J))
    push!(history.gnorm, Float64(gnorm))
    push!(history.w0_profiles, collect(Float64.(wprof)))

    return history
end

optim_iteration(state) =
    hasproperty(state, :iteration) ? Int(getproperty(state, :iteration)) :
    hasproperty(state, :pseudo_iteration) ? Int(getproperty(state, :pseudo_iteration)) :
    0

function optim_state_x(state)
    hasproperty(state, :x) || error("Could not find parameter vector in Optim state.")
    return getproperty(state, :x)
end

optim_state_value(state, w0) =
    hasproperty(state, :value) ? Float64(getproperty(state, :value)) :
    hasproperty(state, :f_x) ? Float64(getproperty(state, :f_x)) :
    Float64(cost_function(w0))

function make_history_callback(history, phase_name)
    last_iter = Ref(-1)

    return function cb(state)
        k = optim_iteration(state)

        if k == last_iter[]
            return false
        end

        last_iter[] = k

        w0 = project_w0(copy(optim_state_x(state)))
        J = optim_state_value(state, w0)
        g = ForwardDiff.gradient(cost_function, w0)

        push_history!(history, k, phase_name, w0, J, norm(g))

        return false
    end
end

# ---------------------------------------------------------------------------
# Gauss-Newton / Levenberg-Marquardt directly in w0
# ---------------------------------------------------------------------------

function gauss_newton_phase(
    w0_init;
    history,
    phase_name="GN",
    max_iters=GN_MAX_ITERS,
    damping0=GN_DAMPING0,
    c1=GN_ARMIJO_C1,
    backtrack=GN_BACKTRACK,
    min_step=GN_MIN_STEP,
    g_tol=GN_G_FINAL,
)
    w0 = project_w0(copy(w0_init))
    μ = damping0

    for k in 1:max_iters
        r = residual_vector(w0)
        J = 0.5 * dot(r, r)

        Jr = ForwardDiff.jacobian(residual_vector, w0)
        g = Jr' * r
        gnorm = norm(g)

        if gnorm < g_tol
            push_history!(history, history.iter[end] + 1, phase_name, w0, J, gnorm)
            println("$phase_name converged on gradient norm.")
            break
        end

        δ = -((Jr' * Jr + μ * I) \ g)

        α = 1.0
        accepted = false
        wtrial = w0
        Jtrial = J

        while α >= min_step
            wcand = project_w0(w0 .+ α .* δ)
            rcand = residual_vector(wcand)
            Jcand = 0.5 * dot(rcand, rcand)

            if Jcand <= J + c1 * α * dot(g, δ)
                wtrial = wcand
                Jtrial = Jcand
                accepted = true
                break
            end

            α *= backtrack
        end

        if !accepted
            println("$phase_name iter $k: line search failed, stopping.")
            break
        end

        step_avg = norm(wtrial .- w0) / sqrt(length(w0))

        w0 = wtrial
        push_history!(history, history.iter[end] + 1, phase_name, w0, Jtrial, gnorm)

        println(
            "$phase_name iter $k: J=$(round(Jtrial; digits=12)), " *
            "‖g‖=$(round(gnorm; digits=8)), " *
            "α=$(round(α; digits=6)), " *
            "avg_step_w0=$(round(step_avg; digits=8)), " *
            "μ=$(round(μ; digits=8))"
        )

        μ = Jtrial < J ? max(0.5μ, 1e-8) : min(10μ, 1e4)
    end

    return project_w0(w0)
end

# ---------------------------------------------------------------------------
# Plot helpers
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
    ε, u1, u2, w, B = collect(obs.ε), collect(obs.u1), collect(obs.u2), collect(obs.w), collect(obs.Bvals)
    rec = reconstruct_from_observables(ε, u1, u2, w, B)

    return (;
        x,
        ε,
        u1,
        u2,
        w,
        B,
        h1=collect(rec.h1),
        h2=collect(rec.h2),
        q1=collect(rec.q1),
        q2=collect(rec.q2),
        w0_profile=collect(w0_profile),
        label,
        t_end,
    )
end

function add_elevation!(ax, sn; legend_position=:lt)
    lines!(ax, sn.x, sn.ε, linewidth=2, label="ε")
    lines!(ax, sn.x, sn.w, linewidth=2, linestyle=:dash, label="w")
    lines!(ax, sn.x, sn.B, linewidth=2, label="B")
    axislegend(ax; position=legend_position)
end

function add_velocity!(ax, sn; legend_position=:lt)
    lines!(ax, sn.x, sn.u1, linewidth=2, label="u1")
    lines!(ax, sn.x, sn.u2, linewidth=2, label="u2")
    axislegend(ax; position=legend_position)
end

function animate_iteration_updates(history; filename, framerate=2)
    snaps = [
        snapshot(wprof; label="$(ph) iter $(k)", t_end=T_END)
        for (k, ph, wprof) in zip(history.iter, history.phase, history.w0_profiles)
    ]

    prof_obs = Observable(snaps[1].w0_profile)
    ε_obs = Observable(snaps[1].ε)
    w_obs = Observable(snaps[1].w)
    B_obs = Observable(snaps[1].B)
    u1_obs = Observable(snaps[1].u1)
    u2_obs = Observable(snaps[1].u2)

    iter_obs = Observable(history.iter[1])
    phase_obs = Observable(history.phase[1])
    J_obs = Observable(history.J[1])

    fig = Figure(size=(1500, 900), fontsize=22)
    Label(fig[1, 1:2], @lift("Phase: $phase_obs | iteration $iter_obs | J=$(round($J_obs; digits=10))"), fontsize=24)

    ax_prof = Axis(fig[2, 1:2], title="Recovered interface profile with bathymetry", xlabel="x", ylabel="elevation")
    lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true")
    lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial")
    lines!(ax_prof, X_GRID, prof_obs, linewidth=3, label="current")
    lines!(ax_prof, X_GRID, B_PROFILE, linewidth=2, label="bathymetry B")
    axislegend(ax_prof, position=:rb)

    ax_eps = Axis(fig[3, 1], title="Elevation fields", xlabel="x", ylabel="elevation")
    lines!(ax_eps, X_GRID, ε_obs, linewidth=2, label="ε")
    lines!(ax_eps, X_GRID, w_obs, linewidth=2, linestyle=:dash, label="w")
    lines!(ax_eps, X_GRID, B_obs, linewidth=2, label="B")
    axislegend(ax_eps, position=:lt)

    ax_u = Axis(fig[3, 2], title="Velocity fields", xlabel="x", ylabel="velocity")
    lines!(ax_u, X_GRID, u1_obs, linewidth=2, label="u1")
    lines!(ax_u, X_GRID, u2_obs, linewidth=2, label="u2")
    axislegend(ax_u, position=:lt)

    record(fig, filename, eachindex(snaps); framerate=framerate) do i
        sn = snaps[i]

        prof_obs[] = sn.w0_profile
        ε_obs[] = sn.ε
        w_obs[] = sn.w
        B_obs[] = sn.B
        u1_obs[] = sn.u1
        u2_obs[] = sn.u2

        iter_obs[] = history.iter[i]
        phase_obs[] = history.phase[i]
        J_obs[] = history.J[i]
    end

    return filename
end

# ---------------------------------------------------------------------------
# Optimization helper
# ---------------------------------------------------------------------------

function run_lbfgs_then_gn(w0_start; history, lbfgs_phase_name, gn_phase_name)
    g_start = ForwardDiff.gradient(cost_function, w0_start)
    push_history!(history, 0, "INIT", w0_start, cost_function(w0_start), norm(g_start))

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
        callback=make_history_callback(history, lbfgs_phase_name),
    )

    result_lbfgs = optimize(
        cost_function,
        grad!,
        LOWER_W0_PROFILE,
        UPPER_W0_PROFILE,
        w0_start,
        Fminbox(LBFGS(; m=LBFGS_M)),
        opts,
    )

    w0_lbfgs = project_w0(copy(Optim.minimizer(result_lbfgs)))
    J_lbfgs = cost_function(w0_lbfgs)
    g_lbfgs = ForwardDiff.gradient(cost_function, w0_lbfgs)

    println("\n=== After LBFGS-Fminbox phase ($lbfgs_phase_name) ===")
    println("Cost after LBFGS          = $(round(J_lbfgs; digits=12))")
    println("Gradient norm after LBFGS = $(round(norm(g_lbfgs); digits=12))")

    w0_opt = gauss_newton_phase(w0_lbfgs; history=history, phase_name=gn_phase_name)

    final_cost = cost_function(w0_opt)
    final_grad = ForwardDiff.gradient(cost_function, w0_opt)
    profile_error = norm(w0_opt .- W0_TRUE_PROFILE) / sqrt(NX)

    return (;
        result_lbfgs,
        w0_lbfgs,
        w0_opt,
        final_cost,
        final_grad,
        profile_error,
        history,
    )
end

# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

w0_start = project_w0(W0_INIT_PROFILE)

println("\n" * "="^70)
println("PHASE 1: ORIGINAL OPTIMIZATION (baseline)")
println("="^70)

reset_to_original_weights!()

history_original = OptimizationHistory()
original_run = run_lbfgs_then_gn(
    w0_start;
    history=history_original,
    lbfgs_phase_name="LBFGS-Fminbox-original",
    gn_phase_name="GN-original",
)

w0_opt_original = original_run.w0_opt
final_cost_original = original_run.final_cost
final_grad_original = original_run.final_grad
profile_error_original = original_run.profile_error

println("\n=== Original Optimization Complete ===")
println("Final raw cost   = $(round(final_cost_original; digits=12))")
println("Final grad norm  = $(round(norm(final_grad_original); digits=12))")
println("Profile L2 error = $(round(profile_error_original; digits=12))")

original_results = Dict(
    "w0_profile" => Float64.(w0_opt_original),
    "cost" => Float64(final_cost_original),
    "grad_norm" => Float64(norm(final_grad_original)),
    "profile_error" => Float64(profile_error_original),
    "history" => history_original,
    "weights" => Dict("W_EPS" => W_EPS, "W_U1" => W_U1, "W_U2" => W_U2),
)

println("\n" * "="^70)
println("SENSITIVITY ANALYSIS")
println("="^70)

reset_to_original_weights!()
fig_sens = plot_sensitivity_diagnostics(
    original_run.w0_lbfgs,
    joinpath(SAVE_DIR, "sensitivity_diagnostics_original.png"),
)

println("\nComputing adaptive weights from observable Jacobian sensitivities...")
w_eps_adaptive, w_u1_adaptive, w_u2_adaptive, obs_sens =
    adaptive_weights_from_observable_sensitivities(original_run.w0_lbfgs; min_clip=1e-2, max_clip=1e6)

println("Observable sensitivities:")
println("  sens_eps = $(@sprintf "%.6e" obs_sens.sens_eps)")
println("  sens_u1  = $(@sprintf "%.6e" obs_sens.sens_u1)")
println("  sens_u2  = $(@sprintf "%.6e" obs_sens.sens_u2)")

println("Original weights: W_EPS=$W_EPS, W_U1=$W_U1, W_U2=$W_U2")
println("Adaptive weights: W_EPS=$(@sprintf "%.6f" w_eps_adaptive), " *
        "W_U1=$(@sprintf "%.6f" w_u1_adaptive), " *
        "W_U2=$(@sprintf "%.6f" w_u2_adaptive)")
println("Weight ratios (adaptive/original):")
println("  W_EPS: $(@sprintf "%.4f" w_eps_adaptive/W_EPS)")
println("  W_U1:  $(@sprintf "%.4f" w_u1_adaptive/W_U1)")
println("  W_U2:  $(@sprintf "%.4f" w_u2_adaptive/W_U2)")

println("\n" * "="^70)
println("PHASE 2: ADAPTIVE WEIGHTED OPTIMIZATION")
println("="^70)

set_adaptive_weights!(w_eps_adaptive, w_u1_adaptive, w_u2_adaptive; use_taper=false)

history_adaptive = OptimizationHistory()
adaptive_run = run_lbfgs_then_gn(
    w0_start;
    history=history_adaptive,
    lbfgs_phase_name="LBFGS-Fminbox-adaptive",
    gn_phase_name="GN-adaptive",
)

w0_opt_adaptive = adaptive_run.w0_opt
final_cost_adaptive = adaptive_run.final_cost
final_grad_adaptive = adaptive_run.final_grad
profile_error_adaptive = adaptive_run.profile_error

println("\n=== Adaptive Optimization Complete ===")
println("Final raw cost   = $(round(final_cost_adaptive; digits=12))")
println("Final grad norm  = $(round(norm(final_grad_adaptive); digits=12))")
println("Profile L2 error = $(round(profile_error_adaptive; digits=12))")

adaptive_results = Dict(
    "w0_profile" => Float64.(w0_opt_adaptive),
    "cost" => Float64(final_cost_adaptive),
    "grad_norm" => Float64(norm(final_grad_adaptive)),
    "profile_error" => Float64(profile_error_adaptive),
    "history" => history_adaptive,
    "weights" => Dict("W_EPS" => w_eps_adaptive, "W_U1" => w_u1_adaptive, "W_U2" => w_u2_adaptive),
)

println("\n" * "="^70)
println("COMPARISON: Original vs Adaptive")
println("="^70)

cost_improvement = (final_cost_original - final_cost_adaptive) / max(abs(final_cost_original), eps(Float64)) * 100
grad_improvement = (norm(final_grad_original) - norm(final_grad_adaptive)) / max(norm(final_grad_original), eps(Float64)) * 100
error_improvement = (profile_error_original - profile_error_adaptive) / max(profile_error_original, eps(Float64)) * 100

println("Cost improvement:          $(@sprintf "%+.2f%%" cost_improvement)")
println("Gradient norm improvement: $(@sprintf "%+.2f%%" grad_improvement)")
println("Profile error improvement: $(@sprintf "%+.2f%%" error_improvement)")

jld2_file = joinpath(RESULTS_DIR, "optimization_results_comparison.jld2")
save(jld2_file, Dict(
    "original" => original_results,
    "adaptive" => adaptive_results,
    "comparison" => Dict(
        "cost_improvement_percent" => cost_improvement,
        "grad_improvement_percent" => grad_improvement,
        "error_improvement_percent" => error_improvement,
    ),
))
println("\nSaved results to: $jld2_file")

w0_opt_profile = w0_opt_adaptive

# ---------------------------------------------------------------------------
# Final plotting
# ---------------------------------------------------------------------------

reset_to_original_weights!()

snap_ic  = snapshot(W0_INIT_PROFILE; label="initial condition", t_end=0.0)
snap_syn = snapshot(W0_TRUE_PROFILE; label="synthetic", t_end=T_END)
snap_opt_original = snapshot(w0_opt_original; label="optimized (original weights)", t_end=T_END)
snap_opt_adaptive = snapshot(w0_opt_adaptive; label="optimized (adaptive weights)", t_end=T_END)

fig = Figure(size=(1800, 1500), fontsize=20)

title_text = "Optimization Results: Original vs Adaptive Weighting\n" *
             "Cost improvement: $(@sprintf "%.2f%%" cost_improvement) | " *
             "Grad improvement: $(@sprintf "%.2f%%" grad_improvement) | " *
             "Error improvement: $(@sprintf "%.2f%%" error_improvement)"
Label(fig[0, :], title_text, fontsize=22, font=:bold)

ax_prof = Axis(fig[1, 1:2], title="Recovered interface profiles with bathymetry", xlabel="x", ylabel="elevation")
lines!(ax_prof, X_GRID, W0_TRUE_PROFILE, linewidth=3, linestyle=:dash, label="true w₀", color=:black)
lines!(ax_prof, X_GRID, W0_INIT_PROFILE, linewidth=2, linestyle=:dot, label="initial guess")
lines!(ax_prof, X_GRID, w0_opt_original, linewidth=2.5, label="optimized (original)", color=:orange)
lines!(ax_prof, X_GRID, w0_opt_adaptive, linewidth=2.5, label="optimized (adaptive)", color=:green)
lines!(ax_prof, X_GRID, B_PROFILE, linewidth=2, label="bathymetry B", color=:brown)
axislegend(ax_prof, position=:rb, nbanks=2)

for (row, sn) in enumerate([snap_ic, snap_syn, snap_opt_original, snap_opt_adaptive])
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

    pos = sn.label == "initial condition" ? :rt : :lt
    add_elevation!(ax_eps, sn; legend_position=pos)
    add_velocity!(ax_u, sn; legend_position=pos)
end

display(fig)

fig_file = joinpath(SAVE_DIR, "optimization_profile_result_adaptive.png")
save(fig_file, fig)
println("Saved comparison figure to: $fig_file")

anim_file = animate_iteration_updates(
    history_adaptive;
    filename=joinpath(SAVE_DIR, "optimization_profile_iterations_adaptive.mp4"),
    framerate=2,
)
println("Saved adaptive animation to: $anim_file")

anim_file_original = animate_iteration_updates(
    history_original;
    filename=joinpath(SAVE_DIR, "optimization_profile_iterations_original.mp4"),
    framerate=2,
)
println("Saved original animation to: $anim_file_original")

println("\n" * "="^70)
println("OPTIMIZATION SUMMARY REPORT")
println("="^70)

summary_text = """
================================================================================
OPTIMIZATION WITH ADAPTIVE SENSITIVITY-BASED WEIGHTING
================================================================================

ORIGINAL OPTIMIZATION (Baseline):
  - Weights: W_EPS=$(W_EPS), W_U1=$(W_U1), W_U2=$(W_U2)
  - Final cost: $(round(final_cost_original; digits=12))
  - Final gradient norm: $(round(norm(final_grad_original); digits=12))
  - Profile L2 error: $(round(profile_error_original; digits=12))

ADAPTIVE OPTIMIZATION:
  - Observable sensitivities:
    * sens_eps: $(@sprintf "%.6e" obs_sens.sens_eps)
    * sens_u1:  $(@sprintf "%.6e" obs_sens.sens_u1)
    * sens_u2:  $(@sprintf "%.6e" obs_sens.sens_u2)
  - Weights: W_EPS=$(round(w_eps_adaptive; digits=6)), W_U1=$(round(w_u1_adaptive; digits=6)), W_U2=$(round(w_u2_adaptive; digits=6))
  - Weight ratios adaptive/original:
    * W_EPS: $(round(w_eps_adaptive/W_EPS; digits=4))
    * W_U1: $(round(w_u1_adaptive/W_U1; digits=4))
    * W_U2: $(round(w_u2_adaptive/W_U2; digits=4))
  - Final cost: $(round(final_cost_adaptive; digits=12))
  - Final gradient norm: $(round(norm(final_grad_adaptive); digits=12))
  - Profile L2 error: $(round(profile_error_adaptive; digits=12))

IMPROVEMENTS:
  - Cost improvement: $(@sprintf "%.2f%%" cost_improvement)
  - Gradient norm improvement: $(@sprintf "%.2f%%" grad_improvement)
  - Profile error improvement: $(@sprintf "%.2f%%" error_improvement)

FILES GENERATED:
  - Results comparison: $(jld2_file)
  - Sensitivity diagnostics: $(joinpath(SAVE_DIR, "sensitivity_diagnostics_original.png"))
  - Profile comparison plot: $(fig_file)
  - Adaptive optimization animation: $(anim_file)
  - Original optimization animation: $(anim_file_original)

NOTES:
  - Adaptive weights computed from observable Jacobian sensitivities.
  - Weights use inverse-squared sensitivity scaling because residuals use sqrt(W).
  - Weights are clipped to [1e-2, 1e6].
  - The boundary taper indexing bug is fixed using mod1 over CELL_INDICES.
  - WallBC remains the default. Test OutflowBC by setting USE_ALTERNATIVE_BC=true and BOUNDARY_CONDITION_TYPE="OutflowBC".
================================================================================
"""

println(summary_text)

summary_file = joinpath(RESULTS_DIR, "optimization_summary.txt")
open(summary_file, "w") do io
    write(io, summary_text)
end
println("Saved summary to: $(summary_file)")
