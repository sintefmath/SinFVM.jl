using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters
using LinearAlgebra

# ---------------------------------------------------------------------------
# Physical + numerical constants
# ---------------------------------------------------------------------------

const DESING_KAPPA = 1e-4
const EPS_CUT = 1e-4

const NX = 64
const XMIN, XMAX = 0.0, 100.0
const T_END = 20.0

const OBS_TIMES = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0]
const CELL_INDICES = collect(1:NX)

# ---------------------------------------------------------------------------
# 🔴 Increased density contrast (KEY CHANGE)
# ---------------------------------------------------------------------------

const RHO1 = 0.5      # was 0.98
const RHO2 = 1.5      # was 1.00

# ---------------------------------------------------------------------------
# True interface
# ---------------------------------------------------------------------------

const H1_CONST_ABOVE_INTERFACE = 0.75
const W0_LEFT_TRUE, W0_RIGHT_TRUE = 1.85, 0.10
const X_DAM = 50.0
const TRANSITION_WIDTH_W_TRUE = 10.0

const W0_INIT_CONST = 0.5

# ---------------------------------------------------------------------------
# Optimization tuning
# ---------------------------------------------------------------------------

const W_EPS, W_U1, W_U2 = 1.0, 1.0, 1.0
const W_REG_H1 = 1e-4

const LBFGS_M = 10
const LBFGS_MAX_ITERS = 800
const LBFGS_G_TOL = 1e-4

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

function make_reference_grid()
    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])
    x = collect(SinFVM.cell_centers(grid))
    return x, x[2] - x[1]
end

const X_GRID, DX = make_reference_grid()

# ---------------------------------------------------------------------------
# Smooth profile
# ---------------------------------------------------------------------------

function smooth_step_profile(x; left, right, center, width)
    out = similar(x)
    @inbounds for i in eachindex(x)
        s = (1 + tanh((center - x[i]) / width)) / 2
        out[i] = right + (left - right) * s
    end
    return out
end

const W0_TRUE_PROFILE = smooth_step_profile(
    X_GRID;
    left=W0_LEFT_TRUE,
    right=W0_RIGHT_TRUE,
    center=X_DAM,
    width=TRANSITION_WIDTH_W_TRUE,
)

const EPS_TRUE_PROFILE = W0_TRUE_PROFILE .+ H1_CONST_ABOVE_INTERFACE

const LOWER_W0_PROFILE = fill(EPS_CUT, NX)
const UPPER_W0_PROFILE = EPS_TRUE_PROFILE .- EPS_CUT

const W0_INIT_PROFILE = clamp.(fill(W0_INIT_CONST, NX), LOWER_W0_PROFILE, UPPER_W0_PROFILE)

project_w0(w0) = clamp.(w0, LOWER_W0_PROFILE, UPPER_W0_PROFILE)

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

function setup_sim(; ε_profile, w0_profile)
    TT = promote_type(eltype(ε_profile), eltype(w0_profile))

    grid = SinFVM.CartesianGrid(NX; gc=2, boundary=SinFVM.WallBC(), extent=[XMIN XMAX])
    B = SinFVM.ConstantBottomTopography(zero(TT))

    eq = SinFVM.TwoLayerShallowWaterEquations1D(
        B;
        ρ1=TT(RHO1),
        ρ2=TT(RHO2),
        g=TT(9.81),
        depth_cutoff=TT(EPS_CUT),
        desingularizing_kappa=TT(DESING_KAPPA),
    )

    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))
    flux = SinFVM.PathConservativeCentralUpwind(eq)

    cs = SinFVM.ConservedSystem(
        SinFVM.make_cpu_backend(TT),
        rec, flux, eq, grid,
        [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()],
    )

    sim = SinFVM.Simulator(SinFVM.make_cpu_backend(TT), cs, SinFVM.RungeKutta2(), grid; cfl=0.4)

   initial = map(1:NX) do i
        w = max(w0_profile[i], TT(EPS_CUT))
        h1 = max(ε_profile[i] - w, TT(EPS_CUT))
        @SVector([h1, zero(TT), w, zero(TT)])
    end

    SinFVM.set_current_state!(sim, initial)
    return sim
end

# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

function observable_fields(sim)
    st = SinFVM.current_interior_state(sim)
    T = eltype(st.h1)

    h1, q1, w, q2 = st.h1, st.q1, st.w, st.q2

    h2 = w
    ε = h1 .+ w

    u1 = q1 ./ (h1 .+ T(1e-8))
    u2 = q2 ./ (h2 .+ T(1e-8))

    return ε, u1, u2
end

function simulate_obs(w0)
    TT = eltype(w0)

    sim = setup_sim(
        ε_profile=TT.(EPS_TRUE_PROFILE),
        w0_profile=TT.(w0),
    )

    data = TT[]

    for t in OBS_TIMES
        SinFVM.simulate_to_time(sim, t)
        ε, u1, u2 = observable_fields(sim)

        for i in CELL_INDICES
            push!(data, ε[i])
            push!(data, u1[i])
            push!(data, u2[i])
        end
    end

    return data
end

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

const EXACT_OBS = simulate_obs(W0_TRUE_PROFILE)

# ---------------------------------------------------------------------------
# Residual + cost
# ---------------------------------------------------------------------------

const N_TRIPLES = length(EXACT_OBS) ÷ 3

function residual(w0)
    w0 = project_w0(w0)
    pred = simulate_obs(w0)

    r = similar(pred)

    for k in 1:3:length(pred)
        r[k]   = (pred[k]   - EXACT_OBS[k])
        r[k+1] = (pred[k+1] - EXACT_OBS[k+1])
        r[k+2] = (pred[k+2] - EXACT_OBS[k+2])
    end

    # regularization
    reg = sqrt(W_REG_H1 / DX) .* diff(w0)

    return vcat(r, reg)
end

function cost(w0)
    r = residual(w0)
    return 0.5 * dot(r, r)
end

grad!(g, w0) = ForwardDiff.gradient!(g, cost, w0)

# ---------------------------------------------------------------------------
# Optimization (Fminbox only)
# ---------------------------------------------------------------------------

w0_start = project_w0(W0_INIT_PROFILE)

result = optimize(
    cost,
    grad!,
    LOWER_W0_PROFILE,
    UPPER_W0_PROFILE,
    w0_start,
    Fminbox(LBFGS(m=LBFGS_M)),
    Optim.Options(
        iterations=LBFGS_MAX_ITERS,
        g_tol=LBFGS_G_TOL,
        show_trace=true,
    )
)

w0_opt = Optim.minimizer(result)

println("\nFinal cost = ", cost(w0_opt))
println("L2 error   = ", norm(w0_opt - W0_TRUE_PROFILE) / sqrt(NX))