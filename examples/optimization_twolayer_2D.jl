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

using SinFVM, StaticArrays, ForwardDiff, Optim, Parameters, CairoMakie, LinearAlgebra

# -----------------------------------------------------------------------------
# Simple 2D twin experiment:
# recover a spatially varying initial interface w0(x,y)
#
# To keep it simple, w0(x,y) is parameterized by 4 values:
# a 2x2 block field over the wet half x < x_dam.
#
# Observables used in the cost:
#   η, u1, v1, u2, v2
#
# Internal state:
#   U = (h1, q1, p1, w, q2, p2),  where w = h2 + B
# -----------------------------------------------------------------------------

const CFL_2D = 0.2
const W_MIN, W_MAX = 0.5, 3.0

σ(z) = inv(one(z) + exp(-z))
to_w(z) = W_MIN + (W_MAX - W_MIN) * σ(z)
from_w(w) = log((w - W_MIN) / (W_MAX - w))

# -----------------------------------------------------------------------------
# Bathymetry builders
# -----------------------------------------------------------------------------

function make_bottom_cos_sin_2d(; B0=-3.0, Ax=0.4, Ay=0.3, mx=1, my=1, φx=0.0, φy=0.0, backend, grid::SinFVM.CartesianGrid{2})
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
        xhat -= floor(xhat); yhat -= floor(yhat)
        B[i, j] = B0 + Ax * cos(2π * mx * xhat + φx) + Ay * sin(2π * my * yhat + φy)
    end
    SinFVM.BottomTopography2D(B, backend, grid)
end

# -----------------------------------------------------------------------------
# Simple 2x2 block parameterization of w0(x,y) on wet half
# coeffs = [bottom-left, top-left, bottom-right, top-right] within x < x_dam
# -----------------------------------------------------------------------------

function w0_from_coeffs(xy, coeffs; x_dam=50.0, y_mid=25.0, dry_eps=1e-4)
    x, y = xy
    x >= x_dam && return eltype(coeffs)(dry_eps)
    left_right = x < x_dam / 2 ? 0 : 1
    bot_top = y < y_mid ? 0 : 1
    idx = 1 + left_right + 2 * bot_top
    coeffs[idx]
end

# -----------------------------------------------------------------------------
# Simulator factory
# -----------------------------------------------------------------------------

function setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wcoeffs, h10::T) where {T}
    nx, ny, gc = 64, 64, 2
    grid = SinFVM.CartesianGrid(nx, ny; gc=gc, boundary=SinFVM.PeriodicBC(), extent=[0.0 100.0; 0.0 50.0])
    bottom = make_bottom_cos_sin_2d(; B0=-3.0, Ax=0.4, Ay=0.3, mx=1, my=1, backend=backend, grid=grid)
    eq = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=T(1.00), ρ2=T(1.02), g=T(9.81))
    rec = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)
    cs = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=CFL_2D)

    xy_int = SinFVM.cell_centers(grid; interior=true)
    B_int = SinFVM.collect_topography_cells(eq.B, grid; interior=true)

    ε_h = T(1e-4)
    initial = [begin
        xy = xy_int[I]
        w0 = T(w0_from_coeffs(xy, wcoeffs; x_dam=50.0, y_mid=25.0, dry_eps=ε_h))
        if xy[1] < 50.0
            h2 = w0 - B_int[I]
            h2 <= 0 && error("Initial w0 makes h2 <= 0 at xy=$xy")
            @SVector [h10, zero(T), zero(T), w0, zero(T), zero(T)]
        else
            wdry = max(T(B_int[I] + ε_h), ε_h)
            @SVector [ε_h, zero(T), zero(T), wdry, zero(T), zero(T)]
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

    (; Bcell, h1, q1, p1, w, q2, p2, h2, η, u1, v1, u2, v2)
end

reconstruct_from_observables(η, u1, v1, u2, v2, w, B) = (
    h1 = η .- w,
    h2 = w .- B,
    q1 = (η .- w) .* u1,
    p1 = (η .- w) .* v1,
    q2 = (w .- B) .* u2,
    p2 = (w .- B) .* v2,
)

# -----------------------------------------------------------------------------
# Observation callback
# cell_indices are tuples like (i,j)
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
            push!(cb.data, obs.u1[i, j]); push!(cb.data, obs.v1[i, j])
            push!(cb.data, obs.u2[i, j]); push!(cb.data, obs.v2[i, j])
        end
        cb.next_obs += 1
    end
end

# -----------------------------------------------------------------------------
# Forward solve
# -----------------------------------------------------------------------------

function simulate_observations(; T, wcoeffs, h10, obs_times, cell_indices)
    ADType = promote_type(eltype(wcoeffs), typeof(h10))
    sim, _, _ = setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(ADType), wcoeffs=ADType.(wcoeffs), h10=ADType(h10))
    recorder = ObservableRecorder(obs_times=obs_times, cell_indices=cell_indices, data=ADType[])
    SinFVM.simulate_to_time(sim, T; callback=recorder)
    @assert recorder.next_obs == length(obs_times) + 1 "Not all observation times were recorded"
    recorder.data
end

# -----------------------------------------------------------------------------
# Twin experiment setup
# -----------------------------------------------------------------------------

const T_END = 6.0
const H10_FIXED = 1.0
const OBS_TIMES = [2.0, 4.0, 6.0]

# 4-parameter true initial field on wet half
const WCOEFFS_TRUE = [1.80, 2.10, 1.65, 1.95]

# a few observation cells
const CELL_INDICES = [(4, 4), (8, 8), (12, 12), (20, 20), (30, 30)]

const EXACT_OBS = simulate_observations(
    T=T_END, wcoeffs=WCOEFFS_TRUE, h10=H10_FIXED,
    obs_times=OBS_TIMES, cell_indices=CELL_INDICES
)

println("Generated synthetic exact observations:")
println("  true coeffs   = $(round.(WCOEFFS_TRUE; digits=4))")
println("  H10_FIXED     = $H10_FIXED")
println("  CFL_2D        = $CFL_2D")
println("  n_observables = $(length(EXACT_OBS))")

# -----------------------------------------------------------------------------
# Cost
# -----------------------------------------------------------------------------

const W_ETA, W_U1, W_U2, W_REG = 1.0, 0.5, 0.5, 1e-8

function raw_cost(wcoeffs)
    pred = simulate_observations(
        T=T_END, wcoeffs=wcoeffs, h10=H10_FIXED,
        obs_times=OBS_TIMES, cell_indices=CELL_INDICES
    )
    J = zero(eltype(pred))
    @inbounds for k in 1:5:length(pred)
        dη = pred[k]   - EXACT_OBS[k]
        du1 = pred[k+1] - EXACT_OBS[k+1]
        dv1 = pred[k+2] - EXACT_OBS[k+2]
        du2 = pred[k+3] - EXACT_OBS[k+3]
        dv2 = pred[k+4] - EXACT_OBS[k+4]
        J += 0.5 * (W_ETA*dη^2 + W_U1*(du1^2 + dv1^2) + W_U2*(du2^2 + dv2^2))
    end
    J + 0.5 * W_REG * sum(wcoeffs.^2)
end

function cost_function(zvec)
    wcoeffs = to_w.(zvec)
    J = raw_cost(wcoeffs)
    J isa ForwardDiff.Dual && @assert all(.!isnan.(J.partials)) "NaN in gradient"
    J
end

grad!(storage, zvec) = ForwardDiff.gradient!(storage, cost_function, zvec)

# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

initial_guess_coeffs = [1.0, 1.0, 1.0, 1.0]
initial_guess = from_w.(initial_guess_coeffs)

opts = Optim.Options(
    store_trace=true, show_trace=true, show_every=1,
    iterations=20, g_tol=1e-8, f_abstol=1e-10, x_abstol=1e-10,
    allow_f_increases=false,
)

result = optimize(cost_function, grad!, initial_guess, LBFGS(; m=5), opts)

z_opt = Optim.minimizer(result)
wcoeffs_opt = to_w.(z_opt)
final_cost = raw_cost(wcoeffs_opt)

println("\n=== Optimization complete ===")
println("True coeffs      = $(round.(WCOEFFS_TRUE; digits=6))")
println("Recovered coeffs = $(round.(wcoeffs_opt; digits=6))")
println("Coeff error norm = $(round(norm(wcoeffs_opt .- WCOEFFS_TRUE); digits=10))")
println("Final raw cost   = $(round(final_cost; digits=12))")

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

function initial_w_field(grid, wcoeffs)
    xy_int = SinFVM.cell_centers(grid; interior=true)
    nx_int, ny_int = SinFVM.interior_size(grid)
    vals = [Float64(w0_from_coeffs(xy_int[I], wcoeffs; x_dam=50.0, y_mid=25.0, dry_eps=1e-4)) for I in eachindex(xy_int)]
    reshape(vals, nx_int, ny_int)
end

function snapshot(wcoeffs; h10=H10_FIXED, label="")
    sim, eq, grid = setup_twolayer_simulator_2d(; backend=SinFVM.make_cpu_backend(), wcoeffs=Float64.(wcoeffs), h10=Float64(h10))
    SinFVM.simulate_to_time(sim, T_END)
    obs = observable_fields(sim, eq, grid)
    rec = reconstruct_from_observables(obs.η, obs.u1, obs.v1, obs.u2, obs.v2, obs.w, obs.Bcell)
    w0_init = initial_w_field(grid, wcoeffs)
    (; w0_init=Array(w0_init), η=Array(obs.η), w=Array(obs.w), B=Array(obs.Bcell), u2=Array(obs.u2), v2=Array(obs.v2), h1=Array(rec.h1), h2=Array(rec.h2), label)
end

snap_true = snapshot(WCOEFFS_TRUE; label="truth")
snap_init = snapshot(initial_guess_coeffs; label="initial guess")
snap_opt  = snapshot(wcoeffs_opt; label="optimized")

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

cost_vals = try [t.value for t in Optim.trace(result) if hasproperty(t, :value)] catch; Float64[] end

fig = Figure(size=(1500, 1100), fontsize=18)

if !isempty(cost_vals)
    ax_conv = Axis(fig[1, 1:2], title="2D twin-experiment convergence", xlabel="iteration", ylabel="objective")
    lines!(ax_conv, 1:length(cost_vals), cost_vals, linewidth=2)
    scatter!(ax_conv, 1:length(cost_vals), cost_vals, markersize=8)
else
    Label(fig[1, 1:2], "2D twin experiment", fontsize=20)
end

for (row, sn) in enumerate([snap_true, snap_init, snap_opt])
    ax1 = Axis(fig[row+1, 1], title="$(sn.label): initial w₀(x,y)")
    ax2 = Axis(fig[row+1, 2], title="$(sn.label): final η(x,y) at T=$(T_END)")
    heatmap!(ax1, sn.w0_init)
    heatmap!(ax2, sn.η)
end

display(fig)