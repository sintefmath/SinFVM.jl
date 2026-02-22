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

# ============================================================
# Path-Conservative Central-Upwind (PCCU) Runner
# Two-layer Shallow Water Equations in 1D (w-storage formulation)
#
# State: V = (h1, q1, w, q2)  with  w = h2 + B
# Physical: h2 = w - Bcell (cells) or w - Bface (faces)
#
# Numerical flux: PathConservativeCentralUpwind(equation)
# Sources: bottom + nonconservative (as in your framework)
# ============================================================

using CairoMakie
using StaticArrays
using SinFVM

# ----------------------------
# Helpers: bathymetry builders
# ----------------------------
xwrap(x) = x - floor(x)

make_bottom_constant(B0) = SinFVM.ConstantBottomTopography(B0)

function make_bottom_faces_1d(Bface_fun, backend, grid::SinFVM.CartesianGrid{1})
    xF = SinFVM.cell_faces(grid; interior=false)  # includes ghost faces
    Bint = similar(xF)
    @inbounds for i in eachindex(xF)
        Bint[i] = Bface_fun(xwrap(xF[i]))
    end
    return SinFVM.BottomTopography1D(Bint, backend, grid)
end

"Smooth periodic cosine bottom on faces"
bottom_cosine_faces(; B0=-2.0, A=0.4, m=1) = (x -> (B0 + A*cos(2π*m*xwrap(x))))

# ----------------------------
# Initial conditions (equilibrium-consistent)
# ----------------------------
"""
Uniform upper layer thickness h1=h10, constant interface elevation w=w0,
lower layer at rest, optionally uniform upper velocity u0.

Stored state: V = (h1, q1, w, q2).
"""
function ic_uniform_h1_w(; h10=1.0, w0=-1.0, u0=0.0, min_h=1e-10)
    return (x, Bcell) -> begin
        h1 = max(h10, min_h)

        # implied physical h2 must stay positive
        h2 = w0 - Bcell
        if h2 <= 0
            error("IC makes h2 <= 0 at x=$x: h2 = w0 - Bcell = $w0 - $Bcell = $h2. Choose w0 > max(Bcell).")
        end
        h2 = max(h2, min_h)

        q1 = h1 * u0
        q2 = 0.0

        @SVector [h1, q1, w0, q2]
    end
end

# ----------------------------
# Runner
# ----------------------------
function run_pccu_two_layer_1d(;
    nx=256, gc=2, cfl=0.6, T=10.0,
    bottom=bottom_cosine_faces(B0=-2.0, A=0.4, m=1),
    ic_fun=ic_uniform_h1_w(h10=1.0, w0=-1.0, u0=0.0),
    ρ1=0.98, ρ2=1.0, g=10.0,
    title="Two-layer SWE 1D (PCCU, w-storage)"
)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, boundary=SinFVM.PeriodicBC())

    bottom_obj = bottom isa Function ? make_bottom_faces_1d(bottom, backend, grid) : bottom

    # Equation (w-storage formulation)
    equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom_obj; ρ1=ρ1, ρ2=ρ2, g=g)

    # PCCU numerical flux (your new struct)
    numericalflux = SinFVM.PathConservativeCentralUpwind(equation)

    # Reconstruction (2nd order)
    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1))

    # Source terms (kept as in your framework)
    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cs = SinFVM.ConservedSystem(
        backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src]
    )

    simulator = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    # ----------------------------
    # Set ICs
    # ----------------------------
    x     = SinFVM.cell_centers(grid)
    Bvals = SinFVM.collect_topography_cells(bottom_obj, grid; interior=true)

    initial = [ic_fun(x[i], Bvals[i]) for i in eachindex(x)]
    SinFVM.set_current_state!(simulator, initial)

    # ----------------------------
    # Diagnostics: initial
    # ----------------------------
    st0 = SinFVM.current_interior_state(simulator)
    h1_0, q1_0, w_0, q2_0 = st0.h1, st0.q1, st0.w, st0.q2
    h2_0 = w_0 .- Bvals
    η_0  = h1_0 .+ w_0

    u1_0 = SinFVM.desingularize.(Ref(equation), h1_0, q1_0)
    u2_0 = SinFVM.desingularize.(Ref(equation), h2_0, q2_0)

    println("---- initial checks ----")
    @show extrema(Bvals)
    @show minimum(h1_0) maximum(h1_0)
    @show minimum(h2_0) maximum(h2_0)
    @show minimum(w_0)  maximum(w_0)
    @show maximum(abs.(u1_0)) maximum(abs.(u2_0))
    @show minimum(η_0) maximum(η_0)

    # ----------------------------
    # Plot: initial
    # ----------------------------
    f = Figure(size=(1600, 600), fontsize=24)
    ax_surf = Axis(f[1, 1],
        title="$title (surfaces). nx=$nx, T=$T",
        ylabel="elevations", xlabel=L"x"
    )
    ax_vel  = Axis(f[1, 2],
        title="$title (velocities). nx=$nx, T=$T",
        ylabel="u", xlabel=L"x"
    )

    lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
    lines!(ax_surf, x, w_0, label=L"w(x,0)")
    lines!(ax_surf, x, η_0, label=L"\eta(x,0)=h_1+w")

    lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
    lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

    axislegend(ax_surf, position=:lt)
    axislegend(ax_vel, position=:lt)
    display(f)

    # ----------------------------
    # Micro-step sanity
    # ----------------------------
    println("---- micro-step sanity (t = 1e-4) ----")
    SinFVM.simulate_to_time(simulator, 1e-4)
    st_micro = SinFVM.current_interior_state(simulator)
    h2_micro = st_micro.w .- Bvals

    @show minimum(st_micro.h1) minimum(h2_micro)
    @show any(isnan, st_micro.h1) any(isnan, st_micro.w) any(isnan, st_micro.q1) any(isnan, st_micro.q2)

    if any(isnan, st_micro.h1) || any(isnan, st_micro.w) || any(isnan, st_micro.q1) || any(isnan, st_micro.q2)
        error("NaNs appeared by t=1e-4. Check PCCU flux + bottoms (Bface-/+) + reconstruction consistency.")
    end
    if minimum(h2_micro) <= 0
        error("Dry lower layer (h2<=0) appeared by t=1e-4. Check positivity and w0 > max(Bcell).")
    end

    # Reinitialize from IC so main run starts at t=0
    SinFVM.set_current_state!(simulator, initial)

    # ----------------------------
    # Run
    # ----------------------------
    println("---- run to T ----")
    @time SinFVM.simulate_to_time(simulator, T)

    stF = SinFVM.current_interior_state(simulator)
    h1F, q1F, wF, q2F = stF.h1, stF.q1, stF.w, stF.q2

    h2F = wF .- Bvals
    ηF  = h1F .+ wF

    u1F = SinFVM.desingularize.(Ref(equation), h1F, q1F)
    u2F = SinFVM.desingularize.(Ref(equation), h2F, q2F)

    println("---- final checks ----")
    @show minimum(h1F) maximum(h1F)
    @show minimum(h2F) maximum(h2F)
    @show minimum(q1F) maximum(q1F)
    @show maximum(abs.(u1F)) maximum(abs.(u2F))
    @show minimum(wF) maximum(wF)
    @show minimum(ηF) maximum(ηF)

    # Overlay final
    lines!(ax_surf, x, wF, linestyle=:dot, linewidth=5, label=L"w(x,t)")
    lines!(ax_surf, x, ηF, linestyle=:dot, linewidth=5, label=L"\eta(x,t)=h_1+w")
    lines!(ax_vel,  x, u1F, linestyle=:dashdot, linewidth=5, label=L"u_1(x,t)")
    lines!(ax_vel,  x, u2F, linestyle=:dashdot, linewidth=5, label=L"u_2(x,t)")

    axislegend(ax_surf, position=:lt)
    axislegend(ax_vel, position=:lt)
    display(f)

    return f, simulator
end

# ============================================================
# Example run: equilibrium test on cosine bathymetry
# ============================================================
bottom = bottom_cosine_faces(B0=-2.0, A=0.4, m=1)
ic     = ic_uniform_h1_w(h10=1.0, w0=-1.0, u0=0.0)

f, sim = run_pccu_two_layer_1d(
    nx=128,
    gc=2,
    bottom=bottom,
    ic_fun=ic,
    T=100.0,
    cfl=0.6,
    title="Equilibrium test: constant h1 and constant w on cosine bathymetry (PCCU)"
)