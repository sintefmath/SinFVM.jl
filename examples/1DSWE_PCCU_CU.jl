# ============================================================
# Example 5.4 (Barotropic Tidal Flow) — 1D Two-layer SWE
# Paper-faithful setup (domain [-10,10], flat bottom, periodic forcing on LEFT
# for h1 and h2, and q1/q2 by zero-order interpolation; open BC on RIGHT).
#
# Reference: Castro, Kurganov, Morales de Luna (2019) ESAIM:M2AN 53, 959–985
#            Kurganov & Petrova (2009) SIAM J. Sci. Comput. 31, 1742–1773
#
# Storage in SinFVM:
#   V = (h1, q1, w, q2), with w = h2 + B
#   B = Zref = –0.5*(h1L+h2L+h1R+h2R)  (paper eq. (5.5), "CUc" reference level)
#
# Plots (paper):
#   water surface  ξ = h1 + h2 + Zref = h1 + w   (since w = h2 + Zref)
#   interface      ω = h2 + Zref       = w
#
# Key fixes vs. original:
#   - g = 9.81 (paper §5: “the constant gravitational acceleration g = 9.81”)
#   - B = Zref (so w = h2 + Zref; ensures CU matches paper’s CUc scheme)
# ============================================================

using CairoMakie
using StaticArrays
using SinFVM

# ----------------------------
# Paper parameters (Example 5.4)
# ----------------------------
const UL_paper = (h1=0.69914, q1=-0.21977, h2=1.26932, q2=0.20656)
const UR_paper = (h1=0.37002, q1=-0.18684, h2=1.59310, q2=0.17416)

# Reference level Zref (paper eq. (5.5)) for flat bottom
compute_Zref(UL, UR) = -0.5 * (UL.h1 + UL.h2 + UR.h1 + UR.h2)

# Left boundary forcing (paper):
# h1(-10,t) = h1L + h1L*(0.03/|Zref|)*sin(π t / 50)
# h2(-10,t) = h2L + h1L*(0.03/|Zref|)*sin(π t / 50)
function left_boundary_heights(t, UL, Zref)
    amp = (0.03 / abs(Zref)) * sin(pi * t / 50.0)
    h1 = UL.h1 + UL.h1 * amp
    h2 = UL.h2 + UL.h1 * amp
    return h1, h2
end

# Initial Riemann data: UL for x<0, UR for x>=0
function riemann_ic(x, UL, UR)
    if x < 0.0
        return @SVector [UL.h1, UL.q1, UL.h2, UL.q2]
    else
        return @SVector [UR.h1, UR.q1, UR.h2, UR.q2]
    end
end

# ----------------------------
# IMPORTANT: disable built-in BC updates
# (otherwise SinFVM may overwrite your ghost cells each substep)
# ----------------------------

function bc_callback!(t, sim, grid, UL, UR)
    gc = SinFVM.ghost_cells(grid, SinFVM.XDIR)
    U  = SinFVM.current_state(sim)

    first_interior = gc + 1
    last_interior  = grid.totalcells[1] - gc

    Zref = compute_Zref(UL, UR)
    h1L, h2L = left_boundary_heights(t, UL, Zref)  # physical heights

    # Zero-order interpolation for q1,q2 on the left edge
    q1L = U[first_interior][2]
    q2L = U[first_interior][4]

    # State vector is (h1, q1, w, q2) with w = h2 + Zref
    Uleft = @SVector [h1L, q1L, h2L + Zref, q2L]

    # LEFT ghost cells: indices immediately left of first interior
    @inbounds for i in 1:gc
        U[first_interior - i] = Uleft
    end

    # RIGHT ghost cells: open boundary (copy last interior state)
    Uright = U[last_interior]
    @inbounds for i in 1:gc
        U[last_interior + i] = Uright
    end

    return nothing
end

# ----------------------------
# Runner
# ----------------------------
function run_example_5_4_1d(; nx=1000, gc=2, cfl=0.45, T=64.0,
    scheme=:pccu, ρ1=0.98, ρ2=1.0, g=9.81,
    checkpoints = [10.0, 25.0, 60.0, 64.0])

    backend = SinFVM.make_cpu_backend()

    # Paper domain [-10,10]
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[-10.0 10.0], boundary=SinFVM.NeumannBC())
    x    = SinFVM.cell_centers(grid; interior=true)

    # Paper eq. (5.5): reference bottom level for CU well-balancing ("CUc" choice)
    # PCCU is invariant to this choice; CU gives best results with this Zref.
    Zref = compute_Zref(UL_paper, UR_paper)

    bottom   = SinFVM.ConstantBottomTopography(Zref)  # flat B = Zref
    equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1, ρ2=ρ2, g=g)

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))

    numericalflux =
        scheme == :pccu ? SinFVM.PathConservativeCentralUpwind(equation) :
        scheme == :cu   ? SinFVM.CentralUpwind(equation) :
        error("scheme must be :cu or :pccu")

    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cs  = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    # IC: Riemann data at x=0
    # State vector is (h1, q1, w, q2) with w = h2 + B = h2 + Zref
    function riemann_ic_w(xi)
        if xi < 0.0
            return @SVector [UL_paper.h1, UL_paper.q1, UL_paper.h2 + Zref, UL_paper.q2]
        else
            return @SVector [UR_paper.h1, UR_paper.q1, UR_paper.h2 + Zref, UR_paper.q2]
        end
    end
    initial = [riemann_ic_w(x[i]) for i in eachindex(x)]
    SinFVM.set_current_state!(sim, initial)

    # Enforce BC at t=0 (after IC is set)
    bc_callback!(0.0, sim, grid, UL_paper, UR_paper)

    # Helper to read fields (interior) and produce paper variables
    # With B = Zref: w = h2 + Zref, so h2 = w - Zref
    # Paper water surface ξ = h1 + h2 + Zref = h1 + w
    # Paper interface    ω = h2 + Zref          = w
    function fields()
        st = SinFVM.current_interior_state(sim)
        h1 = collect(st.h1)
        q1 = collect(st.q1)
        w  = collect(st.w)      # w = h2 + Zref (equilibrium variable)
        q2 = collect(st.q2)
        h2 = w .- Zref          # physical lower-layer depth

        ξ = h1 .+ w             # paper water surface ξ = h1 + h2 + Zref = h1 + w
        ω = w                   # paper interface ω = h2 + Zref = w

        return (; h1, q1, h2, q2, ξ, ω, Zref)
    end

    # ----------------------------
    # Plot setup (robust labels; no TeX interpolation issues)
    # ----------------------------
    fig = Figure(size=(1600, 750), fontsize=18)
    Label(fig[0, 1:2],
        "Example 5.4 (1D) | scheme=$(scheme) | nx=$nx | T=$T",
        fontsize=22)
    
    axξ = Axis(fig[1, 1], title="Water surface ξ = h₁ + h₂ + Zref", xlabel="x", ylabel="ξ")
    axω = Axis(fig[1, 2], title="Interface ω = h₂ + Zref",           xlabel="x", ylabel="ω")

    # initial curves
    fld0 = fields()
    lines!(axξ, x, fld0.ξ, linewidth=2, label="t=0")
    lines!(axω, x, fld0.ω, linewidth=2, label="t=0")

    # STEP-CALLBACK: attempt to update BC every integrator callback point.
    # (In many SinFVM setups this is called every step/stage, unlike IntervalWriter.)
    step_cb = (t, sim_) -> bc_callback!(t, sim_, grid, UL_paper, UR_paper)

    # Time stepping to checkpoints
    for ttarget in checkpoints
        SinFVM.simulate_to_time(sim, ttarget; callback=step_cb)

        # Ensure ghosts are consistent at sampling time too
        bc_callback!(ttarget, sim, grid, UL_paper, UR_paper)

        fld = fields()
        tstr = string(round(ttarget; digits=2))
        lines!(axξ, x, fld.ξ, linewidth=2, label="t=$tstr")
        lines!(axω, x, fld.ω, linewidth=2, label="t=$tstr")
        println("Reached checkpoint t=$ttarget")
    end

    axislegend(axξ, position=:lt)
    axislegend(axω, position=:lt)

    display(fig)
    return fig, sim
end

# ============================================================
# Run PCCU (recommended) or CU
# ============================================================
fig, sim = run_example_5_4_1d(
    nx=1000,
    gc=2,
    T=64.0,
    cfl=0.45,
    scheme=:pccu,   # :cu or :pccu
    ρ1=0.999,
    ρ2=1.0,
    g=9.81,         # paper uses g=9.81 (stated in §5)
    checkpoints=[10.0, 25.0, 60.0, 64.0]
)