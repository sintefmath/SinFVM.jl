using CairoMakie
using StaticArrays
using SinFVM

# ----------------------------
# Helpers: bathymetry builders
# ----------------------------

xwrap(x) = x - floor(x)

"Constant bottom B(x)=B0"
make_bottom_constant(B0) = SinFVM.ConstantBottomTopography(B0)

"Face-defined bottom for 1D, you give Bface(x_face) returning bottom at intersections"
function make_bottom_faces_1d(Bface_fun, backend, grid::CartesianGrid{1})
    xF = SinFVM.cell_faces(grid; interior=false) # includes ghosts, intersections
    Bint = similar(xF)
    @inbounds for i in eachindex(xF)
        Bint[i] = Bface_fun(xwrap(xF[i]))
    end
    return SinFVM.BottomTopography1D(Bint, backend, grid)
end

"Paper-style hump bottom in a window; otherwise constant"
function bottom_paper_hump(B0=-2.0)
    return (x -> ((0.4 < x < 0.6) ? (0.25*(cos(10π*(x - 0.5)) + 1.0) - 2.0) : B0))
end

# ----------------------------
# Helpers: initial conditions
# ----------------------------

"""
Build IC function returning PHYSICAL state (h1,q1,h2,q2) from:
- h1fun(x)
- wfun(x)  (w = h2 + B)
- q1fun(x), q2fun(x) (momenta)
"""
function make_ic_from_h1_w(h1fun, wfun; q1fun = x->0.0, q2fun = x->0.0)
    return (x, Bcell) -> begin
        h1 = h1fun(x)
        w  = wfun(x)
        h2 = w - Bcell
        q1 = q1fun(x)
        q2 = q2fun(x)
        @SVector [h1, q1, h2, q2]
    end
end

"Your current paper bump-in-h1, lake-at-rest in layer2: w=-1"
function ic_paper_bump(; bump=0.001, w0=-1.0)
    h1fun(x) = (0.1 < xwrap(x) < 0.2) ? (1.0 + bump) : 1.0
    wfun(x)  = w0
    return make_ic_from_h1_w(h1fun, wfun)
end

"Riemann-type jump in h2 via w jump, with constant h1"
function ic_riemann_w(; x0=0.5, h1=1.0, wL=-1.0, wR=-1.2)
    h1fun(x) = h1
    wfun(x)  = (xwrap(x) < x0) ? wL : wR
    return make_ic_from_h1_w(h1fun, wfun)
end

"""
Stable free surface η = const, with a cosine bump in interface w = h2 + B.
Returns IC function (x, Bcell) -> SVector(h1,q1,h2,q2).

Params:
- η0: constant free surface level
- w0: baseline interface level
- amp: bump amplitude (positive makes h2 thicker where bump is)
- x0: bump center (in wrapped domain [0,1))
- width: bump half-width (support); bump active for |x-x0| < width
- mode: number of cosine periods inside the bump window (usually 1)
- min_h1, min_h2: positivity clamps
"""
function ic_stable_surface_cosine_interface(; η0=0.0, w0=-1.0, amp=0.5,
    x0=0.5, width=0.2, mode=1,
    min_h1=1e-6, min_h2=1e-6)

    return (x, Bcell) -> begin
        ξ = xwrap(x)
        # wrapped distance to center
        d = abs(ξ - x0)
        d = min(d, 1 - d)

        bump = 0.0
        if d < width
            s = (d / width)                # s in [0,1)
            bump = amp * 0.5 * (1 + cos(mode * π * s))  # smooth cosine cap
        end

        w = w0 + bump               # interface elevation (B + h2)
        h2 = w - Bcell              # physical lower-layer depth
        h1 = η0 - w                 # enforce flat free surface η0

        # clamp (optional but recommended for big bumps)
        h1 = max(h1, min_h1)
        h2 = max(h2, min_h2)

        @SVector [h1, 0.0, h2, 0.0]
    end
end


# ----------------------------
# One runner for all cases
# ----------------------------

function run_case(; nx=128, gc=2, cfl=0.60, Tshow=0.15,
    bottom,
    ic_fun,
    ρ1=0.98, ρ2=1.0, g=10.0,
    title="Two-layer SWE 1D")

    backend = make_cpu_backend()
    grid = CartesianGrid(nx; gc=gc, boundary=SinFVM.PeriodicBC())

    equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1, ρ2=ρ2, g=g)
    numericalflux = CentralUpwind(equation)

    reconstruction = LinearLimiterReconstruction(SinFVM.VanLeerLimiter())

    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cs = ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
    simulator = Simulator(backend, cs, RungeKutta2(), grid; cfl=cfl)

    # Interior coords and cell-centered bathymetry
    x     = SinFVM.cell_centers(grid)
    Bvals = SinFVM.collect_topography_cells(bottom, grid; interior=true)

    # Set ICs (PHYSICAL storage)
    initial = [ic_fun(x[i], Bvals[i]) for i in eachindex(x)]
    SinFVM.set_current_state!(simulator, initial)

    # ----------------------------
    # Plot initial
    # ----------------------------
    f = Figure(size=(1600, 600), fontsize=24)

    ax_surf = Axis(f[1, 1], title="$title (surfaces). nx=$nx, T=$Tshow", ylabel="elevations", xlabel=L"x")
    ax_vel  = Axis(f[1, 2], title="$title (velocities). nx=$nx, T=$Tshow", ylabel="u", xlabel=L"x")

    st0 = SinFVM.current_interior_state(simulator)

    h1_0, q1_0, h2_0, q2_0 = st0.h1, st0.q1, st0.h2, st0.q2
    w0 = h2_0 .+ Bvals
    η0 = h1_0 .+ w0

    u1_0 = SinFVM.desingularize.(Ref(equation), h1_0, q1_0)
    u2_0 = SinFVM.desingularize.(Ref(equation), h2_0, q2_0)

    println("---- initial checks ----")
    @show minimum(h1_0) minimum(h2_0)
    @show maximum(abs.(u1_0)) maximum(abs.(u2_0))
    @show minimum(η0) maximum(η0)

    lines!(ax_surf, x, Bvals, linestyle=:dash, label=L"B(x)")
    lines!(ax_surf, x, w0, label=L"w(x,0)=B+h_2")
    lines!(ax_surf, x, η0, label=L"\eta(x,0)=h_1+h_2+B")

    lines!(ax_vel, x, u1_0, label=L"u_1(x,0)")
    lines!(ax_vel, x, u2_0, label=L"u_2(x,0)")

    axislegend(ax_surf, position=:lt)
    axislegend(ax_vel, position=:lt)
    display(f)

    # ----------------------------
    # Run + overlay final
    # ----------------------------
    @time SinFVM.simulate_to_time(simulator, Tshow)

    st = SinFVM.current_interior_state(simulator)
    h1, q1, h2, q2 = st.h1, st.q1, st.h2, st.q2
    w  = h2 .+ Bvals
    η  = h1 .+ w

    u1 = SinFVM.desingularize.(Ref(equation), h1, q1)
    u2 = SinFVM.desingularize.(Ref(equation), h2, q2)

    println("---- final checks ----")
    @show minimum(h1) minimum(h2)
    @show maximum(abs.(u1)) maximum(abs.(u2))
    @show minimum(η) maximum(η)

    lines!(ax_surf, x, w, linestyle=:dot, linewidth=5, label=L"w(x,t)=B+h_2")
    lines!(ax_surf, x, η, linestyle=:dot, linewidth=5, label=L"\eta(x,t)=h_1+h_2+B")

    lines!(ax_vel, x, u1, linestyle=:dashdot, linewidth=5, label=L"u_1(x,t)")
    lines!(ax_vel, x, u2, linestyle=:dashdot, linewidth=5, label=L"u_2(x,t)")

    axislegend(ax_surf, position=:lt)
    axislegend(ax_vel, position=:lt)
    display(f)

    return f, simulator
end


#============================================================================================================#
# Example runs
#=============================================================================================================#
#bottom = make_bottom_constant(-2.0)
#ic = ic_paper_bump(bump=0.1, w0=-1.0)
#f, sim = run_case(bottom=bottom, ic_fun=ic, Tshow=0.15, title="Paper bump, constant bottom")


bottom = make_bottom_constant(-2.0)
ic = ic_stable_surface_cosine_interface(η0=0.0, w0=-1.0, amp=0.8, x0=0.5, width=0.15)
f, sim = run_case(bottom=bottom, ic_fun=ic, Tshow=100, title="Stable surface, interface bump")
