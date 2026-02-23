using CairoMakie
using StaticArrays
using SinFVM

# ============================================================
# 2D Interface propagation example (Kurganov–Petrova style)
# - Flat bottom B = -1
# - Two-layer SWE in SinFVM storage:
#     U = (h1, q1, p1, w, q2, p2),  where w = h2 + B
# - Paper IC is piecewise constant on Ω (defined in paper coords x,y ∈ [-1,1])
# - We run to T=0.1 and plot:
#     ε = η - η0, where η = h1 + w  (free surface elevation)
#     h1 (upper layer thickness)
#   with dashed initial interface (circle) overlay
# - Plot window (paper-like):
#     x ∈ [-0.5, 0.5], y ∈ [-0.5, 0.7]
# ============================================================

# ----------------------------
# Coordinate map: SinFVM default domain is [0,1]×[0,1]
# Paper uses [-1,1]×[-1,1]
# ----------------------------
paper_x(x̂) = 2.0*x̂ - 1.0
paper_y(ŷ) = 2.0*ŷ - 1.0

# ----------------------------
# Ω set from the paper
# Ω = {x < -0.5, y < 0} ∪ {(x+0.5)^2+(y+0.5)^2 < 0.25} ∪ {x < 0, y < -0.5}
# ----------------------------
function in_Omega(x, y)
    cond1 = (x < -0.5) && (y < 0.0)
    cond2 = (x + 0.5)^2 + (y + 0.5)^2 < 0.25
    cond3 = (x < 0.0) && (y < -0.5)
    return cond1 || cond2 || cond3
end

# ----------------------------
# Initial condition in SinFVM variables
# inside Ω:  (h1,q1,p1,h2,q2,p2) = (0.50, 1.250, 1.250, 0.50, 1.250, 1.250)
# outside Ω: (h1,q1,p1,h2,q2,p2) = (0.45, 1.125, 1.125, 0.55, 1.375, 1.375)
# Flat bottom: B=-1 => w = h2 + B = h2 - 1
# ----------------------------
function ic_interface_propagation(; B0=-1.0)
    return (xŷ) -> begin
        x̂ = xŷ[1]
        ŷ = xŷ[2]
        x  = paper_x(x̂)
        y  = paper_y(ŷ)

        if in_Omega(x, y)
            h1, q1, p1 = 0.50, 1.250, 1.250
            h2, q2, p2 = 0.50, 1.250, 1.250
        else
            h1, q1, p1 = 0.45, 1.125, 1.125
            h2, q2, p2 = 0.55, 1.375, 1.375
        end

        w = h2 + B0
        return @SVector [h1, q1, p1, w, q2, p2]
    end
end

# ----------------------------
# Helpers for plotting
# ----------------------------
function paper_xy_arrays(grid)
    xy = SinFVM.cell_centers(grid; interior=true)  # matrix of tuples (x̂,ŷ)
    nx, ny = size(xy)
    X = Matrix{Float64}(undef, nx, ny)
    Y = Matrix{Float64}(undef, nx, ny)
    @inbounds for J in 1:ny, I in 1:nx
        X[I,J] = paper_x(xy[I,J][1])
        Y[I,J] = paper_y(xy[I,J][2])
    end
    return X, Y
end

function interior_h1_w(sim)
    st = SinFVM.current_interior_state(sim)
    return collect(st.h1), collect(st.w)
end

eta_from(h1, w) = h1 .+ w  # free-surface elevation

"Dashed initial interface location: circle boundary (x+0.5)^2+(y+0.5)^2=0.25"
function initial_interface_circle(; n=600)
    θ = range(0, 2π; length=n)
    x = -0.5 .+ 0.5*cos.(θ)
    y = -0.5 .+ 0.5*sin.(θ)
    return x, y
end

# ----------------------------
# One run (one resolution) and return sim + grid + eta0
# ----------------------------
function run_interface_propagation_once(; nx=200, ny=200, T=0.1, cfl=0.45,
    use_pccu=true, θ=1.0, g=10.0, r=0.98, B0=-1.0)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx, ny; gc=2, boundary=SinFVM.PeriodicBC())

    bottom = SinFVM.ConstantBottomTopography(B0)

    # ρ2 is the reference; set ρ1=r, ρ2=1.0 so r=ρ1/ρ2
    equation = SinFVM.TwoLayerShallowWaterEquations2D(bottom; ρ1=r, ρ2=1.0, g=g)

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(θ))
    numericalflux  = use_pccu ? SinFVM.PathConservativeCentralUpwind(equation) : SinFVM.CentralUpwind(equation)

    bottom_src = SinFVM.SourceTermBottom()
    ncp_src    = SinFVM.SourceTermNonConservative()

    cs  = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_src, ncp_src])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    # IC
    xy_int = SinFVM.cell_centers(grid; interior=true)
    ic = ic_interface_propagation(B0=B0)
    initial = [ic(xy_int[I]) for I in eachindex(xy_int)]
    SinFVM.set_current_state!(sim, initial)

    # Store eta0
    h1_0, w_0 = interior_h1_w(sim)
    eta0 = eta_from(h1_0, w_0)

    # Run
    scheme = use_pccu ? "PCCU" : "CU"
    println("---- running: $scheme nx=$nx ny=$ny T=$T cfl=$cfl ----")
    @time SinFVM.simulate_to_time(sim, T)

    return sim, grid, eta0
end

# ----------------------------
# Paper-style 3×2 contour plot: rows = resolutions, columns = (ε, h1)
# ε = η - η0
# Axis window like the paper:
#    x ∈ [-0.5, 0.5], y ∈ [-0.5, 0.7]
# ----------------------------
function plot_paper_pair(sim, grid, eta0;
    levels_eps=12,
    levels_h1=12,
    title="")

    X, Y = paper_xy_arrays(grid)

    h1, w = interior_h1_w(sim)
    η = eta_from(h1, w)
    ε = η .- eta0

    xi, yi = initial_interface_circle()

    fig = Figure(size=(1400, 600), fontsize=22)

    Label(fig[0, 1:2], title, fontsize=26)

    axε = Axis(fig[1,1],
        title=L"\varepsilon",
        xlabel=L"x",
        ylabel=L"y",
        aspect=DataAspect())

    axh1 = Axis(fig[1,2],
        title=L"h_1",
        xlabel=L"x",
        ylabel=L"y",
        aspect=DataAspect())

    # Paper window
    xlims!(axε, -0.5, 0.5)
    ylims!(axε, -0.5, 0.7)

    xlims!(axh1, -0.5, 0.5)
    ylims!(axh1, -0.5, 0.7)

    contour!(axε, X, Y, ε;
        levels=levels_eps,
        linewidth=2)

    contour!(axh1, X, Y, h1;
        levels=levels_h1,
        linewidth=2)

    lines!(axε, xi, yi,
        linestyle=:dash,
        linewidth=3,
        color=:black)

    lines!(axh1, xi, yi,
        linestyle=:dash,
        linewidth=3,
        color=:black)

    display(fig)

    return fig
end

# ============================================================
# MAIN: run 3 resolutions and plot like the paper
# ============================================================

Ns = [200, 400, 800]
use_pccu = false  # set to false to run with CU instead of PCCU
cfl = 0.8
T = 0.1
θ = 1.0
g = 10.0
r = 0.98
B0 = -1.0

scheme = use_pccu ? "PCCU" : "CU"

for N in Ns

    sim, grid, eta0 = run_interface_propagation_once(
        nx=N, ny=N, T=T, cfl=cfl,
        use_pccu=use_pccu,
        θ=θ, g=g, r=r, B0=B0
    )

    fig = plot_paper_pair(sim, grid, eta0;
        title="Interface propagation | $scheme | nx=$N | T=$T")

end
