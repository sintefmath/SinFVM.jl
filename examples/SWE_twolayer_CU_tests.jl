using CairoMakie
using StaticArrays
using SinFVM

# ----------------------------
# Parameters (paper)
# ----------------------------
const g_ex   = 10.0
const r_ex   = 0.98
const ρ1_ex  = r_ex
const ρ2_ex  = 1.0
const Tfinal = 0.15

# Bottom topography B(x) (Eq. 2.35)
function Bfun(x)
    if 0.4 < x < 0.6
        return 0.25*(cos(10π*(x - 0.5)) + 1.0) - 2.0
    else
        return -2.0
    end
end

# Bottom is stored on faces/intersections: length = totalcells+1
function bottom_faces_array(grid)
    xf = SinFVM.cell_faces(grid; interior=false)
    return [Bfun(x) for x in xf]
end

# IC (paper)
function ic_state(x)
    h1 = (0.1 < x < 0.2) ? 1.00001 : 1.00000
    q1 = 0.0
    w  = -1.0
    q2 = 0.0
    return @SVector [h1, q1, w, q2]
end

function run_2_7_2_once(; nx::Int, gc::Int=2, cfl=0.45,
    scheme::Symbol=:pccu, θ=1.0, boundary::Symbol=:neumann)

    backend = SinFVM.make_cpu_backend()

    bc =
        boundary == :periodic ? SinFVM.PeriodicBC() :
        boundary == :neumann  ? SinFVM.NeumannBC()  :
        boundary == :wall     ? SinFVM.WallBC()     :
        error("boundary must be :periodic, :neumann, or :wall")

    grid = SinFVM.CartesianGrid(nx; gc=gc, boundary=bc, extent=[0.0 1.0])

    Bfaces = bottom_faces_array(grid)
    bottom = SinFVM.BottomTopography1D(Bfaces, backend, grid)

    equation = SinFVM.TwoLayerShallowWaterEquations1D(bottom; ρ1=ρ1_ex, ρ2=ρ2_ex, g=g_ex)

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(θ))

    numericalflux =
        scheme == :cu   ? SinFVM.CentralUpwind(equation) :
        scheme == :pccu ? SinFVM.PathConservativeCentralUpwind(equation) :
        error("scheme must be :cu or :pccu")

    cs  = SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid,
                                 [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()])
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    # IC
    x = SinFVM.cell_centers(grid; interior=true)
    SinFVM.set_current_state!(sim, [ic_state(xi) for xi in x])

    # ε at t=0 (for optional fluctuation plot)
    st0 = SinFVM.current_interior_state(sim)
    ε0 = collect(st0.h1) .+ collect(st0.w)

    println("Running nx=$nx scheme=$scheme boundary=$boundary to T=$Tfinal ...")
    @time SinFVM.simulate_to_time(sim, Tfinal)

    st = SinFVM.current_interior_state(sim)
    ε  = collect(st.h1) .+ collect(st.w)
    Δε = ε .- ε0

    return sim, grid, ε0, ε, Δε
end

function plot_epsilon(sim_grid_pairs; title="", plot_fluctuation=false)
    fig = Figure(size=(1600, 700), fontsize=18)
    Label(fig[0, 1], title, fontsize=22)

    ax = Axis(fig[1, 1],
        title = plot_fluctuation ? "Fluctuation Δε = ε(t) − ε(0)" : "Surface ε = h₁ + w",
        xlabel="x",
        ylabel = plot_fluctuation ? "Δε" : "ε")

    for (_sim, _grid, _ε0, _ε, _Δε, lbl) in sim_grid_pairs
        x = SinFVM.cell_centers(_grid; interior=true)
        y = plot_fluctuation ? _Δε : _ε
        lines!(ax, x, y, linewidth=2, label=lbl)
    end

    axislegend(ax, position=:lt)
    display(fig)
    return fig
end

function main()
    scheme   = :cu
    boundary = :neumann    # <-- matches the paper-style plot much better than periodic at t=0.15
    cfl = 0.45
    θ = 1.0

    nx_list = [100, 200, 1600]
    sims = Tuple{Any,Any,Vector{Float64},Vector{Float64},Vector{Float64},String}[]

    for nx in nx_list
        sim, grid, ε0, ε, Δε = run_2_7_2_once(nx=nx, scheme=scheme, boundary=boundary, cfl=cfl, θ=θ)
        push!(sims, (sim, grid, ε0, ε, Δε, "Δx=1/$nx"))
    end

    # Paper axis label is ε, and caption says “water surface ε at t=0.15”
    plot_epsilon(sims; title="Sec. 2.7.2 | scheme=$scheme | boundary=$boundary | t=$Tfinal",
                 plot_fluctuation=false)

    # If you want to verify the “fluctuation” wording in the caption:
    # plot_epsilon(sims; title="Sec. 2.7.2 | Δε | scheme=$scheme | boundary=$boundary | t=$Tfinal",
    #              plot_fluctuation=true)
end

main()