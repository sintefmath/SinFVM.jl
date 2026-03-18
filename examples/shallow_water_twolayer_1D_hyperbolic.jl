using CairoMakie
using StaticArrays
using SinFVM


const PLOT_DIR = raw"C:\Users\peder\OneDrive - NTNU\År 5\Masteroppgave\Plots- Two-Layer"

function save_plot(fig, filename; folder=PLOT_DIR)
    mkpath(folder)
    path = joinpath(folder, filename)
    save(path, fig)
    println("Saved plot to: ", path)
end

# ============================================================
# Problem setup
# ============================================================

const B0 = 0.0
const UL_test = (h1 = 0.50, q1 = 1.0, h2 = 0.50, q2 = -0.10)
const UR_test = (h1 = 0.50, q1 = 0.05, h2 = 0.50, q2 = 0.04)

const FOUR_CASES = [
    ("old eig, no corr", :old, false),
    ("old eig, corr",    :old, true),
    ("new eig, no corr", :new, false),
    ("new eig, corr",    :new, true),
]

const CORRECTED_CASES = [
    ("old eig, corr", :old),
    ("new eig, corr", :new),
]

riemann_ic_w(x, UL, UR, B) =
    x < 0.0 ? @SVector([UL.h1, UL.q1, UL.h2 + B, UL.q2]) :
              @SVector([UR.h1, UR.q1, UR.h2 + B, UR.q2])

function make_equation_1d(; ρ1=0.98, ρ2=1.0, g=9.81,
                          depth_cutoff=1e-5, desingularizing_kappa=1e-5,
                          eigenvalue_method=:old, hyperbolicity_correction=true)
    bottom = SinFVM.ConstantBottomTopography(B0)
    SinFVM.TwoLayerShallowWaterEquations1D(
        bottom;
        ρ1, ρ2, g,
        depth_cutoff,
        desingularizing_kappa,
        eigenvalue_method,
        hyperbolicity_correction,
    )
end

# ============================================================
# Core simulation / postprocessing
# ============================================================

function run_case_1d(; nx=800, gc=2, T=0.25, cfl=0.45,
                     ρ1=0.98, ρ2=1.0, g=9.81,
                     UL=UL_test, UR=UR_test,
                     eigenvalue_method=:old,
                     hyperbolicity_correction=true)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[-1.0 1.0], boundary=SinFVM.NeumannBC())
    x = SinFVM.cell_centers(grid; interior=true)

    eq = make_equation_1d(
        ρ1=ρ1, ρ2=ρ2, g=g,
        eigenvalue_method=eigenvalue_method,
        hyperbolicity_correction=hyperbolicity_correction,
    )

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)
    sources = [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()]

    cs = SinFVM.ConservedSystem(backend, reconstruction, flux, eq, grid, sources)
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    initial = [riemann_ic_w(xi, UL, UR, B0) for xi in x]
    SinFVM.set_current_state!(sim, initial)
    SinFVM.simulate_to_time(sim, T)

    return sim, x, eq
end

function state_to_fields(h1, q1, w, q2, eq)
    B = eq.B isa SinFVM.ConstantBottomTopography ? eq.B.B : B0
    h2 = w .- B

    u1 = similar(h1)
    u2 = similar(h2)
    shear = similar(h1)

    @inbounds for i in eachindex(h1)
        u1[i] = SinFVM.desingularize(eq, h1[i], q1[i])
        u2[i] = SinFVM.desingularize(eq, h2[i], q2[i])
        shear[i] = abs(u1[i] - u2[i])
    end

    ξ = h1 .+ h2 .+ B
    ω = h2 .+ B

    return (; h1, h2, q1, q2, u1, u2, shear, ξ, ω)
end

function extract_fields_1d(sim, eq)
    st = SinFVM.current_interior_state(sim)
    state_to_fields(collect(st.h1), collect(st.q1), collect(st.w), collect(st.q2), eq)
end

function extract_ic_fields_1d(x, eq; UL=UL_test, UR=UR_test)
    U0 = [riemann_ic_w(xi, UL, UR, B0) for xi in x]
    h1 = [U[1] for U in U0]
    q1 = [U[2] for U in U0]
    w  = [U[3] for U in U0]
    q2 = [U[4] for U in U0]
    state_to_fields(h1, q1, w, q2, eq)
end

# ============================================================
# Plotting
# ============================================================

function make_axes_4panel(fig; title="", row0=0)
    Label(fig[row0, 1:2], title, fontsize=24)
    ax11 = Axis(fig[row0 + 1, 1], title="Upper-layer depth h₁", xlabel="x", ylabel="h₁")
    ax12 = Axis(fig[row0 + 1, 2], title="Lower-layer depth h₂", xlabel="x", ylabel="h₂")
    ax21 = Axis(fig[row0 + 2, 1], title="Velocities u₁ and u₂", xlabel="x", ylabel="velocity")
    ax22 = Axis(fig[row0 + 2, 2], title="Interfacial shear |u₁-u₂|", xlabel="x", ylabel="shear")
    return ax11, ax12, ax21, ax22
end

function plot_fields_comparison(results, x; title="Comparison")
    fig = Figure(size=(1700, 1100), fontsize=18)
    axh, axw, axu, axs = make_axes_4panel(fig; title=title)

    for (label, fld) in results
        lines!(axh, x, fld.h1, linewidth=2, label=label)
        lines!(axw, x, fld.h2, linewidth=2, label=label)
        lines!(axu, x, fld.u1, linewidth=2, label="$label (u₁)")
        lines!(axu, x, fld.u2, linewidth=2, linestyle=:dash, label="$label (u₂)")
        lines!(axs, x, fld.shear, linewidth=2, label=label)
    end

    axislegend(axh, position=:rb)
    axislegend(axw, position=:rb)
    axislegend(axu, position=:rb)
    axislegend(axs, position=:rb)

    display(fig)
    return fig
end

function plot_ic_1d(ic, x)
    plot_fields_comparison(Dict("initial" => ic), x; title="Initial conditions at t = 0")
end

function plot_corrected_resolution_comparison_1d(study; T=0.25, nxs=(200, 800, 3200))
    fig = Figure(size=(1800, 1000), fontsize=18)

    Label(fig[0, 1:2], "Corrected schemes: grid refinement comparison at t = $T", fontsize=24)

    ax11 = Axis(fig[1, 1], title="old eig, corr — h₁", xlabel="x", ylabel="h₁")
    ax12 = Axis(fig[1, 2], title="old eig, corr — h₂", xlabel="x", ylabel="h₂")
    ax21 = Axis(fig[2, 1], title="new eig, corr — h₁", xlabel="x", ylabel="h₁")
    ax22 = Axis(fig[2, 2], title="new eig, corr — h₂", xlabel="x", ylabel="h₂")

    for nx in nxs
        lw = nx == maximum(nxs) ? 3 : 2
        ls = nx == maximum(nxs) ? :dash : :solid
        lab = nx == maximum(nxs) ? "nx = $nx (ref)" : "nx = $nx"

        old = study["old eig, corr"][nx]
        new = study["new eig, corr"][nx]

        lines!(ax11, old.x, old.fields.h1, linewidth=lw, linestyle=ls, label=lab)
        lines!(ax12, old.x, old.fields.h2, linewidth=lw, linestyle=ls, label=lab)
        lines!(ax21, new.x, new.fields.h1, linewidth=lw, linestyle=ls, label=lab)
        lines!(ax22, new.x, new.fields.h2, linewidth=lw, linestyle=ls, label=lab)
    end

    axislegend(ax11, position=:rb)
    axislegend(ax12, position=:rb)
    axislegend(ax21, position=:rb)
    axislegend(ax22, position=:rb)

    display(fig)
    return fig
end

# ============================================================
# Drivers
# ============================================================

function run_four_case_study_1d(; nx=800, gc=2, T=0.25, cfl=0.45,
                                ρ1=0.98, ρ2=1.0, g=9.81,
                                UL=UL_test, UR=UR_test)

    results = Dict{String, Any}()
    xref, eqref = nothing, nothing

    for (label, eigmethod, corrflag) in FOUR_CASES
        println("Running case: $label")
        sim, x, eq = run_case_1d(
            nx=nx, gc=gc, T=T, cfl=cfl,
            ρ1=ρ1, ρ2=ρ2, g=g,
            UL=UL, UR=UR,
            eigenvalue_method=eigmethod,
            hyperbolicity_correction=corrflag,
        )
        results[label] = extract_fields_1d(sim, eq)
        xref, eqref = x, eq
    end

    ic = extract_ic_fields_1d(xref, eqref; UL=UL, UR=UR)
    fig_ic = plot_ic_1d(ic, xref)
    fig_final = plot_fields_comparison(results, xref; title="1D PCCU comparison at t = $T")

    return (; fig_ic, fig_final, ic, results, x=xref)
end

function run_corrected_resolution_study_1d(; nxs=(200, 800, 3200), gc=2, T=0.25, cfl=0.45,
                                           ρ1=0.99, ρ2=1.0, g=9.81,
                                           UL=UL_test, UR=UR_test)

    out = Dict(label => Dict{Int, Any}() for (label, _) in CORRECTED_CASES)

    for nx in nxs
        for (label, eigmethod) in CORRECTED_CASES
            println("Running $label at nx = $nx")
            sim, x, eq = run_case_1d(
                nx=nx, gc=gc, T=T, cfl=cfl,
                ρ1=ρ1, ρ2=ρ2, g=g,
                UL=UL, UR=UR,
                eigenvalue_method=eigmethod,
                hyperbolicity_correction=true,
            )
            out[label][nx] = (; x, fields=extract_fields_1d(sim, eq))
        end
    end

    return out
end

# ============================================================
# Example runs
# ============================================================

study4 = run_four_case_study_1d(
    nx=100,
    gc=2,
    T=0.25,
    cfl=0.45,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)

study_corr = run_corrected_resolution_study_1d(
    nxs=(200, 800, 3200),
    gc=2,
    T=0.25,
    cfl=0.45,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)

fig_corr = plot_corrected_resolution_comparison_1d(study_corr; T=0.25, nxs=(200, 800, 3200))