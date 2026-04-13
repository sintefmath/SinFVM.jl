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
const UL_test = (h1 = 0.50, q1 = 3.2, h2 = 0.50, q2 = -0.10)
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

function make_equation_1d(; B=B0, ρ1=0.98, ρ2=1.0, g=9.81,
                          depth_cutoff=1e-5, desingularizing_kappa=1e-5,
                          eigenvalue_method=:old, hyperbolicity_correction=true)
    bottom = SinFVM.ConstantBottomTopography(B)
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
                     B=B0,
                     ρ1=0.98, ρ2=1.0, g=9.81,
                     UL=UL_test, UR=UR_test,
                     eigenvalue_method=:old,
                     hyperbolicity_correction=true)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[-1.0 1.0], boundary=SinFVM.PeriodicBC())
    x = SinFVM.cell_centers(grid; interior=true)

    eq = make_equation_1d(
        B=B, ρ1=ρ1, ρ2=ρ2, g=g,
        eigenvalue_method=eigenvalue_method,
        hyperbolicity_correction=hyperbolicity_correction,
    )

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)
    # flux = SinFVM.CentralUpwind(eq)
    sources = [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()]

    cs = SinFVM.ConservedSystem(backend, reconstruction, flux, eq, grid, sources)
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    initial = [riemann_ic_w(xi, UL, UR, B) for xi in x]
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
    B = eq.B isa SinFVM.ConstantBottomTopography ? eq.B.B : B0
    U0 = [riemann_ic_w(xi, UL, UR, B) for xi in x]
    h1 = [U[1] for U in U0]
    q1 = [U[2] for U in U0]
    w  = [U[3] for U in U0]
    q2 = [U[4] for U in U0]
    state_to_fields(h1, q1, w, q2, eq)
end

# ============================================================
# Plotting
# ============================================================

function make_axes_3panel(fig; title="", row0=0)
    Label(fig[row0, 1:2], title, fontsize=24)

    axL = Axis(
        fig[row0 + 1:row0 + 2, 1],
        title="Free surface ε and interface w",
        xlabel="x",
        ylabel="height",
    )

    axu1 = Axis(
        fig[row0 + 1, 2],
        title="Upper-layer velocity u₁",
        xlabel="x",
        ylabel="u₁",
    )

    axu2 = Axis(
        fig[row0 + 2, 2],
        title="Lower-layer velocity u₂",
        xlabel="x",
        ylabel="u₂",
    )

    return axL, axu1, axu2
end

function plot_fields_comparison(results, x; title="Comparison")
    fig = Figure(size=(1700, 900), fontsize=18)
    axL, axu1, axu2 = make_axes_3panel(fig; title=title)

    colors = Makie.wong_colors()

    surf_plots = Any[]
    surf_labels = String[]
    int_plots = Any[]
    int_labels = String[]
    u1_plots = Any[]
    u1_labels = String[]
    u2_plots = Any[]
    u2_labels = String[]

    for (i, (label, fld)) in enumerate(results)
        color = colors[mod1(i, length(colors))]

        pξ = lines!(axL,  x, fld.ξ,  linewidth=2, color=color, linestyle=:solid)
        pω = lines!(axL,  x, fld.ω,  linewidth=2, color=color, linestyle=:dash)
        pu1 = lines!(axu1, x, fld.u1, linewidth=2, color=color)
        pu2 = lines!(axu2, x, fld.u2, linewidth=2, color=color)

        push!(surf_plots, pξ)
        push!(surf_labels, "$label: ε")

        push!(int_plots, pω)
        push!(int_labels, "$label: w")

        push!(u1_plots, pu1)
        push!(u1_labels, label)

        push!(u2_plots, pu2)
        push!(u2_labels, label)
    end

    axislegend(axL, surf_plots, surf_labels; position=:ct)
    axislegend(axL, int_plots, int_labels; position=:lb)

    axislegend(axu1, u1_plots, u1_labels;  position=:rb)
    axislegend(axu2, u2_plots, u2_labels; position=:rb)

    display(fig)
    return fig
end

function plot_ic_1d(ic, x)
    fig = Figure(size=(1700, 900), fontsize=18)
    axL, axu1, axu2 = make_axes_3panel(fig; title="Initial conditions at t = 0")

    color = Makie.wong_colors()[1]

    # plots
    pξ  = lines!(axL,  x, ic.ξ,  linewidth=2, color=color, linestyle=:solid)
    pω  = lines!(axL,  x, ic.ω,  linewidth=2, color=color, linestyle=:dash)
    pu1 = lines!(axu1, x, ic.u1, linewidth=2, color=color)
    pu2 = lines!(axu2, x, ic.u2, linewidth=2, color=color)

    # ---- special IC axis limits ----
    ylims!(axL, 0, 1.2)

    axislegend(axL,[pξ, pω],["ε", "w"];position=:rb)

    axislegend(axu1, [pu1], ["u₁"]; position=:rt)
    axislegend(axu2, [pu2], ["u₂"]; position=:rb)

    display(fig)
    return fig
end

function plot_corrected_resolution_comparison_1d(study; T=0.25, nxs=(800, 3200, 6400))
    fig = Figure(size=(1800, 1400), fontsize=18)

    Label(fig[0, 1:2],
          "Corrected schemes: grid refinement comparison at t = $T",
          fontsize=24)

    axL1 = Axis(fig[1:2, 1],
        title="old eig, corr — free surface ε and interface w",
        xlabel="x", ylabel="height")
    axu11 = Axis(fig[1, 2],
        title="old eig, corr — upper velocity u₁",
        xlabel="x", ylabel="u₁")
    axu12 = Axis(fig[2, 2],
        title="old eig, corr — lower velocity u₂",
        xlabel="x", ylabel="u₂")

    axL2 = Axis(fig[3:4, 1],
        title="new eig, corr — free surface ε and interface w",
        xlabel="x", ylabel="height")
    axu21 = Axis(fig[3, 2],
        title="new eig, corr — upper velocity u₁",
        xlabel="x", ylabel="u₁")
    axu22 = Axis(fig[4, 2],
        title="new eig, corr — lower velocity u₂",
        xlabel="x", ylabel="u₂")

    colors = Makie.wong_colors()

    surf1_plots = Any[]; surf1_labels = String[]
    int1_plots  = Any[]; int1_labels  = String[]
    u11_plots   = Any[]; u11_labels   = String[]
    u12_plots   = Any[]; u12_labels   = String[]

    surf2_plots = Any[]; surf2_labels = String[]
    int2_plots  = Any[]; int2_labels  = String[]
    u21_plots   = Any[]; u21_labels   = String[]
    u22_plots   = Any[]; u22_labels   = String[]

    for (i, nx) in enumerate(nxs)
        color = colors[mod1(i, length(colors))]
        lw = nx == maximum(nxs) ? 3 : 2
        lab = nx == maximum(nxs) ? "nx = $nx (ref)" : "nx = $nx"

        old = study["old eig, corr"][nx]
        new = study["new eig, corr"][nx]

        pξ1 = lines!(axL1,  old.x, old.fields.ξ,  linewidth=lw, color=color, linestyle=:solid)
        pω1 = lines!(axL1,  old.x, old.fields.ω,  linewidth=lw, color=color, linestyle=:dash)
        pu11 = lines!(axu11, old.x, old.fields.u1, linewidth=lw, color=color)
        pu12 = lines!(axu12, old.x, old.fields.u2, linewidth=lw, color=color)

        push!(surf1_plots, pξ1); push!(surf1_labels, "$lab: ε")
        push!(int1_plots,  pω1); push!(int1_labels,  "$lab: w")
        push!(u11_plots,  pu11); push!(u11_labels, lab)
        push!(u12_plots,  pu12); push!(u12_labels, lab)

        pξ2 = lines!(axL2,  new.x, new.fields.ξ,  linewidth=lw, color=color, linestyle=:solid)
        pω2 = lines!(axL2,  new.x, new.fields.ω,  linewidth=lw, color=color, linestyle=:dash)
        pu21 = lines!(axu21, new.x, new.fields.u1, linewidth=lw, color=color)
        pu22 = lines!(axu22, new.x, new.fields.u2, linewidth=lw, color=color)

        push!(surf2_plots, pξ2); push!(surf2_labels, "$lab: ε")
        push!(int2_plots,  pω2); push!(int2_labels,  "$lab: w")
        push!(u21_plots,  pu21); push!(u21_labels, lab)
        push!(u22_plots,  pu22); push!(u22_labels, lab)
    end

    axislegend(axL1, surf1_plots, surf1_labels; position=:lt)
    axislegend(axL1, int1_plots, int1_labels; position=:cb)
    axislegend(axu11, u11_plots, u11_labels; position=:ct)
    axislegend(axu12, u12_plots, u12_labels; position=:ct)

    axislegend(axL2, surf2_plots, surf2_labels; position=:lt)
    axislegend(axL2, int2_plots, int2_labels; position=:cb)
    axislegend(axu21, u21_plots, u21_labels; position=:ct)
    axislegend(axu22, u22_plots, u22_labels; position=:ct)

    display(fig)
    return fig
end

# ============================================================
# Drivers
# ============================================================

function run_four_case_study_1d(; nx=800, gc=2, T=0.25, cfl=0.45,
                                B=B0,
                                ρ1=0.98, ρ2=1.0, g=9.81,
                                UL=UL_test, UR=UR_test)

    results = Dict{String, Any}()
    xref, eqref = nothing, nothing

    for (label, eigmethod, corrflag) in FOUR_CASES
        println("Running case: $label")
        sim, x, eq = run_case_1d(
            nx=nx, gc=gc, T=T, cfl=cfl,
            B=B,
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

function run_corrected_resolution_study_1d(; nxs=(800, 3200, 6400), gc=2, T=0.25, cfl=0.8,
                                           B=B0,
                                           ρ1=0.99, ρ2=1.0, g=9.81,
                                           UL=UL_test, UR=UR_test)

    out = Dict(label => Dict{Int, Any}() for (label, _) in CORRECTED_CASES)

    for nx in nxs
        for (label, eigmethod) in CORRECTED_CASES
            println("Running $label at nx = $nx")
            sim, x, eq = run_case_1d(
                nx=nx, gc=gc, T=T, cfl=cfl,
                B=B,
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

"""
# Run 1:
study4 = run_four_case_study_1d(
    nx=200,
    gc=2,
    T=0.25,
    cfl=0.8,
    B=0.0,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)

save_plot(study4.fig_ic, "IC_1.png")
save_plot(study4.fig_final, "Correction_1.png")

study_corr = run_corrected_resolution_study_1d(
    nxs=(200, 1600, 12800),
    gc=2,
    T=0.25,
    cfl=0.8,
    B=0.0,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)
fig_corr = plot_corrected_resolution_comparison_1d(study_corr; T=0.25, nxs=(200, 1600, 12800))

save_plot(fig_corr, "Resolution_1.png")
"""

"""
# Run 2:
study4 = run_four_case_study_1d(
    nx=200,
    gc=2,
    T=0.25,
    cfl=0.8,
    B=0.0,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)

save_plot(study4.fig_ic, "IC_2.png")
save_plot(study4.fig_final, "Correction_2.png")
"""

study_corr = run_corrected_resolution_study_1d(
    nxs=(200, 1600, 12800),
    gc=2,
    T=0.25,
    cfl=0.4,
    B=0.0,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
)


fig_corr = plot_corrected_resolution_comparison_1d(
    study_corr;
    T=0.25,
    nxs=(200, 1600, 12800)
)

save_plot(fig_corr, "Resolution_2.png")


"""
# ============================================================
# Animation
# ============================================================

function run_case_timeseries_1d(; nx=100, gc=2, T=5.0, cfl=0.45,
                                B=B0,
                                ρ1=0.99, ρ2=1.0, g=9.81,
                                UL=UL_test, UR=UR_test,
                                eigenvalue_method=:new,
                                hyperbolicity_correction=true,
                                nframes=250)

    backend = SinFVM.make_cpu_backend()
    grid = SinFVM.CartesianGrid(nx; gc=gc, extent=[-1.0 1.0], boundary=SinFVM.NeumannBC())
    x = SinFVM.cell_centers(grid; interior=true)

    eq = make_equation_1d(
        B=B, ρ1=ρ1, ρ2=ρ2, g=g,
        eigenvalue_method=eigenvalue_method,
        hyperbolicity_correction=hyperbolicity_correction,
    )

    reconstruction = SinFVM.LinearLimiterReconstruction(SinFVM.MinmodLimiter(1.0))
    flux = SinFVM.PathConservativeCentralUpwind(eq)
    sources = [SinFVM.SourceTermBottom(), SinFVM.SourceTermNonConservative()]

    cs = SinFVM.ConservedSystem(backend, reconstruction, flux, eq, grid, sources)
    sim = SinFVM.Simulator(backend, cs, SinFVM.RungeKutta2(), grid; cfl=cfl)

    initial = [riemann_ic_w(xi, UL, UR, B) for xi in x]
    SinFVM.set_current_state!(sim, initial)

    times = range(0.0, T; length=nframes)

    frames = Vector{Any}(undef, length(times))
    frames[1] = extract_fields_1d(sim, eq)

    for k in 2:length(times)
        println("Animating frame $k / $(length(times))   (t = $(round(times[k], digits=3)))")
        SinFVM.simulate_to_time(sim, times[k])
        frames[k] = extract_fields_1d(sim, eq)
    end

    return (; x, times, frames, eq)
end

function animate_solution_1d(data; filename="two_layer_animation_n100_T5.mp4", folder=PLOT_DIR)
    mkpath(folder)
    path = joinpath(folder, filename)

    x = data.x
    times = data.times
    frames = data.frames

    ξmin = minimum(minimum(f.ξ) for f in frames)
    ξmax = maximum(maximum(f.ξ) for f in frames)

    ωmin = minimum(minimum(f.ω) for f in frames)
    ωmax = maximum(maximum(f.ω) for f in frames)

    u1min = minimum(minimum(f.u1) for f in frames)
    u1max = maximum(maximum(f.u1) for f in frames)

    u2min = minimum(minimum(f.u2) for f in frames)
    u2max = maximum(maximum(f.u2) for f in frames)

    fig = Figure(size=(1700, 1100), fontsize=18)

    title_obs = Observable("Two-layer solution at t = 0.00 s")
    Label(fig[0, 1:2], title_obs, fontsize=24)

    ax11 = Axis(fig[1, 1], title="Free surface ξ", xlabel="x", ylabel="ξ",
                limits=(minimum(x), maximum(x), ξmin, ξmax))
    ax12 = Axis(fig[1, 2], title="Interface ω", xlabel="x", ylabel="ω",
                limits=(minimum(x), maximum(x), ωmin, ωmax))
    ax21 = Axis(fig[2, 1], title="Upper-layer velocity u₁", xlabel="x", ylabel="u₁",
                limits=(minimum(x), maximum(x), u1min, u1max))
    ax22 = Axis(fig[2, 2], title="Lower-layer velocity u₂", xlabel="x", ylabel="u₂",
                limits=(minimum(x), maximum(x), u2min, u2max))

    ξ_obs  = Observable(frames[1].ξ)
    ω_obs  = Observable(frames[1].ω)
    u1_obs = Observable(frames[1].u1)
    u2_obs = Observable(frames[1].u2)

    lines!(ax11, x, ξ_obs, linewidth=3)
    lines!(ax12, x, ω_obs, linewidth=3)
    lines!(ax21, x, u1_obs, linewidth=3)
    lines!(ax22, x, u2_obs, linewidth=3)

    record(fig, path, collect(eachindex(times)); framerate=30) do i
        ξ_obs[]  = frames[i].ξ
        ω_obs[]  = frames[i].ω
        u1_obs[] = frames[i].u1
        u2_obs[] = frames[i].u2
        title_obs[] = "Two-layer solution at t = $(round(times[i], digits=3)) s"
    end

    println("Saved animation to: ", path)
    return fig
end

# ============================================================
# Run animation example
# ============================================================

anim_data = run_case_timeseries_1d(
    nx=100,
    gc=2,
    T=5.0,
    cfl=0.45,
    B=0.0,
    ρ1=0.99,
    ρ2=1.0,
    g=9.81,
    UL=UL_test,
    UR=UR_test,
    eigenvalue_method=:new,
    hyperbolicity_correction=true,
    nframes=250
)

animate_solution_1d(anim_data; filename="two_layer_animation_n100_T5.mp4")
"""