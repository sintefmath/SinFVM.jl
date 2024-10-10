using CairoMakie
using Cthulhu
using StaticArrays

using SinSWE
function conv(v::Vector{T}, kernel::Vector{T}) where {T}
    n = length(v)
    m = length(kernel)
    padded_v = [zeros(T, m ÷ 2); v; zeros(T, m ÷ 2)]
    result = zeros(T, n)
    for i = 1:n
        result[i] = sum(padded_v[i:i+m-1] .* kernel)
    end
    return result
end
function mollify_vector(v::Vector{T}; σ::Float64 = 1.0) where {T}
    n = length(v)
    kernel = exp.(-((1:n) .- (n + 1) / 2) .^ 2 / (2σ^2))
    kernel /= sum(kernel)
    return conv(v, kernel)
end

function run_simulation(sigmas)
    f = Figure(size = (2200, length(sigmas) * 600), fontsize = 24)
    f[1, 1:3] = Label(f, "Different smoothness of bottom topography.\nHere h is no reconstruction, while h_2 is linear reconstruction (minmod)", fontsize = 32, tellheight = true)

    for (sigma_n, sigma) in enumerate(sigmas)
        u0 = x -> @SVector[0.0, 0.0]
        nx = 1024
        grid = SinSWE.CartesianGrid(nx; gc = 2)
        # backend = make_cuda_backend()
        backend = make_cpu_backend()

        bottom_topography = zeros(Float64, nx + 5)
        bottom_topography[3nx÷8:5nx÷8] .= 50

        bottom_topography = mollify_vector(bottom_topography, σ = sigma)

        fig_topo = Figure(size = (800, 400), fontsize = 24)
        ax_topo = Axis(
            fig_topo[1, 1],
            title = "Bottom Topography",
            ylabel = "Elevation",
            xlabel = L"x",
        )
        lines!(
            ax_topo,
            1:length(bottom_topography),
            bottom_topography,
            label = "Bottom Topography",
        )
        axislegend(ax_topo, position = :lt)
        display(fig_topo)
        # bottom_topography[nx ÷ 4:3nx ÷ 4] .= 4.0

        infiltration_map = 0.1 * ones(Float64, nx + 4)
        infiltration_map[3nx÷8:5nx÷8] .= 0.0

        bottom_topography_backend =
            SinSWE.BottomTopography1D(bottom_topography, backend, grid)
        bottom_source = SinSWE.SourceTermBottom()
        equation = SinSWE.ShallowWaterEquations1D(bottom_topography_backend)
        reconstruction = SinSWE.NoReconstruction()
        linrec = SinSWE.LinearReconstruction(1.05)
        numericalflux = SinSWE.CentralUpwind(equation)
        rain_source = SinSWE.ConstantRain(0.01)  # Define a rain source term with a rate of 0.01
        infiltration_source =
            SinSWE.HortonInfiltration(grid, backend; factor = infiltration_map)
        friction = SinSWE.ImplicitFriction()  # Define a friction term
        conserved_system = SinSWE.ConservedSystem(
            backend,
            reconstruction,
            numericalflux,
            equation,
            grid,
            [rain_source, bottom_source, infiltration_source],
            friction
        )

        timestepper = SinSWE.ForwardEulerStepper()
        linrec_conserved_system = SinSWE.ConservedSystem(
            backend,
            linrec,
            numericalflux,
            equation,
            grid,
            [rain_source, bottom_source, infiltration_source],
            friction
        )
        x = SinSWE.cell_centers(grid)
        initial = u0.(x)
        T = 1000.0


        ax = Axis(
            f[sigma_n + 1, 1],
            title = L"h\;\mathrm{ at\;time }\;%$(T)",
            ylabel = L"h",
            xlabel = L"x",
        )
        ax1 = Axis(f[sigma_n + 1, 2], title = L"$w = h + B$", ylabel = L"w = h + B", xlabel = L"x")
        ax2 = Axis(f[sigma_n + 1, 3], title = L"\mathrm{Bottom\;topography }\;B_{%$(sigma)}", ylabel = L"B", xlabel = L"x")

        simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid)
        linrec_simulator =
            SinSWE.Simulator(backend, linrec_conserved_system, timestepper, grid; cfl = 0.2)

        SinSWE.set_current_state!(linrec_simulator, initial)
        SinSWE.set_current_state!(simulator, initial)

        initial_state = SinSWE.current_interior_state(simulator)
        lines!(ax, x, collect(initial_state.h), label = L"h_0(x)")

        t = 0.0

        @time SinSWE.simulate_to_time(simulator, T; maximum_timestep = 0.1)
        @time SinSWE.simulate_to_time(linrec_simulator, T; maximum_timestep = 0.1)

        # conserved_new = SinSWE.ConservedSystem(
        #     backend,
        #     reconstruction,
        #     numericalflux,
        #     equation,
        #     grid,
        #     [bottom_source]
        # )
        # simulator_new = SinSWE.Simulator(backend, conserved_new, timestepper, grid)
        # SinSWE.set_current_state!(simulator_new, SinSWE.current_state(simulator))
        # SinSWE.simulate_to_time(simulator_new, 100.0; maximum_timestep = 0.1)

        result = SinSWE.current_interior_state(simulator)
        linrec_results = SinSWE.current_interior_state(linrec_simulator)

        B_cell(index) = SinSWE.B_cell(equation.B, index)

        bottom_averages = B_cell.((1:nx) .+ 2)
        lines!(
            ax,
            x,
            collect(result.h) .- bottom_averages,
            linestyle = :dot,
            color = :red,
            linewidth = 8,
            label = L"h^{\Delta x}(x, t)",
        )

        lines!(
            ax1,
            x,
            collect(result.h),
            linestyle = :dot,
            color = :red,
            linewidth = 8,
            label = L"w^{\Delta x}(x, t)",
        )
        lines!(
            ax2,
            x,
            bottom_averages,
            linestyle = :dashdot,
            color = :red,
            linewidth = 8,
            label = L"B",
        )

        lines!(
            ax1,
            x,
            collect(linrec_results.h),
            linestyle = :dot,
            color = :orange,
            linewidth = 4,
            label = L"w_2^{\Delta x}(x, t)",
        )
        lines!(
            ax,
            x,
            collect(linrec_results.h) .- bottom_averages,
            linestyle = :dot,
            color = :orange,
            linewidth = 4,
            label = L"h_2^{\Delta x}(x, t)",
        )
        axislegend(ax, position = :lt)
        axislegend(ax1, position = :lt)
        axislegend(ax2, position = :lt)


        @show sum(abs.(collect(result.h) - collect(linrec_results.h)))
        @show sum(abs.(collect(result.hu) - collect(linrec_results.hu)))
    end
    save("somefig.png", f)
    display(f)

end

run_simulation([0.01, 5, 20.0, 50.0, 100.0])
