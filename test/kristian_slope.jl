using SinFVM
using Test
using StaticArrays
using CairoMakie

# using GLMakie


function simpleDamBreakHouse(; T=100, dt=1, w0_height=1.0, bump=false)

    nx = 256
    grid = SinFVM.CartesianGrid(nx; gc=2, boundary=SinFVM.WallBC(), extent=[0.0 300],)
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    xGC = SinFVM.cell_centers(grid, interior=false)
    xfGC = SinFVM.cell_faces(grid, interior=false)
    bottom_topography_1D = zeros(size(xGC, 1) + 1)
    max_height = 5
    min_height = 0
    wall_position = 140
    wall_height = 1.5
    for i in 1:size(bottom_topography_1D, 1)
        # Linearly interpolate the height based on the x coordinate
        x_ratio = i / nx
        bottom_topography_1D[i] = max_height - x_ratio * (max_height - min_height)
        bottom_topography_1D[i] += wall_height * exp(-(xfGC[i] - wall_position)^2 / 30.3)
    end

    backend = make_cpu_backend()
    B = SinFVM.BottomTopography1D(bottom_topography_1D, backend, grid)
    eq = SinFVM.ShallowWaterEquations1D(B; depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_bsa2012)
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, [bst])
    #balance_system = SinFVM.BalanceSystem(conserved_system, bst)


    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.01)


    function u0(x)
        if x < 30
            @SVector[7, 0.0]
        else
            @SVector[10^(-6), 0.0]
        end
    end

    initial = u0.(x)


    SinFVM.set_current_state!(simulator, initial)
    all_t = []
    all_h = []
    all_hu = []
    t = 0.0

    w_data = SinFVM.current_interior_state(simulator).h

    fig1 = Figure()
    ax = Axis(fig1[1, 1], title="Simple Plot", xlabel="Grid", ylabel="B values")

    # Plot the data
    lines!(ax, x, w_data, color="blue", label="Initial h")
    lines!(ax, xf, collect(B.B[3:end-2]), color="red", label="B")

    # Display the figure
    display(fig1)

    @time while t < T
        t += dt
        SinFVM.simulate_to_time(simulator, t)

        state = SinFVM.current_interior_state(simulator)
        push!(all_h, collect(state.h))
        push!(all_hu, collect(state.hu))
        push!(all_t, t)

    end

    index = Observable(1)
    @show index

    #@show all_h[$index]
    #@show all_hu[$index]
    #@show all_t[$index]
    h = @lift(all_h[$index])
    hu = @lift(all_hu[$index])
    t = @lift(all_t[$index])
    final_w = collect(SinFVM.current_interior_state(simulator).h)

    f = Figure(size=(800, 800), fontsize=24)
    infostring = @lift("simpleDamBreak, t=$(all_t[$index]) nx=$(nx)") #\n$(split(typeof(rec),".")[2]) and $(split(typeof(flux), ".")[2])"
    ax_h = Axis(
        f[1, 1],
        title=infostring,
        ylabel="h",
        xlabel="x",
    )

    lines!(ax_h, x, h, color="blue", label='w')
    lines!(ax_h, xf, collect(B.B[3:end-2]), label='B', color="red")

    ax_u = Axis(
        f[2, 1],
        title="hu",
        ylabel="hu",
        xlabel="x",
    )
    ylims!(ax_u, -3, 8)

    lines!(ax_u, x, hu, label="hu")

    record(f, "simpleDamBreak.mp4", 1:length(all_t); framerate=10) do i
        index[] = i
    end
    #display(f)

    return final_w, collect(B.B[3:end-2]), x, xf
end


h1Dhouse, B1Dhouse, x1Dhouse, xf1Dhouse = simpleDamBreakHouse(T=500; w0_height=4.0, bump=true)
println("done")


# Create a figure and axis
fig = Figure()
ax = Axis(fig[1, 1], title="Simple Plot", xlabel="Grid", ylabel="B values")

# Plot the data
lines!(ax, xf1Dhouse, B1Dhouse)
lines!(ax, x1Dhouse, h1Dhouse)

# Display the figure
display(fig)