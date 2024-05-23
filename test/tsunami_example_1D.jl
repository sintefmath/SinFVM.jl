using SinSWE
using Test
using StaticArrays
using CairoMakie

# using GLMakie

function tsunami(;T=10, dt=1, w0_height=1.0, bump=false)

    nx = 1024
    grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=[0.0  1000.0], )
    function terrain(x)
        b =  25/1000*(x - 800)
        if bump
            b += 8*exp(-(x - 500)^2/100)
        end
        return b
    end
    
    B_data = Float64[terrain(x) for x in SinSWE.cell_faces(grid, interior=false)]

    backend = make_cpu_backend()
    B = SinSWE.BottomTopography1D(B_data, backend, grid)
    eq = SinSWE.ShallowWaterEquations1D(B; depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    rec = SinSWE.LinearReconstruction(2)
    flux = SinSWE.CentralUpwind(eq)
    bst = SinSWE.SourceTermBottom()
    conserved_system = SinSWE.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = SinSWE.BalanceSystem(conserved_system, bst)
    
    
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid, cfl=0.1)
    
    x = SinSWE.cell_centers(grid)
    xf = SinSWE.cell_faces(grid)
    u0 = x -> @SVector[w0_height*(x < 100), 0.0]
    initial = u0.(x)

    SinSWE.set_current_state!(simulator, initial)
    all_t = []
    all_h = []
    all_hu = []
    t = 0.0
    @time while t < T
        t += dt
        SinSWE.simulate_to_time(simulator, dt)
        
        state = SinSWE.current_interior_state(simulator)
        push!(all_h, collect(state.h))
        push!(all_hu, collect(state.hu))
        push!(all_t, t)
    end

    index = Observable(1)
    h = @lift(all_h[$index])
    hu = @lift(all_hu[$index])
    t = @lift(all_t[$index])

    f = Figure(size=(800, 800), fontsize=24)
    infostring = @lift("tsunami, t=$(all_t[$index]) nx=$(nx)") #\n$(split(typeof(rec),".")[2]) and $(split(typeof(flux), ".")[2])"
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

        record(f, "tsunami.mp4", 1:length(all_t); framerate=10) do i 
            index[] =i                
        end
        #display(f)
end

tsunami(T=200; bump=true)
println("done")
nothing