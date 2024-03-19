using SinSWE
using Test
using StaticArrays
using CairoMakie

function test_lake_at_rest(grid, B, w0; plot=true)

    backend = make_cpu_backend()
    eq = SinSWE.ShallowWaterEquations1D(B)
    rec = SinSWE.LinearReconstruction(2)
    flux = SinSWE.CentralUpwind(eq)
    bst = SinSWE.SourceTermBottom()
    conserved_system = SinSWE.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = SinSWE.BalanceSystem(conserved_system, bst)
    
    
    timestepper = SinSWE.ForwardEulerStepper()
    simulator = SinSWE.Simulator(backend, conserved_system, timestepper, grid, cfl=0.25)
    
    x = SinSWE.cell_centers(grid)
    xf = SinSWE.cell_faces(grid)
    u0 = x -> @SVector[w0, 0.0]
    initial = u0.(x)

    SinSWE.set_current_state!(simulator, initial)
    
    t = 0.001
    SinSWE.simulate_to_time(simulator, t)
    w = first.(SinSWE.current_interior_state(simulator))
    hu = map(x -> x[2], SinSWE.current_interior_state(simulator))
        
    if plot
        f = Figure(size=(800, 800), fontsize=24)
        infostring = "lake at rest \nt=$(t) nx=$(nx)\n$(typeof(rec)) and $(typeof(flux))"
        ax_h = Axis(
            f[1, 1],
            title="water and terrain"*infostring,
            ylabel="y",
            xlabel="x",
        )

        lines!(ax_h, x, w, color="blue", label='w')
        lines!(ax_h, xf, B[3:end-2], label='B', color="red")

        ax_u = Axis(
            f[2, 1],
            title="hu",
            ylabel="hu",
            xlabel="x",
        )

        lines!(ax_u, x, hu, label="hu")

        # axislegend(ax_h)
        # axislegend(ax_u)
        display(f)
    end
    @test maximum(abs.(hu)) ≈ 0.0 atol=10^-15
    @test maximum(abs.(w[1] - w0)) ≈ 0.0 atol=10^-15
end

nx = 32
grid = SinSWE.CartesianGrid(nx; gc=2, boundary=SinSWE.WallBC(), extent=[0.0  10.0], )
x0 = 5.0
#step_bottom = x-> x < x0 ? @SVector[0.45+0.0*x] : @SVector[0.55 + 0.0*x ]
#B = step_bottom.(SinSWE.cell_centers(grid))
B = [x < x0 ? 0.45 : 0.55 for x in SinSWE.cell_faces(grid, interior=false)]
test_lake_at_rest(grid, B, 0.7, plot=true)


# bst = SinSWE.SourceTermBottom()
# rain = SinSWE.SourceTermRain(1.0)
# infl = SinSWE.SourceTermInfiltration(-1.0)

# v_st::Vector{SinSWE.SourceTerm} = [bst, rain, infl]
# @show(v_st)
@show maximum(B)