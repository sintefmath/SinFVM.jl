using SinFVM
using Test
using StaticArrays
using CairoMakie
using LinearAlgebra


function test_lake_at_rest(backend, grid, B_data, w0, t=0.001; plot=true)

    B = SinFVM.BottomTopography2D(B_data, backend, grid)
    eq = SinFVM.ShallowWaterEquations(B)
    rec = SinFVM.LinearReconstruction(2)
    flux = SinFVM.CentralUpwind(eq)
    bst = SinFVM.SourceTermBottom()
    conserved_system = SinFVM.ConservedSystem(backend, rec, flux, eq, grid, bst)
    
    #balance_system = SinFVM.BalanceSystem(conserved_system, bst)
    
    
    timestepper = SinFVM.ForwardEulerStepper()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.2)
    
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    u0 = x -> @SVector[w0, 0.0, 0.0]
    initial = u0.(x)

    SinFVM.set_current_state!(simulator, initial)
    
    SinFVM.simulate_to_time(simulator, t)

    
    # initial_state = SinFVM.current_interior_state(simulator)
    # lines!(ax, x, collect(initial_state.h), label=L"h_0(x)")
    # lines!(ax2, x, collect(initial_state.hu), label=L"hu_0(x)")

    state = SinFVM.current_interior_state(simulator)
    w =  collect(state.h)
    hu = collect(state.hu)
    hv = collect(state.hv)
        
    if plot
        infostring = "lake at rest \nt=$(t) nx=$(nx)\n$(typeof(rec)) and $(typeof(flux))"
        f = Figure(size=(800, 800), fontsize=24)
        ax_B = Axis(
            f[1, 1],
            title=infostring*"\nterrain",
        )
        hm = heatmap!(ax_B, collect(B.B[3:end-2, 3:end-2]))
        Colorbar(f[1, 2], hm)
        ax_w = Axis(
            f[1, 3],
            title="w",
        )
        hm = heatmap!(ax_w, w)
        Colorbar(f[1, 4], hm)
        ax_hu = Axis(
            f[2, 1],
            title="hu (max = $(round(maximum(abs.(hu)); digits=18)))",
        )
        hm = heatmap!(ax_hu, hu)
        Colorbar(f[2, 2], hm)
        ax_hv = Axis(
            f[2, 3],
            title="hv (max = $(round(maximum(abs.(hv)); digits=18)))",
        )
        hm = heatmap!(ax_hv, hv)
        Colorbar(f[2, 4], hm)
    
        # axislegend(ax_h)
        # axislegend(ax_u)
        display(f)
    end
    @test maximum(abs.(hu)) ≈ 0.0 atol=10^-14
    @test maximum(abs.(w[1] - w0)) ≈ 0.0 atol=10^-14
end


for backend in SinFVM.get_available_backends()
    nx = 64
    ny = 64
    grid = SinFVM.CartesianGrid(nx, ny; gc=2, boundary=SinFVM.WallBC(), extent=[0.0  10.0; 0.0 10.0], )
    x0 = 5.0
    y0 = 5.0
    B = [x[1] < x0 && x[2] < y0 ? 0.45 : 0.55 for x in SinFVM.cell_faces(grid, interior=false)]

    nx_bumpy = 124
    ny_bumpy = 124
    grid_bumpy = SinFVM.CartesianGrid(nx_bumpy, ny_bumpy; gc=2, boundary=SinFVM.WallBC(), extent=[-2*pi  2*pi; -2*pi 2*pi], )
    B_bumpy = [(cos(x[1] + x[2])-0.5 - 1.5*(norm(x) < 1.0)) for x in SinFVM.cell_faces(grid_bumpy, interior=false)]

   @testset "lake_at_rest_$(SinFVM.name(backend))" begin

        test_lake_at_rest(backend, grid, B, 0.7, 0.01, plot=false)
        test_lake_at_rest(backend, grid_bumpy, B_bumpy, 0.7, 0.01, plot=false)
   end
end

# bst = SinFVM.SourceTermBottom()
# rain = SinFVM.SourceTermRain(1.0)
# infl = SinFVM.SourceTermInfiltration(-1.0)

# v_st::Vector{SinFVM.SourceTerm} = [bst, rain, infl]
# @show(v_st)
#@show maximum(B)  