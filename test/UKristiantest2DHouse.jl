using CairoMakie
using Cthulhu
using StaticArrays
using LinearAlgebra
using Test
import CUDA
#using GLMakie
#using Plots



module Correct
include("fasit.jl")
end
using SinFVM


function test_2D_dambreak_house(wall_height, wall_xpos, wall_ypos)
    backend = make_cpu_backend()

    backend_name = SinFVM.name(backend)
    nx = 128
    ny = 32
    grid = SinFVM.CartesianGrid(nx, ny, extent=[0 200; 0 28], gc=2, boundary=SinFVM.WallBC())
    x = SinFVM.cell_centers(grid)
    xf = SinFVM.cell_faces(grid)
    xVals = SinFVM.cell_centers(grid, XDIR)
    yVals = SinFVM.cell_centers(grid, YDIR)
    xFace = collect(SinFVM.cell_faces(grid, XDIR, interior=false))
    yFace = collect(SinFVM.cell_faces(grid, YDIR, interior=false))
    dx = SinFVM.compute_dx(grid)
    bottom_topography_array = zeros(size(grid) .+ (1, 1))
    @show size(yFace)
    @show size(bottom_topography_array, 1)
    #Insert wall
    #height = 1.0


    max_height = 5.0   # Maximum height at the top
    min_height = 0.0   # Minimum height at the bottom

    wall_width = 4

    #half_width_y = ny ÷ 4  # Half of the wall covering half of the y-axis

    # Loop through the 2D array and set the bottom topography
    #=for i in 1:size(bottom_topography_array, 1)
        for j in 1:size(bottom_topography_array, 2)
            # Linearly interpolate the height based on the x coordinate
            x_ratio = i / size(bottom_topography_array, 1)
            bottom_topography_array[i, j] = max_height - x_ratio * (max_height - min_height)
            #bottom_topography_array[i,j] += wall_height*exp(-(xFace[i]-wall_xpos)^2/30.3)
            if j >= wall_ypos - half_width_y && j <= wall_ypos + half_width_y
                bottom_topography_array[i,j] += wall_height*exp(-(xFace[i]-wall_xpos)^2/30.3)
            end
        end
    end=#
    for i in 1:size(bottom_topography_array, 1)
        for j in 1:size(bottom_topography_array, 2)
            # Linearly interpolate the height based on the x coordinate
            x_ratio = i / size(bottom_topography_array, 1)
            bottom_topography_array[i, j] = max_height - x_ratio * (max_height - min_height)

            # Add the wall contribution if within the specified range
            if abs(yFace[j] - wall_ypos) <= wall_width / 2
                bottom_topography_array[i, j] += wall_height * exp(-(xFace[i] - wall_xpos)^2 / 30.3)
            end
        end
    end
    # Create the heatmap
    fig = Figure(resolution=(800, 600))
    ax = Axis(fig[1, 1], title="2D Bottom Topography Heatmap", xlabel="X", ylabel="Y")
    hm = heatmap!(ax, xFace, yFace, bottom_topography_array, colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="Height")
    display(fig)
    #=
    x_values = collect(LinRange(1, 200, nx))
    y_values = collect(LinRange(1, 28, ny))
    #plot(heatmap(x_values, y_values, bottom_topography_array))
    f = Figure(size=(800, 800), fontsize=24)
        ax_B = Axis(
            f[1, 1]
        )
        hm = heatmap!(ax_B, x_values, y_values, bottom_topography_array)
        Colorbar(f[1, 2], hm)
        display(f)

    @show size(x_values)
    @show size(y_values)
    @show size(bottom_topography_array[3:end-2, 3:end-2])
    #=
    fig1 = Figure(resolution = (800, 800), fontsize = 24)

    # Create an axis
    ax = Axis3(fig1[1, 1], title = "Total Costs", xlabel = "x", ylabel = "y", zlabel = "z")

    # Plot the surface
    surface!(ax, xFace, yFace, bottom_topography_array[3:end-2, 3:end-2])

    # Set camera view
    ax.scene.camera = cameracontrols(ax.scene, :perspective)
    ax.scene.camera[] = LScene.Camera(perspective = true)
    campos!(ax.scene, Point3f0(45, 45, 45))
    lookat!(ax.scene, Point3f0(0, 0, 0))

    # Display the figure
    display(fig1)
    =#
    =#
    bottom_topography = SinFVM.BottomTopography2D(bottom_topography_array, backend, grid)
    bottom_source = SinFVM.SourceTermBottom()

    equation = SinFVM.ShallowWaterEquations(bottom_topography, depth_cutoff=10^-3, desingularizing_kappa=10^-3)
    reconstruction = SinFVM.LinearReconstruction(2)
    numericalflux = SinFVM.CentralUpwind(equation)
    friction = SinFVM.ImplicitFriction(friction_function=SinFVM.friction_bsa2012)
    conserved_system =
        SinFVM.ConservedSystem(backend, reconstruction, numericalflux, equation, grid, [bottom_source], friction)
    timestepper = SinFVM.RungeKutta2()
    simulator = SinFVM.Simulator(backend, conserved_system, timestepper, grid, cfl=0.45)
    T = 75

    function u0(x)
        if x[1] < 30
            return SVector(7, 0.0, 0.0)
        else
            return SVector(10^(-6), 0.0, 0.0)
        end
    end

    # presentasjon av optimering: plotte kostfunksjon sfa iterasjon, plotte benefit og cost sfa iterasjon

    # sett opp en bunntopografi som er skrå, som et dambrudd.
    # sett opp vegg i bunntopografien, huset står bak. 

    # veggen kan variere i høyde langs y-aksen, så det blir en 
    # f eks 8 celler i y-retning, da blir det optimering mhp 8 parametere
    # tomta er 1x1 celle


    initial = u0.(x)


    w_elements = [initial[i, j][1] for i in 1:size(initial, 1), j in 1:size(initial, 2)]



    g = Figure(size=(800, 800), fontsize=24)
    ax_C = Axis(
        g[1, 1]
    )
    hm2 = heatmap!(ax_C, xVals, yVals, w_elements)
    Colorbar(g[1, 2], hm2)
    display(g)



    SinFVM.set_current_state!(simulator, initial)

    #= 2) Via volumes:
    @show size(x)
    @show size(grid)
    init_volume = SinFVM.Volume(backend, equation, grid)
    CUDA.@allowscalar SinFVM.InteriorVolume(init_volume)[1:end, 1:end] = [SVector{3, Float64}(exp.(-(norm(xi .- 0.5)^2 / 0.01)) .+ 1.5, 0.0, 0.0) for xi in x]
    SinFVM.set_current_state!(simulator, init_volume)
    =#

    f = Figure(size=(1600, 1200), fontsize=24)
    names = [L"h", L"hu", L"hv"]
    titles = ["Initial, backend=$(backend_name)", "At time $(T), backend=$(backend_name)"]
    axes = [[Axis(f[i, 2*j-1], ylabel=L"y", xlabel=L"x", title="$(titles[j])\n$(names[i])") for i in 1:3] for j in 1:2]


    current_simulator_state = collect(SinFVM.current_state(simulator))
    @test !any(isnan.(current_simulator_state))

    initial_state = SinFVM.current_interior_state(simulator)
    hm = heatmap!(axes[1][1], xVals, yVals, collect(initial_state.h))
    Colorbar(f[1, 2], hm)
    hm = heatmap!(axes[1][2], xVals, yVals, collect(initial_state.hu))
    Colorbar(f[2, 2], hm)
    hm = heatmap!(axes[1][3], xVals, yVals, collect(initial_state.hv))
    Colorbar(f[3, 2], hm)

    t = 0.0
    @time SinFVM.simulate_to_time(simulator, T)
    #@test SinFVM.current_time(simulator) == T

    result = SinFVM.current_interior_state(simulator)
    h = collect(result.h)
    hu = collect(result.hu)
    hv = collect(result.hv)

    hm = heatmap!(axes[2][1], xVals, yVals, h)
    if !any(isnan.(h))
        Colorbar(f[1, 4], hm)
    end
    hm = heatmap!(axes[2][2], xVals, yVals, hu)
    if !any(isnan.(hu))
        Colorbar(f[2, 4], hm)
    end

    hm = heatmap!(axes[2][3], xVals, yVals, hv)
    if !any(isnan.(hv))
        Colorbar(f[3, 4], hm)
    end
    display(f)

    # Test symmetry (field[x, y])
    tolerance = 10^-13
    xleft = Int(floor(nx / 3))
    xright = nx - xleft + 1
    ylower = Int(floor(ny / 3))
    yupper = ny - ylower + 1
    # @show xleft, xright
    return h, bottom_topography.B
end

w2D, B2D = test_2D_dambreak_house(1.5, 140, 15)
