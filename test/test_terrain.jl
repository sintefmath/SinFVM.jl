using Vannlinje
using Rumpetroll
import Meshes
using Plots

terrain = loadgrid("testdata/bay.txt")
grid = Meshes.CartesianGrid(size(terrain) .- 1 .- 2*2)



initialdata = ConservedVariables(grid)

initialdata.h .= 0.5 .* identity(initialdata.h)


bathymetry = Bathymetry(grid)
bathymetry.Bi .= terrain

function callback(
    wdev,
    hudev,
    vdev,
    infiltration_rates_dev,
    B,
    dx,
    dy,
    Nx,
    Ny,
    t,
    Q_infiltrated,
    runoff,
)
    # @show collect(B)
    l = @layout [a b c]
    p1 = heatmap(collect(wdev).-collect(B), title="h($(t))")
    p2 = heatmap(collect(hudev), title="hu($(t))")
    p3 = heatmap(collect(vdev), title="hv($(t))")

    display(plot!(p1, p2, p3; layout=l, size=(2000, 600)))

end

T = 1000.

run_swe(grid, initialdata, bathymetry, T, Rumpetroll.lots_of_rain, callback)
display(heatmap(bathymetry.Bi, title="Bathymetry"))