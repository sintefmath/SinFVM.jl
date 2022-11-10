using Vannlinje
using Rumpetroll
import Meshes
using Plots
using Printf
using Parameters

terrain = loadgrid("testdata/bay.txt")
grid = Meshes.CartesianGrid(size(terrain) .- 1 .- 2*2)



initialdata = ConservedVariables(grid)


bathymetry = Bathymetry(grid)
bathymetry.Bi .= terrain

@with_kw mutable struct IntervalWriter{WriterType}
    current_t::Float64 = 0.0
    step::Float64
    writer::WriterType
end

function (writer::IntervalWriter)(wdev,
    hudev,
    vdev,
    infiltration_rates_dev,
    B,
    dx,
    dy,
    dt,
    Nx,
    Ny,
    t,
    Q_infiltrated,
    runoff,)

    if t + dt >= current_t
        writer( hudev,
        vdev,
        infiltration_rates_dev,
        B,
        dx,
        dy,
        dt,
        Nx,
        Ny,
        t,
        Q_infiltrated,
        runoff,)
        current_t += step
    end



function callback(
    wdev,
    hudev,
    vdev,
    infiltration_rates_dev,
    B,
    dx,
    dy,
    dt,
    Nx,
    Ny,
    t,
    Q_infiltrated,
    runoff,
)

    tstr = @sprintf "%0.3f" t
    # @show collect(B)
    l = @layout [a b; c d]
    p0 = heatmap(collect(wdev), title="Total height w($(tstr)), dt=$(dt)")
    p1 = heatmap(collect(wdev) .- collect(B), title="h($(tstr)) = w($(tstr)) - B($(tstr))")
    p2 = heatmap(collect(hudev), title="hu($(tstr))")
    p3 = heatmap(collect(vdev), title="hv($(tstr))")

    display(plot!(p0, p1, p2,  p3; layout=l, size=(1600, 1200)))

end

T = 1000.
function infiltration(x, y, t)
    fc = 3.272e-5
    f0 = 1.977e-4
    k  = 2.43e-3 
    return fc + (f0 - fc)*exp(-k*t)
end 
rainy_day(x,y, t) = 0.000125
run_swe(grid, initialdata, bathymetry, T, rainy_day, callback; 
    infiltration_function =infiltration)
# display(heatmap(bathymetry.Bi, title="Bathymetry"))