import Vannlinje
using NPZ

using Rumpetroll
import Meshes
using Plots
pyplot()
using Printf
using Parameters


terrain = Vannlinje.loadgrid("testdata/bay.txt")
upper_corner = Float64.(size(terrain))

coarsen_times  = 3

function coarsen(data, times)
    for c in 1:times
        old_size = size(data)
        newsize = floor.(Int64, old_size./2)
        data_new = zeros(newsize)

        for i in CartesianIndices(data_new)
            for j in max(1, 2*(i[1]-1)):min(old_size[1], 2*(i[1]-1)+1)
                for k in max(1, 2*(i[2]-1)):min(old_size[2], 2*(i[2]-1)+1)
                    data_new[i] += data[CartesianIndex((j, k))] / 4
                end
            end
        end
        data = data_new
    end
    return data
end
p1 = heatmap(terrain, title="original")
terrain = coarsen(terrain, coarsen_times)
p2 = heatmap(terrain, title="Coarsened")


l = @layout [a b]
display(plot(p1, p2, layout=l, size=(1200,400)))
grid = Meshes.CartesianGrid(size(terrain) .- 1 .- 2*2, (0.0,0.0), upper_corner)
@show grid



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

    if t + dt >= writer.current_t
        writer.writer(
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
        round(writer.current_t/writer.step)*writer.step,
        Q_infiltrated,
        runoff,)
        writer.current_t += writer.step
    end

end
mkpath("figs")
mkpath("data")
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

    p = plot(p0, p1, p2,  p3; layout=l, size=(1600, 1200))
    tstr_print = @sprintf "%d" floor(Int64, t)

    savefig(p, "figs/plot_$(tstr_print).png")

    closeall()
    npzwrite("data/w_$(tstr_print).npz", collect(wdev))
    npzwrite("data/hu_$(tstr_print).npz", collect(hudev))
    npzwrite("data/hv_$(tstr_print).npz", collect(vdev))

    println("Current t=$(tstr), dt=$(dt).")
    
end

T = 1000000.

# @Odd: cut down on time to limit runtime for testing
T = T/5    
    
function infiltration(x, y, t)
    fc = 3.272e-5
    f0 = 1.977e-4
    k  = 2.43e-3 
    return fc + (f0 - fc)*exp(-k*t)
end 
rainy_day(x,y, t) = 0.000125
run_swe(grid, initialdata, bathymetry, T, rainy_day, IntervalWriter(step=100.0, writer=callback); 
    infiltration_function =infiltration)
# display(heatmap(bathymetry.Bi, title="Bathymetry"))
