import DelimitedFiles
using NPZ

using Rumpetroll
import Meshes
using ColorSchemes
using GLMakie
import Plots

using Printf
using Parameters
import Vannlinje
using ProgressMeter
mycmap1 = ColorSchemes.ColorScheme(
    [GLMakie.RGB{Float64}(0.933, 0.867,0.510),  # spill to outside
     GLMakie.RGB{Float64}(0.604, 0.804, 0.196), # catchment area
     GLMakie.RGB{Float64}(0.0, 0.0, 0.804),       # lakes
     GLMakie.RGB{Float64}(0.0, 0.545, 545)])    # rivers

mycmap2 = ColorSchemes.ColorScheme(
    [GLMakie.RGB{Float64}(0.604, 0.804, 0.196), # ground
     GLMakie.RGB{Float64}(0.0, 0.0, 0.804),     # filled lakes
     GLMakie.RGB{Float64}(0.933, 0.867,0.510),  # dry lakes
     GLMakie.RGB{Float64}(0.0, 0.545, 545),    # rivers
     GLMakie.RGB{Float64}(0.5, 0.5, 0.5)])   # buildings

function loadgrid(filename::String; delim=',')
    return DelimitedFiles.readdlm(filename, delim)
end

terrain = loadgrid("testdata/bay.txt")
upper_corner = Float64.(size(terrain))

coarsen_times  = 2

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
terrain = coarsen(terrain, coarsen_times)


grid = Meshes.CartesianGrid(size(terrain) .- 1 .- 2*2, (0.0,0.0), upper_corner)
(nx, ny) = size(grid)
w = npzread("data/w_100.npz")
w1 = copy(terrain)
w1[2:end,2:end] += w - 2*terrain[2:end,2:end]


w1 ./= maximum(w1)

# w1 .+= 1

threshold = 4.0
texture = Matrix{Float64}(collect(w1 .> threshold))
#Plots.display(Plots.heatmap(w1))

surf = Vannlinje.plotgrid(terrain, texture=texture, colormap=mycmap1)

@showprogress for i in 1:100000
    h = npzread("data/w_$(i)00.npz")
    h1 = copy(terrain)
    h1[2:end,2:end] += h - 2*terrain[2:end,2:end]


    # h1 ./= maximum(h1)

    # w1 .+= 1

    tex = Matrix{Float64}(collect(h1 .> threshold))
    Vannlinje.drape_surface(surf, tex)
    sleep(.1)
end

# w1 = zeros(nx+1)

# h = w - reshape(reshape(terrain))