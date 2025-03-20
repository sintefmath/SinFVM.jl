# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Parameters
using Printf
using NPZ
function loadgrid(filename::String; delim=',')
    return DelimitedFiles.readdlm(filename, delim)
end


function coarsen(data, times)
    for c in 1:times
        old_size = size(data)
        newsize = floor.(Int64, old_size ./ 2)
        data_new = zeros(newsize)

        for i in CartesianIndices(data_new)
            for j in max(1, 2 * (i[1] - 1)):min(old_size[1], 2 * (i[1] - 1) + 1)
                for k in max(1, 2 * (i[2] - 1)):min(old_size[2], 2 * (i[2] - 1) + 1)
                    data_new[i] += data[CartesianIndex((j, k))] / 4
                end
            end
        end
        data = data_new
    end
    return data
end



@with_kw struct TotalWaterVolume
    water_volume::Vector{Float64} = Float64[]
    times::Vector{Float64} = Float64[]
    bottom_topography
end

function bottom_per_cell(bottom_topography::SinFVM.BottomTopography2D)
    bottom_per_cell(bottom_topography.B)
end

function bottom_per_cell(bottom_topography::Array)
    dimensions = size(bottom_topography) .- (1, 1)
    data = zeros(dimensions)

    bottom_topography = SinFVM.BottomTopography2D(collect(bottom_topography); should_never_be_called=1)
    for j in 1:dimensions[2]
        for i in 1:dimensions[1]
            data[i, j] = SinFVM.B_cell(bottom_topography, i, j)
        end
    end

    return data
end

function bottom_per_cell(bottom_topography::SinFVM.BottomTopography2D)
    dimensions = size(bottom_topography.B) .- (1, 1)
    data = zeros(dimensions)

    bottom_topography = SinFVM.BottomTopography2D(collect(bottom_topography.B); should_never_be_called=1)
    for j in 1:dimensions[2]
        for i in 1:dimensions[1]
            data[i, j] = SinFVM.B_cell(bottom_topography, i, j)
        end
    end

    return data
end
function (callback::TotalWaterVolume)(t, simulator)
    bottom_avg = bottom_per_cell(callback.bottom_topography)
    h = collect(SinFVM.current_interior_state(simulator).h) .- bottom_avg[3:end-2, 3:end-2]
    push!(callback.water_volume, sum(h))
    push!(callback.times, t)
    mkpath("figs/bay/$(basename)")

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        lines!(ax1, callback.times, callback.water_volume, label="Remaining water", colorrange=(0.0, 3.0))
        scatter!(ax1, callback.times, callback.water_volume,)
        lines!(ax1, callback.times, prod(size(h)) .* callback.times .* 15 / (1000.0), label="Total rained")
        axislegend(ax1, position=:lt)
        if isfile("figs/bay/total_water.png")
            cp("figs/bay/total_water.png", "figs/bay/total_water_old.png", force=true)
        end
        save("figs/bay/total_water.png", f, px_per_unit=2)

    end
end

mkpath("figs")
mkpath("data")



function callback(bottom, basename, t, simulator)

    state = SinFVM.current_interior_state(simulator)
    tstr = string(t)
    w = collect(state.h)
    h = w .- bottom[3:end-3, 3:end-3]
    hu = collect(state.hu)
    hv = collect(state.hv)
    mkpath("figs/bay/$(basename)")

    with_theme(theme_latexfonts()) do
        f = Figure()
        ax1 = Axis(f[1, 1])
        hm = heatmap!(ax1, collect(h), label="Time t=$(tstr)", colorrange=(0.0, 3.0))
        Colorbar(f[:, end+1], hm)
        save("figs/bay/$(basename)/state_$(tstr).png", f, px_per_unit=2)
    end

    tstr_print = "$(ceil(Int64, t))"

    mkpath("data/bay/$(basename)")
    npzwrite("data/bay/$(basename)/w_$(tstr_print).npz", h)
    npzwrite("data/bay/$(basename)/hu_$(tstr_print).npz", hu)
    npzwrite("data/bay/$(basename)/hv_$(tstr_print).npz", hv)
end
