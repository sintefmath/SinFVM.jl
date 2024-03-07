using CairoMakie
import CSV



f = Figure( size=(1600,600), fontsize=24)
ax = Axis(f[1, 1],
   
    title = "Runtime per timestep",
    ylabel = "Runtime per timestep [s]",
    xlabel = "Resolution [number of cells]",
    xscale= log2,
    yscale= log2,
)

ax2 = Axis(f[1, 2],
   
    title = "Speedup: (runtime CPU)/(runtime GPU)",
    xlabel = "Resolution",
    ylabel = "Speedup",
    xscale= log2,
    yscale= log2,
)

runtimes_all = Dict("swe" =>  Float64[], "bb" =>  Float64[])
resolutions_all = Dict("swe" =>  Int64[], "bb" =>  Int64[])
use_keys = ["swe", "bb"]
labels = Dict("swe" => "GPU", "bb" => "Barbones CPU (single core)")
for k in use_keys
    runtime_per_timestep = Float64[]
    resolutions = Int64[]
    csvfile = CSV.File("results_cuda.txt")

    for row in csvfile
        runtime = Float64(row[Symbol("time_$(k)")])/Float64(row[Symbol("timesteps_$(k)")])
        resolution =  Int64(row[Symbol("resolution")])

        push!(runtime_per_timestep, runtime)
        push!(resolutions, resolution)

        push!(resolutions_all[k], resolution)
        push!(runtimes_all[k], runtime)
    end

    lines!(f[1,1], resolutions, runtime_per_timestep, label=labels[k])
    scatter!(f[1,1], resolutions, runtime_per_timestep)

    if k == "bb"
        lines!(f[1,1], resolutions, runtime_per_timestep ./ 16, linestyle=:dash, label="Idealized 16 core run")
        scatter!(f[1,1], resolutions, runtime_per_timestep ./ 16)

        lines!(f[1,1], resolutions, runtime_per_timestep ./ 4, linestyle=:dashdot, label="'Realistic' 4 core run")
        scatter!(f[1,1], resolutions, runtime_per_timestep ./ 4)
    end
end
axislegend(ax, position=:lt)

lines!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"], label="Against single core")
scatter!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"])

lines!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"]./16, linestyle=:dash, label="Against Idealized 16 core")
scatter!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"]./16)

lines!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"]./4, linestyle=:dashdot, label="Against 'realistic' 4 core")
scatter!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"]./4)

axislegend(ax2, position=:lt)

println("Done")
save("benchmark.png", f)
display(f)
