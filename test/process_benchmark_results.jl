using CairoMakie
import CSV



f = Figure( size=(1600,600), fontsize=24)
ax = Axis(f[1, 1],
   
    title = "Wall runtime",
    ylabel = "Runtime [s]",
    xlabel = "Resolution [number of cells]",
    xscale= log2,
    yscale= log2,
)

ax2 = Axis(f[1, 2],
   
    title = "Speedup: (runtime CPU)/(runtime GPU)",
    xlabel = "Resolution",
    ylabel = "Speedup",
    xscale= log2,
    yscale= log10,
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
        runtime = Float64(row[Symbol("time_$(k)")])
        resolution =  Int64(row[Symbol("resolution")])

        push!(runtime_per_timestep, runtime)
        push!(resolutions, resolution)

        push!(resolutions_all[k], resolution)
        push!(runtimes_all[k], runtime)
    end

    lines!(f[1,1], resolutions, runtime_per_timestep, label=labels[k])
    scatter!(f[1,1], resolutions, runtime_per_timestep)
end
axislegend(ax, position=:lt)

lines!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"])
scatter!(f[1,2], resolutions_all["swe"], runtimes_all["bb"]./runtimes_all["swe"])

println("Done")
save("benchmark.png", f)
display(f)
