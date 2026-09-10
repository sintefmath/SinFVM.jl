using BenchmarkTools

include(joinpath(@__DIR__, "..", "src", "SinFVM.jl"))

function benchmark_array_update(; n = 1024, backend = VolumeFluxes.make_cpu_backend(Float32))
    data = rand(Float32, n, n)
    device_data = VolumeFluxes.convert_to_backend(backend, data)

    trial = @benchmark begin
        tmp = copy($device_data)
        tmp .+= 1.0f0
    end
    return minimum(trial.times) / 1e6
end

function benchmark_cpu_vs_metal(; n = 1024)
    cpu_time = benchmark_array_update(; n=n, backend=VolumeFluxes.make_cpu_backend(Float32))
    if VolumeFluxes.has_metal_backend()
        metal_time = benchmark_array_update(; n=n, backend=VolumeFluxes.make_metal_backend(Float32))
        return (; cpu = cpu_time, metal = metal_time)
    end
    return (; cpu = cpu_time, metal = missing)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = benchmark_cpu_vs_metal()
    println("CPU: $(result.cpu) ms")
    if ismissing(result.metal)
        println("Metal: unavailable on this machine")
    else
        println("Metal: $(result.metal) ms")
        println("Speedup: $(result.cpu / result.metal)")
    end
end
