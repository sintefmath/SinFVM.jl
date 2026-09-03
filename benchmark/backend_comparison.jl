# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# CPU vs Apple GPU (Metal) throughput for a 2D shallow water simulation.
#
# Run from the repository root:
#
#     julia --project -t auto benchmark/backend_comparison.jl
#     julia --project -t auto benchmark/backend_comparison.jl --quick
#
# `-t auto` matters: the KernelAbstractions CPU backend is thread parallel and Julia
# defaults to a single thread, which would flatter the GPU by a large factor.
#
# Three configurations are timed. CPU/Float32 is not redundant -- reporting it next to
# CPU/Float64 separates the speedup that comes from the GPU from the speedup that comes
# from halving the precision, which is the only precision Metal supports.
#
# NOTE: Metal is reached as `VolumeFluxes.make_metal_backend()` rather than by importing
# Metal here. `@fvmloop` (src/meta/loops.jl) decides which locals become kernel arguments
# by reflecting over `names(Main)`, so putting extra exports in `Main` is a habit worth
# avoiding in this repository even where, as here, it happens to be harmless.

import KernelAbstractions
using Printf
using StaticArrays
using LinearAlgebra
using VolumeFluxes

const VF = VolumeFluxes

# Hold cell-updates roughly constant across resolutions instead of step count, so every
# (backend, size) pair does comparable work and the whole sweep stays inside its budget.
const TARGET_CELL_UPDATES = 5e7
const MIN_STEPS = 3
const MAX_STEPS = 2000
const REPEATS = 3
# Once a configuration takes longer than this for one timed run, stop growing the grid for
# it. The CPU/Float64 path at the largest sizes is the long pole.
const TIME_CAP_SECONDS = 25.0

quick = "--quick" in ARGS
const SIZES = quick ? [64, 128, 256, 512] : [64, 128, 256, 512, 1024, 2048]

steps_for(ncells) = clamp(round(Int, TARGET_CELL_UPDATES / ncells), MIN_STEPS, MAX_STEPS)

"""
    build_simulation(backend, n)

A 2D shallow water simulation on an `n` x `n` grid: `ShallowWaterEquationsPure` with
linear reconstruction, a central upwind flux and forward Euler, started from a Gaussian
bump. Mirrors `test/test_shallow_water_2d.jl` so the benchmark exercises the same kernels
the tests cover, and needs no external data.
"""
function build_simulation(backend, n)
    grid = VF.CartesianGrid(n, n; gc=2)
    equation = VF.ShallowWaterEquationsPure()
    reconstruction = VF.LinearReconstruction()
    numericalflux = VF.CentralUpwind(equation)
    system = VF.ConservedSystem(backend, reconstruction, numericalflux, equation, grid)
    simulator = VF.Simulator(backend, system, VF.ForwardEulerStepper(), grid)

    u0 = x -> @SVector[exp(-(norm(x .- 0.5)^2 / 0.01)) + 1.5, 0.0, 0.0]
    VF.set_current_state!(simulator, u0.(VF.cell_centers(grid)))
    return simulator
end

"""
    time_steps!(simulator, backend, steps)

Wall clock seconds for `steps` timesteps.

Kernel launches in this package are fire and forget, so the device is synchronized
explicitly on both sides of the measurement. In practice each step already synchronizes
via the `maximum(wavespeeds)` reduction that computes the CFL timestep, which is worth
knowing when reading the numbers: it is a real serialization point in the time loop, not
an artefact of the benchmark.
"""
function time_steps!(simulator, backend, steps)
    KernelAbstractions.synchronize(backend.backend)
    start = time_ns()
    for _ in 1:steps
        VF.perform_step!(simulator, Inf)
    end
    KernelAbstractions.synchronize(backend.backend)
    return (time_ns() - start) / 1e9
end

function measure(backend, n)
    steps = steps_for(n * n)
    simulator = build_simulation(backend, n)

    # Warm up: the first call pays Julia specialization and, on Metal, shader compilation.
    VF.perform_step!(simulator, Inf)
    KernelAbstractions.synchronize(backend.backend)

    best = Inf
    for _ in 1:REPEATS
        simulator = build_simulation(backend, n)
        best = min(best, time_steps!(simulator, backend, steps))
    end
    return (steps=steps, seconds=best,
            ms_per_step=1000 * best / steps,
            mcells_per_second=(n * n * steps) / best / 1e6)
end

# --- configurations ------------------------------------------------------------------

configs = Tuple{String,Any}[]
push!(configs, ("CPU/Float64", make_cpu_backend(Float64)))
push!(configs, ("CPU/Float32", make_cpu_backend(Float32)))
if VF.has_metal_backend()
    push!(configs, ("Metal/Float32", VF.make_metal_backend()))
else
    @warn "No Metal backend on this machine; reporting CPU results only."
end
if VF.has_cuda_backend()
    push!(configs, ("CUDA/Float32", VF.make_cuda_backend(Float32)))
    push!(configs, ("CUDA/Float64", VF.make_cuda_backend(Float64)))
end

println("=" ^ 78)
println("VolumeFluxes backend comparison -- 2D shallow water")
println("=" ^ 78)
println("julia            : ", VERSION)
println("threads          : ", Threads.nthreads(),
        Threads.nthreads() == 1 ? "   (re-run with -t auto for a fair CPU number)" : "")
if VF.has_metal_backend()
    println("metal device     : ", VF.Metal.device().name)
end
println("configurations   : ", join(first.(configs), ", "))
println("grid sizes       : ", join(string.(SIZES) .* "^2", ", "))
println("work per point   : ~", @sprintf("%.0e", TARGET_CELL_UPDATES), " cell updates",
        " (best of $(REPEATS))")
println()

# --- sweep ---------------------------------------------------------------------------

results = Dict{String,Dict{Int,Any}}(label => Dict{Int,Any}() for (label, _) in configs)
capped = Set{String}()

for n in SIZES
    for (label, backend) in configs
        label in capped && continue
        try
            r = measure(backend, n)
            results[label][n] = r
            @printf("  %-14s %5d^2  steps=%-5d  %8.2f ms/step  %8.1f Mcell/s\n",
                    label, n, r.steps, r.ms_per_step, r.mcells_per_second)
            if r.seconds > TIME_CAP_SECONDS
                push!(capped, label)
                println("    ($(label) exceeded $(TIME_CAP_SECONDS)s; skipping larger grids)")
            end
        catch err
            println("  $(label) $(n)^2 FAILED: ",
                    first(replace(sprint(showerror, err), "\n" => " | "), 200))
            push!(capped, label)
        end
    end
end

# --- table ---------------------------------------------------------------------------

println()
println("Throughput (M cell-updates/s), and each GPU's speedup over CPU/Float64")
println()

# Formatted by hand rather than via PrettyTables: its keyword for column headers changed
# between major versions, and this needs no more than aligned columns.
gpu_labels = [label for (label, _) in configs if !startswith(label, "CPU/")]
header = ["grid"; first.(configs); ["$(g) vs CPU/F64" for g in gpu_labels]]
widths = [max(length(h), 13) for h in header]

print_row(cells) = println("| " * join(rpad.(cells, widths), " | ") * " |")
print_rule() = println("|" * join(["-" ^ (w + 2) for w in widths], "|") * "|")

print_row(header)
print_rule()
for n in SIZES
    row = Any["$(n)^2"]
    for (label, _) in configs
        r = get(results[label], n, nothing)
        push!(row, r === nothing ? "-" : @sprintf("%.1f", r.mcells_per_second))
    end
    baseline = get(results["CPU/Float64"], n, nothing)
    for g in gpu_labels
        r = get(results[g], n, nothing)
        push!(row, (r === nothing || baseline === nothing) ? "-" :
                   @sprintf("%.2fx", r.mcells_per_second / baseline.mcells_per_second))
    end
    print_row(string.(row))
end

# --- CSV -----------------------------------------------------------------------------

csv_path = joinpath(@__DIR__, "backend_comparison.csv")
open(csv_path, "w") do io
    println(io, "configuration,nx,ny,cells,steps,seconds,ms_per_step,mcell_updates_per_second")
    for (label, _) in configs, n in SIZES
        r = get(results[label], n, nothing)
        r === nothing && continue
        @printf(io, "%s,%d,%d,%d,%d,%.6f,%.6f,%.3f\n",
                label, n, n, n * n, r.steps, r.seconds, r.ms_per_step, r.mcells_per_second)
    end
end
println("\nwrote ", csv_path)

# --- correctness cross-check ---------------------------------------------------------
#
# Throughput is worthless if the answer is wrong. Compare the two Float32 backends after
# an identical number of steps: same precision, so they should agree very closely, and any
# real disagreement points at the Metal backend rather than at round-off.

for (label, backend) in configs
    startswith(label, "CPU/") && continue
    println()
    println("Correctness cross-check: CPU/Float32 vs $(label), 256^2, 50 steps")
    ref = build_simulation(make_cpu_backend(VF.realtype(backend)), 256)
    gpu = build_simulation(backend, 256)
    for _ in 1:50
        VF.perform_step!(ref, Inf)
        VF.perform_step!(gpu, Inf)
    end
    h_ref = collect(VF.current_interior_state(ref).h)
    h_gpu = collect(VF.current_interior_state(gpu).h)
    absdiff = maximum(abs.(h_ref .- h_gpu))
    reldiff = absdiff / maximum(abs.(h_ref))
    @printf("  max |h_cpu - h_gpu| = %.3e   (relative %.3e)\n", absdiff, reldiff)
    @printf("  sum h: cpu = %.8f   gpu = %.8f\n", sum(h_ref), sum(h_gpu))
    println(reldiff < 1e-4 ? "  OK: CPU and $(label) agree at the same precision." :
                             "  WARNING: larger disagreement than expected.")
end
