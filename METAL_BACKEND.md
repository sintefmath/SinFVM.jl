# Apple Metal backend for VolumeFluxes.jl

Branch: `stacked-metal-backend` (off `main` at `86af945`), landed as a stack of ten
commits -- see `git log --oneline main..stacked-metal-backend`. This document describes
the end state; each commit message describes its own step.

This adds a Metal backend so simulations can run on Apple GPUs, and extends the test
suite to cover it. It is written up here because the change is larger than "add a
backend": most of the work was making the package generic over its floating point type,
and a few things can only be validated on a machine with an NVIDIA GPU (see
[What still needs validating on a CUDA machine](#what-still-needs-validating-on-a-cuda-machine)).

---

## Using it

```julia
using VolumeFluxes

backend = VolumeFluxes.make_metal_backend()      # Float32; Float64 is not available

grid    = CartesianGrid(512, 512; gc=2)          # no need to name the precision here
equation = ShallowWaterEquationsPure()
system  = ConservedSystem(backend, LinearReconstruction(), CentralUpwind(equation), equation, grid)
simulator = Simulator(backend, system, ForwardEulerStepper(), grid)
```

`ConservedSystem` and `Simulator` convert the grid, equation, reconstruction and source
terms to the backend's precision, so ordinary code needs no changes to move to the GPU.

```julia
get_available_backends()          # [CPU/Float64]  (+ CUDA/Float64 where present)
get_available_backends(Float32)   # [CPU/Float32, Metal/Float32]
VolumeFluxes.has_metal_backend()  # true
```

Asking for a precision a backend cannot run now fails immediately and legibly:

```julia
julia> VolumeFluxes.make_metal_backend(Float64)
ERROR: ArgumentError: backend Metal.MetalKernels.MetalBackend does not support Float64
(requested realtype Float64). Use Float32 instead.
```

Two caveats when driving the low-level API (`for_each_*`, `compute_flux!`, `update_bc!`)
directly rather than through `ConservedSystem`:

- arrays a kernel writes to must live on the backend (`VolumeFluxes.convert_to_backend`);
- equations and other parameter structs you build by hand must be at the backend's
  precision (`VolumeFluxes.convert_realtype(VolumeFluxes.paramtype(backend), x)`).
  Grids are handled automatically by the launchers.

---

## Why this was more than a new backend type

Two things forced the shape of the change.

**1. `main` could not be loaded at all.** `Project.toml` said `name = "SinFVM"` while
`src/SinFVM.jl` declared `module VolumeFluxes` — the rename in `80e2087`/`8d1a08f` renamed
the module and all call sites but not the package. `using VolumeFluxes` failed with
*"Package VolumeFluxes not found in current path"*, and `using SinFVM` with *"Package
VolumeFluxes does not have Logging in its dependencies"*. So no test could run until
`Project.toml` and the entry filename were reconciled.

**2. Metal cannot handle `Float64` anywhere in a kernel.** Not just in arrays. Verified
against Metal.jl v1.9.2 on an M4 Pro:

- `MtlArray(zeros(Float64, 4))` → *"Metal does not support Float64 values"*
- a `@kernel` reading a `Float64` field out of an `Adapt`ed struct →
  `InvalidIRError: unsupported use of double value`
- **a kernel argument struct that merely _contains_ a `Float64` field is rejected even
  when the kernel only reads its integer fields.** This is what made `CartesianGrid` the
  central problem: it is `Adapt`ed into essentially every kernel, and its `extent` and `Δ`
  fields were hardcoded `SVector{N,Float64}`.

The package's *state* layer was already generic (`backend.realtype` → `create_buffer` →
`Volume` takes `RealType = eltype(buffer)`, and `make_cpu_backend(Float32)` already
existed). The *parameter* layer was not: grids, equation constants, the reconstruction
limiter and the source term parameters all defaulted to `Float64` and were handed straight
to kernels.

The existing branch `copilot/add-apple-gpus-support` converted only the *buffers* to
Float32 and never touched those parameters, so it would have failed at the first Metal
kernel launch; it also dropped the zero-argument `make_cuda_backend()` that five files
still call. Only its rename was reused.

---

## Design

### The realtype split: `realtype` vs `paramtype`

New file [src/realtype.jl](src/realtype.jl).

```julia
realtype(backend)   # element type of state arrays; may be a ForwardDiff.Dual
paramtype(backend)  # plain float for geometry and physical parameters
```

These are deliberately different. `test/test_swe_ad.jl` builds a backend with
`realtype = ForwardDiff.Dual`, and geometry must not become a dual number — a simulation
differentiated with respect to a wall height should not turn `grid.Δ` into a `Dual`.
`paramtype` strips the AD wrapper via `base_float`.

`convert_realtype(R, x)` rewrites the parameter layer of `x` to use `R`. It leaves
anything it does not recognise untouched — importantly `ForwardDiff.Dual` passes through
unchanged, because `Dual <: Real` but not `Dual <: AbstractFloat`, so that falls out for
free rather than needing a special case.

### One conversion point, so no caller changes

`CartesianGrid` gained a **trailing** `RealType` type parameter. All 49 of the
`CartesianGrid{1}` / `CartesianGrid{2}` dispatch sites constrain only the *first* parameter
and a left-partial application still matches whatever follows — so none of those
signatures needed touching.

The conversion happens where both the backend and the grid are already in hand:

- `ConservedSystem` converts grid, reconstruction, numerical flux, equation, source terms
  and the implicit source term.
- `Simulator` converts its own grid copy.
- `Volume` converts its grid, because a `Volume` carries its grid into every kernel it is
  passed to.

The consequence is that **the `CartesianGrid(...)` call sites in `test/` and
`examples/` needed no edits** — user-held grids stay Float64 for host-side `cell_centers`
/ `cell_faces`, and only what reaches a kernel is converted.

### Host time stays Float64 while the state is Float32

`Simulator`'s float parameter used to be `backend.realtype`, so a Float32 backend got a
Float32 time accumulator. That is a latent hang, not a precision nicety: with `t ≈ 1e4`,
`eps(1.0f4) ≈ 1e-3` exceeds a typical `dt`, so `t[1] += dt` becomes a no-op and
`simulate_to_time` never terminates. `test/test_rain.jl` runs to `t = 10800`.

`Simulator` now has an independent `TimeType`:

```julia
TimeType = promote_type(typeof(cfl), typeof(t0), realtype(backend))
```

which gives Float64 for a Float32 *or* Float64 backend, and still widens to
`ForwardDiff.Dual` under AD — where `dt` genuinely is a dual, since it depends on the wave
speeds being differentiated. `dt` is then narrowed to `paramtype` at each of the three
points it enters a kernel (`ForwardEulerStepper`, `RungeKutta2`, `implicit_friction`), and
`t` likewise in `add_time_derivative!`.

### Float64 literals inside kernels

About 30 sites across 16 files promoted Float32 to Float64 inside a kernel. The rewrite
uses integer arithmetic (`x / 2` rather than `0.5 * x`) and `zero(x)` / `iszero(x)` rather
than hardcoded Float32, so **the Float64 path stays bit-identical** — every literal
involved was a dyadic rational, so `/2` and `/4` are exact and LLVM folds them back to a
multiply. The house style already existed: `centralupwind.jl:63-64` used `zero(...)`
correctly while lines 40-41 did not.

Two were hard blockers rather than cosmetic:

- `friction.jl` — `if speed == 0.0` promotes to a `double` comparison. Now `iszero(speed)`.
- `friction.jl` — `@SVector [0.0, friction_scalar]` *promotes* its elements rather than
  converting them, making the whole friction update Float64. Now `zero(friction_scalar)`.

### `cbrt` cannot work on Metal

`Base.cbrt(::Float32)` computes through a Float64 intermediate (`Base.Math._improve_cbrt`
widens to `Float64` internally), so it cannot compile for Metal no matter how generic the
calling code is. `friction_bsa2012` and `friction_fcg2016` used it. Replaced with

```julia
cuberoot(x) = copysign(abs(x)^(oftype(x, 1//3)), x)
```

which stays in the input precision. **This changes results in the last bits on CPU and
CUDA too** — it is the one change in this branch that is not bit-identical for Float64.
The default friction model (`friction_bh2021`) does not use it; `examples/terrain.jl` and
`examples/urban.jl` do.

### Backend layer

`src/backends/kernel_abstractions.jl`:

- `make_metal_backend(RealType=Float32)`, `const MetalBackend`, `name(::MetalBackend)`,
  `has_metal_backend()`.
- `make_cuda_backend(RealType)` added, **zero-argument `make_cuda_backend()` kept** (five
  files call it).
- `get_available_backends(realtype=Float64)` returns the devices that support that
  realtype — Metal only for Float32. The default is unchanged for existing callers.
- Default realtype now comes from `KernelAbstractions.supports_float64`, which Metal sets
  to `false`. Asking a backend for a realtype it cannot run throws a clear
  `ArgumentError` instead of an inscrutable `InvalidIRError` later.
- The `@show err` in the backend probe became `@debug` — it printed a CUDA error object at
  each of ~26 test call sites on a machine without CUDA.
- **The hardcoded workgroup size of `1024` was removed from all five launchers.** Metal's
  KA backend only autotunes `maxTotalThreadsPerThreadgroup` when the workgroup size is
  *not* given statically, and 1024 threads is more than these register-heavy kernels can
  be relied on to take. CUDA falls back to `CUDA.launch_configuration` in the same way.
  *This changes CUDA launch parameters and so needs a look on a CUDA machine.*
- `for_each_cell_kernel` no longer takes `grid`; it never used it, and it was being
  marshalled to the device at 12 of the 33 `@fvmloop` sites.
- New `kernel_arg` / `kernel_args` convert **grids only** in the launcher arguments, so the
  low-level API is safe to call directly with a host-built Float64 grid. Deliberately
  narrow: converting everything would copy a Float64 *output* array passed through `y...`
  and writes would land in the copy and be silently lost.

`src/backends/buffer.jl` collapsed to backend-agnostic definitions, since CPU, CUDA and
Metal all implement `KernelAbstractions.zeros` and `Adapt.adapt_storage`:

```julia
create_buffer(backend, nvars, res) = KernelAbstractions.zeros(backend.backend, realtype(backend), (res..., nvars))
```

`convert_to_backend` converts the element type **host-side, before the transfer** — there
is no Float64 Metal array to convert from — and skips the conversion entirely when the
types already agree, so it stays the identity when the array is already on the device.

`src/volume/volume.jl`: the eight per-(backend, wrapper) identity `convert_to_backend`
methods (which would have become twelve) collapsed to four backend-agnostic ones.

### Bugs found and fixed along the way

These were all pre-existing, and all invisible while the suite effectively only ran on the
CPU:

1. **`TimeDependentRain` never moved its arrays to the device**, unlike
   `HortonInfiltration` and `BottomTopography2D` — it would have produced invalid device
   pointers on any GPU backend. Now takes an optional backend and has a hand-written
   `adapt_structure`.
2. **Three bulk `setindex!` methods on `VolumeVariable` / `InteriorVolumeVariable` were
   missing `convert_to_backend`.** Assigning a host Float64 array into a slice of a device
   Float32 array makes GPUArrays allocate a Float64 device scratch array, which Metal
   refuses.
3. **`test_for_each_inner_cell.jl` and `test_update_bc.jl` passed host arrays into
   kernels** — fine on the CPU, impossible on a GPU.
4. **Two backend loops immediately overwrote the loop variable with a CPU backend**
   (`test_for_each_inner_cell.jl`, `test_update_bc.jl`), so the non-CPU backend was never
   exercised.
5. **`get_test_name` in `test_swe1D_sim.jl` / `test_friction_1d.jl`** built names by regex
   over `string(typeof(...))` assuming a module prefix that is not there for exported
   types; `match` returned `nothing` and the test errored. These were the 2 errors in the
   `main` baseline. Replaced with `nameof`.
6. **`extent(grid, direction)` returned an `SVector{2,Int64}`**, truncating non-integer
   extents.
7. `simulator.jl` used `CUDA.@allowscalar` in core library code; now
   `GPUArraysCore.@allowscalar`. (Functionally the same macro — verified that
   `CUDA.@allowscalar` *is* `GPUArraysCore.@allowscalar` and does work on `MtlArray`, which
   is why the 79 `CUDA.@allowscalar` call sites in `test/` needed no change.)

---

## Test suite

New [test/testing_utils.jl](test/testing_utils.jl) (not named `test_*.jl`, so
`runtests.jl` does not auto-run it) provides:

| helper | purpose |
|---|---|
| `test_backends()` | every device at Float64, then every device at Float32 |
| `backend_label(backend)` | `"Metal/Float32"` — names per-backend testsets |
| `test_atol(backend, atol; float32=1e-4)` | floors a Float64-calibrated tolerance at what Float32 can reach |
| `is_float64_backend(backend)` | for the few tests that must stay double precision |
| `reference_backend(backend)` | a CPU backend at the *same* precision |
| `backend_params(backend, x)`, `to_backend(backend, a)`, `test_grid(backend, ...)` | for tests that drive the low-level API directly |

The 24 backend loops (across 23 files) became `@testset "$(backend_label(backend))" for backend in test_backends()`,
which gives per-backend attribution in one line — most of these loops previously had no
enclosing testset at all, so a CPU and a GPU failure were indistinguishable in the summary.

**Including a Float32 CPU backend is the point of the design.** Every Float32 failure
observed during this work appeared identically on CPU/Float32 and Metal/Float32, which is
what makes it possible to say "this is Float32 round-off" rather than "Metal is broken".

Where a test computed a trusted CPU reference and compared another backend against it
(`test_swe1D_sim.jl`, `test_friction_1d.jl`), the reference now uses
`reference_backend(backend)` — the same precision. A Float64 reference for a Float32 run
measures the precision difference, not the backend, and the discrepancy grows with cell
count.

### Deliberately restricted to Float64

`test/test_timesteppers.jl` — **the only test narrowed in coverage.** It measures an
observed order of convergence by fitting a line through errors at `dt` down to `2^-15`,
i.e. up to 32768 sequential steps on a *one-cell* grid. Two independent reasons:

- In Float32 the round-off accumulated over 32768 steps (~2e-5) is the same size as the
  discretization error being measured (`dt ~ 3e-5`), so the fitted slope is meaningless.
  Loosening the tolerance would not make the test correct, only quiet.
- On a GPU it times kernel launch overhead on a single cell tens of thousands of times
  over. It took minutes and tested nothing about the device.

The restriction is a named function (`timestepper_backends()`) with the reasoning in a
comment, not a silent skip.

Tolerances were widened (via `test_atol`) rather than restricted in
`test_bottom_topography.jl`, `test_lake_at_rest.jl`, `test_lake_at_rest_2d.jl`,
`test_friction_1d.jl` and `test_swe1D_sim.jl`. In the last two the compared quantity is a
*discretization* difference between two equation formulations summed over 1024 cells, not
a round-off, which is why those get an explicit larger `float32` floor with a comment
saying so.

---

## Results on this machine (Apple M4 Pro, 20-core GPU, Julia 1.11.8, 10 threads)

### Unit tests

`julia --project -t auto test/runtests.jl`

```
VolumeFluxes tests | 172610  172610  2m18.6s
```

**172610 passed, 0 failed, 0 errored**, across `CPU/Float64`, `CPU/Float32` and
`Metal/Float32`. CUDA is absent here, so it contributes nothing to this run.

For comparison, the `main` baseline (with only the rename applied, so it could load at
all) was **58012 passed, 2 errored** in 50.6s on `CPU/Float64` alone. The two errors were
the pre-existing `get_test_name` bug described above, now fixed.

### The Float64 path is bit-for-bit unchanged

The literal rewrites in Step "Float64 literals inside kernels" were meant to be exact, and
are. Running an identical 96x64 2-D simulation for 40 steps on `CPU/Float64` against the
unmodified code gives identical digits in every configuration:

| case | sum(h) | max(h) | sum(abs(hu)) |
|---|---|---|---|
| pure / euler | 9409.019452636006 | 1.7821488231116376 | 615.1251290839359 |
| pure / rk2 | 9409.019452636006 | 1.7703490764422338 | 609.5687180612507 |
| practical / euler | 9409.019452636006 | 1.783279341279147 | 593.2155662041944 |
| practical / rk2 | 9409.019452636006 | 1.7713645534700992 | 587.9540968169038 |

The one deliberate exception is `cbrt` → `cuberoot`, which only affects
`friction_bsa2012` / `friction_fcg2016` (not the default friction model) and is not
covered by the fingerprint above.

### Performance

`julia --project -t auto benchmark/backend_comparison.jl` — completed in about 4 minutes,
inside the intended budget. 2-D shallow water, `ShallowWaterEquationsPure` + linear
reconstruction + central upwind + forward Euler, work held at ~5e7 cell updates per point,
best of 3, warm-up excluded.

Throughput in M cell-updates/s:

| grid | CPU/Float64 | CPU/Float32 | Metal/Float32 | Metal vs CPU/F64 |
|---|---|---|---|---|
| 64² | 14.4 | 14.6 | 5.2 | 0.36x |
| 128² | 37.2 | 38.6 | 20.5 | 0.55x |
| 256² | 51.0 | 51.3 | 78.3 | 1.54x |
| 512² | 68.0 | 67.4 | 269.0 | 3.95x |
| 1024² | 73.4 | 73.5 | 393.0 | 5.36x |
| 2048² | 77.2 | 77.7 | 470.1 | **6.09x** |

Run-to-run variation is a few percent on this machine — three runs of the same script gave
5.97x, 6.09x and 6.28x at 2048² — so read the speedups as approximate. The small grids vary
most, since they are dominated by per-step launch and synchronization overhead rather than
by arithmetic.

Reading these:

- **The GPU only wins above roughly 256².** Below that, per-step launch overhead and the
  device-to-host synchronization for the CFL timestep dominate; at 64² Metal is 3x
  *slower*. This is worth knowing before switching a small case over.
- **CPU/Float32 is no faster than CPU/Float64** (within noise at every size), so the
  speedup at the top end is genuinely the GPU rather than the halved precision. That is
  the reason for carrying the third configuration.
- The GPU curve is still climbing at 2048², so the asymptote is higher than 6x.
- Every step synchronizes, because `compute_flux!` returns `maximum(wavespeeds)` to
  compute the timestep. That is the obvious next optimization target for the GPU path and
  it is a property of the algorithm as written, not of the benchmark.

### CPU/Float32 and Metal/Float32 agree exactly

The benchmark's built-in cross-check, at 256² after 50 steps:

```
max |h_cpu - h_gpu| = 0.000e+00   (relative 0.000e+00)
sum h: cpu = 100362.87500000   gpu = 100362.87500000
```

Bit-identical. Together with every Float32 test failure during development appearing
*identically* on CPU/Float32 and Metal/Float32, this is the main evidence that the Metal
backend is computing the same thing as the CPU rather than merely producing
plausible-looking numbers.

---

## A Float32 limitation worth knowing about: source terms on fine grids

This is a property of Float32, not of the Metal backend, but Metal is Float32-only so it
is now reachable. It was measured, not assumed.

A source term adds `dt * rate` to the water depth each step. In Float32 that increment is
lost entirely once it falls below `eps(h)`. Because `dt` follows `dx` through the CFL
condition, the risk grows as the grid gets finer. Constant rain of 0.01/h onto `h = 1.0 m`,
100 s of simulated time, 10x10 cells, varying the domain size to vary `dx`:

| domain | dt | per-step increment | expected Δh | CPU/Float64 Δh | CPU/Float32 Δh |
|---|---|---|---|---|---|
| 1000 m | 4.22 s | 1.17e-5 | 2.778e-4 | 2.778e-4 | 2.778e-4 ✅ |
| 10 m | 0.0737 s | 2.05e-7 | 2.778e-4 | 2.778e-4 | 2.987e-4 (7.5% high) |
| 1 m | 0.0019 s | 5.27e-9 | 2.778e-4 | 2.778e-4 | **0.0 — rain silently ignored** |

(`eps(1.0f0) = 1.19e-7`.)

Over *duration* there is no problem: the same setup at the suite's resolution gives a
steady ~2e-6 absolute error whether run for 3, 12 or 24 simulated hours — the error does
not accumulate, because `h` grows alongside it. The failure mode is spatial resolution,
not run length.

**Why this matters here:** `examples/terrain.jl` and `examples/urban.jl` set
`extent = size(terrain)` while coarsening the grid, giving `dx` of order a few metres —
the regime where the middle row of that table lives. A rain-on-grid stormwater run on a
fine grid should be treated as Float64 (CPU or CUDA) until this is addressed.

`test/test_rain.jl` and `test/test_infiltration.jl` are unaffected: they construct
backends explicitly rather than going through `test_backends()`, so they still run in
Float64 only. That is why the suite is green and this is still worth writing down.

Fixing it properly means a compensated (Kahan/Neumaier) or Float64 accumulator for the
depth, with Float32 fluxes — worth an issue, and out of scope here.

---

## What still needs validating on a CUDA machine

**I have no NVIDIA GPU here** — `CUDA.functional()` is `false` on this machine, so
everything below is unexercised. The CUDA path was previously the only GPU path, and
several changes touch it.

Run, from the repository root, in this order — each step gates the next:

```bash
# 1. does the environment resolve and the package load at all?
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using VolumeFluxes; println("loaded")'

# 2. is CUDA discovered, at both precisions?
julia --project -e 'using VolumeFluxes;
  @show get_available_backends();          # expect [CPU/Float64, CUDA/Float64]
  @show get_available_backends(Float32);   # expect [CPU/Float32, CUDA/Float32]
  @show VolumeFluxes.has_cuda_backend()'

# 3. one CUDA simulation, before the whole suite -- clearer errors
julia --project -t auto -e 'using VolumeFluxes, StaticArrays, LinearAlgebra
  b = VolumeFluxes.make_cuda_backend()
  g = CartesianGrid(128, 128; gc=2); eq = ShallowWaterEquationsPure()
  s = ConservedSystem(b, LinearReconstruction(), CentralUpwind(eq), eq, g)
  sim = Simulator(b, s, ForwardEulerStepper(), g)
  u0 = x -> @SVector[exp(-(norm(x .- 0.5)^2/0.01)) + 1.5, 0.0, 0.0]
  VolumeFluxes.set_current_state!(sim, u0.(VolumeFluxes.cell_centers(g)))
  simulate_to_time(sim, 0.02; show_progress=false)
  println("sum h = ", sum(collect(VolumeFluxes.current_interior_state(sim).h)))'

# 4. the full four-configuration suite
julia --project -t auto test/runtests.jl

# 5. performance, now including CUDA/Float32 and CUDA/Float64 automatically
julia --project -t auto benchmark/backend_comparison.jl
```

The benchmark picks up CUDA on its own (`benchmark/backend_comparison.jl` appends
`CUDA/Float32` and `CUDA/Float64` when `has_cuda_backend()`), so it doubles as a check that
the removed workgroup-size override did not cost CUDA throughput. There are no
pre-existing CUDA numbers to compare against, so record this run as the baseline. With
four or five configurations instead of three it will run longer than the ~4 minutes it
takes here; `--quick` halves the size sweep, and `TIME_CAP_SECONDS` at the top of the
script stops growing the grid for a configuration once one timed run exceeds it.

`Manifest.toml` is gitignored and untracked, so there is no pinned Metal version to
inherit -- a CUDA machine resolves the whole environment from scratch and will pick up
whatever Metal version is current. Metal still gets installed there, because it is a plain
`[deps]` entry.

### 1. Does the package still load at all? (highest risk)

`Metal` is now a **hard dependency** (`Project.toml`), matching how CUDA is already
treated, and `src/backends/kernel_abstractions.jl` does `import Metal` at load time.

Metal.jl's `__init__` handles non-Apple platforms by logging and returning rather than
throwing (`Metal/src/initialization.jl:29-33`):

```julia
if !Sys.isapple()
    @error "Metal.jl is only supported on macOS"
    return
end
```

and `LLVMDowngrader_jll` does ship Linux builds, so resolution and loading *should*
succeed. **This is inferred from reading the source, not tested on Linux.** Expect
`using VolumeFluxes` to print that `@error` line on every load.

If it turns out to break, or the error line is unacceptable, the fix is to move both CUDA
and Metal to `[weakdeps]` with `ext/VolumeFluxesCUDAExt.jl` / `ext/VolumeFluxesMetalExt.jl`.
That was considered and deliberately deferred: the `const CUDABackend`/`MetalBackend`
aliases and the per-backend methods are type-level references resolved at load time and
would have to move into the extensions.

### 2. The new `CUDA/Float32` configuration

`test_backends()` returns `[CPU/Float64, CUDA/Float64, CPU/Float32, CUDA/Float32]` on a
CUDA machine — so the suite runs **four** configurations, and `CUDA/Float32` has never
been run before. This is the main new surface.

What "good" looks like, by analogy with this machine: **0 failures and 0 errors**. Expect
roughly 4/3 of the ~172k assertions counted here (four configurations instead of three)
and noticeably longer than the 2m22s seen here.

If `CUDA/Float32` throws where `CUDA/Float64` passes, the likely cause is the same class of
problem Metal exposed — a Float64 value reaching a kernel — except that CUDA *tolerates*
Float64 rather than rejecting it, so it will show up as a wrong answer or a tolerance
failure rather than a compile error. `test_atol` already widens the tolerances for any
non-Float64 backend, so a `CUDA/Float32` tolerance failure is worth investigating rather
than widening further.

A useful narrowing tool: `CPU/Float32` and `CUDA/Float32` should agree very closely, the
way `CPU/Float32` and `Metal/Float32` did here (bit-identical at 256² after 50 steps). If
they diverge, the problem is in the CUDA path, not in Float32.

### 3. Specific changes that alter the CUDA code path

| change | what to check |
|---|---|
| workgroup size `1024` removed from all five launchers | correctness first, then whether throughput moved — CUDA now uses `CUDA.launch_configuration` occupancy autotuning instead of a fixed 1024, which may well be *faster* but is a real behavioural change |
| `create_buffer` now `KernelAbstractions.zeros(backend.backend, ...)` instead of `CUDA.CuArray(zeros(...))` | allocates on the device rather than host-allocating and copying |
| `convert_to_backend` now `Adapt.adapt(backend.backend, ...)` instead of `CUDA.CuArray(array)` | and the reverse path, previously `convert_to_backend(::CPUBackend, ::CuArray) = collect(array)`, is now the generic `Adapt.adapt` |
| eight `Volume` identity `convert_to_backend` methods collapsed to four generic ones | should be behaviour-preserving |
| `GPUArraysCore.@allowscalar` replaces `CUDA.@allowscalar` in `simulator.jl:135` | same macro, but confirm on real CUDA |
| `set_current_state!` 1-D branch now also calls `convert_to_backend` | new behaviour on that path |
| `is_zero(B)` split into `iszero(B.B)` / `all(iszero, B.B)` | the array form now reduces on the device via GPUArrays `mapreduce` |
| `cbrt` → `cuberoot` in `friction_bsa2012` / `friction_fcg2016` | results differ in the last bits; only `examples/terrain.jl` and `examples/urban.jl` use these models |
| `TimeDependentRain` now moves arrays to the backend | previously broken on GPU, so this is the first time `test_rain.jl`'s CUDA-guarded blocks can actually pass |
| `for_each_cell_kernel` lost its unused `grid` argument | fewer kernel arguments on CUDA too |

`KernelAbstractions.supports_float64` defaults to `true` and CUDA.jl does not override it
(verified by reading both packages), so the new constructor guard does not affect
`make_cuda_backend()`.

### 4. The specific test files to watch

The suite is a single `runtests.jl` that auto-includes every `test/test_*.jl`, so there is
nothing to select — but these are the files whose CUDA behaviour actually changed, roughly
in order of interest:

| file | why it matters on CUDA |
|---|---|
| `test_rain.jl` | its `has_cuda_backend()`-guarded blocks exercise `TimeDependentRain`, which never moved its arrays to the device before. **These assertions have most likely never actually passed on a GPU.** |
| `test_infiltration.jl` | the CUDA block also computes the CPU reference (`h_cpu`) *inside* the guard, so on a CPU-only machine those assertions never ran at all. First real exercise. |
| `test_update_bc.jl`, `test_for_each_inner_cell.jl` | were hardcoded to CPU / silently shadowed the loop variable; now genuinely run on every backend. Their data is now moved to the device. |
| `test_volume.jl`, `test_volume_2d.jl` | cover the three bulk `setindex!` methods that were missing `convert_to_backend`. |
| `test_compute_flux_2d.jl`, `test_shallow_water_equations.jl` | now build their grid/equation via `test_grid` / `backend_params`; confirms the low-level API path. |
| `test_kernel_abstractions.jl` | CUDA-only file, entirely inert here. Contains no `@test`, just a timed launch loop — check it still runs. |
| `test_swe_ad.jl` | the `Simulator` `TimeType` promotion is the part that touches ForwardDiff. Passes on CPU here. |
| `test_timesteppers.jl` | now restricted to Float64 backends, so `CUDA/Float64` still runs it but `CUDA/Float32` will not. Expect the same assertion count as before for CUDA/Float64. |
| `test_swe1D_sim.jl`, `test_friction_1d.jl` | the `get_test_name` fix (these were the 2 baseline errors) and the same-precision reference change. |

### 4. Worth a look but lower risk

- `test_swe_ad.jl` (ForwardDiff) passes here on CPU; the `TimeType` promotion is the part
  that interacts with AD.
- `examples/urban.jl:103` still does an unguarded `make_cuda_backend()` — pre-existing,
  untouched.
- `test/benchmark.jl` (the old CUDA-hardcoded benchmark, sweeping to 2^28 cells, with a
  `plot!` call that has no preceding `plot`) was left alone; `benchmark/backend_comparison.jl`
  is new and independent.

---

## Files

New: `src/realtype.jl`, `test/testing_utils.jl`, `benchmark/backend_comparison.jl`.
Renamed: `src/SinFVM.jl` → `src/VolumeFluxes.jl` (+ `Project.toml` `name`).
`Project.toml` gains `Metal` and `GPUArraysCore`.
