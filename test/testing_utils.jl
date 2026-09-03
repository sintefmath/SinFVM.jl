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

# Shared helpers for running the suite across backends of differing precision.
#
# Deliberately not named `test_*.jl`: `runtests.jl` auto-includes every `test/test_*.jl`,
# and this file holds no tests of its own.

using VolumeFluxes

"""
    SHOW_PLOTS

Whether the test suite should render figures. Off by default.

Several test files draw diagnostic figures, and a few do it unconditionally in the middle
of the simulation they are testing. Rendering costs about 0.3 s per figure, roughly five
times what building it costs, and nothing looks at the output during a test run -- and it
is paid once per backend now that the suite runs a matrix. Enable with:

    VOLUMEFLUXES_TEST_PLOTS=true julia --project -t auto test/runtests.jl
"""
const SHOW_PLOTS = get(ENV, "VOLUMEFLUXES_TEST_PLOTS", "false") == "true"

"""
    maybe_display(figure)

`display(figure)` when [`SHOW_PLOTS`](@ref) is set, otherwise a no-op.
"""
maybe_display(figure) = SHOW_PLOTS ? display(figure) : nothing

"""
    test_backends()

The backend matrix the suite runs against: every available device at Float64, then every
available device at Float32.

On a machine with an Apple GPU that is `[CPU/Float64, CPU/Float32, Metal/Float32]`. The
Float32 CPU backend is not redundant -- it is the *same precision* reference for the GPU,
which makes a Metal failure distinguishable from ordinary Float32 round-off.
"""
test_backends() = vcat(VolumeFluxes.get_available_backends(Float64),
                       VolumeFluxes.get_available_backends(Float32))

"""
    backend_label(backend)

`"CPU/Float64"`, `"Metal/Float32"`, ... -- used to name per-backend testsets so a failure
says which backend and which precision it came from.
"""
backend_label(backend) = string(VolumeFluxes.name(backend), "/", VolumeFluxes.realtype(backend))

"""
    is_float64_backend(backend)

Whether this backend runs in full double precision.

Used to restrict the handful of tests that genuinely cannot hold in Float32 -- see
`test_atol` for the ones that can, with a wider tolerance.
"""
is_float64_backend(backend) = VolumeFluxes.realtype(backend) === Float64

"""
    test_atol(backend, atol_float64; float32=1e-4)

Scale a Float64-calibrated absolute tolerance to `backend`'s precision.

Every tolerance in this suite was chosen against Float64, where `eps` is about 2e-16.
Float32 only has about 1e-7 of relative resolution, so tolerances like `1e-14` are far
below what a Float32 backend can achieve on quantities of order one, let alone on sums
over thousands of cells. Rather than rescaling each tolerance by an eps ratio -- which
would loosen the already-generous ones absurdly -- this floors them at a level Float32
can actually reach, and leaves anything already looser untouched.

Pass `float32` to tighten or loosen the floor for a specific comparison.
"""
function test_atol(backend, atol_float64; float32=1e-4)
    is_float64_backend(backend) && return atol_float64
    return max(atol_float64, float32)
end

"""
    reference_backend(backend)

A CPU backend at the *same* precision as `backend`.

Several tests compute a trusted solution on the CPU and compare a second backend against
it. Using a Float64 CPU reference for a Float32 run would measure the precision
difference rather than the backend, and the discrepancy grows with the cell count, so the
reference is matched to the backend's element type instead.
"""
reference_backend(backend) = make_cpu_backend(VolumeFluxes.realtype(backend))

"""
    backend_params(backend, x)

`x` with its parameter floats converted to `backend`'s precision.

Tests that build an equation, grid or bottom topography by hand and hand it straight to a
kernel need this; the high level API does it for them inside `ConservedSystem`. Metal
rejects a kernel argument struct containing a Float64 field even when the kernel never
reads it.
"""
backend_params(backend, x) =
    VolumeFluxes.convert_realtype(VolumeFluxes.paramtype(backend), x)

"""
    to_backend(backend, array)

`array` moved onto `backend`'s device. Thin alias for `VolumeFluxes.convert_to_backend`,
for tests that exercise the kernel launchers directly and therefore have to supply device
arrays rather than host ones.
"""
to_backend(backend, array) = VolumeFluxes.convert_to_backend(backend, array)

"""
    test_grid(backend, args...; kwargs...)

`VolumeFluxes.CartesianGrid` built at `backend`'s parameter precision.

Only needed by tests that drive the low level API -- `compute_flux!`, `update_bc!`, the
`for_each_*` loops -- directly. Those bypass `ConservedSystem`, which is where a normal
simulation gets its grid converted, and quantities derived host-side from the grid (`Δx`,
say) are captured into kernels with whatever type the grid had.
"""
test_grid(backend, args...; kwargs...) =
    VolumeFluxes.CartesianGrid(args...; realtype=VolumeFluxes.paramtype(backend), kwargs...)
