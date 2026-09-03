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

import Adapt

# Host <-> device transfer and buffer allocation.
#
# All three KernelAbstractions backends we support (CPU, CUDA, Metal) implement
# `KernelAbstractions.zeros` and `Adapt.adapt_storage` for their own device, so these are
# backend-agnostic rather than one method per backend.

"""
    convert_to_backend(backend, array)

Move `array` onto `backend`'s device, converting its element type to the backend's
parameter type on the way.

The element type conversion is load-bearing for Metal, which refuses `Float64` buffers
outright -- and refuses them recursively, so a `Matrix{SVector{3,Float64}}` (exactly what
`set_current_state!` is handed) is rejected too. It also has to happen *before* the
transfer for the same reason: there is no Float64 Metal array to convert from.
"""
function convert_to_backend(backend, array::AbstractArray)
    target = paramtype(backend)
    if converted_eltype(target, eltype(array)) === eltype(array)
        # Nothing to convert; this is also the identity case when `array` already lives
        # on `backend`, so it must not force a round trip through the host.
        return Adapt.adapt(backend.backend, array)
    end
    return Adapt.adapt(backend.backend, convert_realtype(target, Adapt.adapt(Array, array)))
end

function create_buffer(backend, number_of_variables::Int64, spatial_resolution)
    KernelAbstractions.zeros(backend.backend, realtype(backend),
        (spatial_resolution..., number_of_variables))
end
