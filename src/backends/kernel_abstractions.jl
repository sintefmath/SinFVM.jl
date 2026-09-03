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

using KernelAbstractions
import CUDA
import Metal

abstract type Backend end

toint(x) = x
toint(x::CartesianIndex{1}) = x[1]

"""
    default_realtype(ka_backend)

The widest float type the given KernelAbstractions backend can actually run.

A backend that does not support `Float64` reports so via
`KernelAbstractions.supports_float64`, and for such a device `Float64` is not merely slow,
it fails to compile.
"""
default_realtype(ka_backend) =
    KernelAbstractions.supports_float64(ka_backend) ? Float64 : Float32

struct KernelAbstractionBackend{KABackendType, RealType} <: Backend
    backend::KABackendType
    realtype::Type{RealType}

    function KernelAbstractionBackend(backend; realtype=default_realtype(backend))
        if !KernelAbstractions.supports_float64(backend) && base_float(realtype) === Float64
            throw(ArgumentError(
                "backend $(typeof(backend)) does not support Float64 " *
                "(requested realtype $(realtype)). Use Float32 instead."))
        end
        return new{typeof(backend), realtype}(backend, realtype)
    end
end

"""
    realtype(backend)

Element type of the state arrays this backend allocates. May be a `ForwardDiff.Dual`
when the backend is used for sensitivity computations (see `test/test_swe_ad.jl`).
"""
realtype(backend::KernelAbstractionBackend{B,RealType}) where {B,RealType} = RealType

"""
    paramtype(backend)

Plain float type to use for geometry and physical parameters on this backend.

This is deliberately *not* the same as [`realtype`](@ref): differentiating a simulation
with respect to a physical parameter makes `realtype` a `ForwardDiff.Dual`, but the grid
spacing and the equation constants should stay plain floats.
"""
paramtype(backend::Backend) = base_float(realtype(backend))

make_cpu_backend() = KernelAbstractionBackend(get_backend(ones(3)))
make_cpu_backend(RealType) = KernelAbstractionBackend(get_backend(ones(RealType, 3)); realtype=RealType)

make_cuda_backend() = KernelAbstractionBackend(get_backend(CUDA.cu(ones(3))); realtype=Float64)
make_cuda_backend(RealType) = KernelAbstractionBackend(get_backend(CUDA.cu(ones(RealType, 3))); realtype=RealType)

# The probe array is always Float32: its only job is to hand us the KernelAbstractions
# backend object, and Float32 is the only element type Metal can allocate. Probing with
# `RealType` would make `make_metal_backend(Float64)` fail inside Metal's allocator instead
# of with the clear ArgumentError from the `KernelAbstractionBackend` constructor.
make_metal_backend(RealType=Float32) =
    KernelAbstractionBackend(get_backend(Metal.MtlArray(ones(Float32, 3))); realtype=RealType)
const CUDABackend = KernelAbstractionBackend{CUDA.CUDAKernels.CUDABackend}
const MetalBackend = KernelAbstractionBackend{Metal.MetalKernels.MetalBackend}
const CPUBackend = KernelAbstractionBackend{KernelAbstractions.CPU}

name(::CUDABackend) = "CUDA"
name(::MetalBackend) = "Metal"
name(::CPUBackend) = "CPU"

"""
    get_available_backends(realtype=Float64)

Every backend on this machine that can run with the given element type, CPU first.

Metal only appears for `Float32`, since it cannot run `Float64` at all. Backends that are
not present are skipped quietly -- the reason is available under `JULIA_DEBUG`.
"""
function get_available_backends(realtype=Float64)
    backends = Any[make_cpu_backend(realtype)]

    for make_backend in (make_cuda_backend, make_metal_backend)
        try
            push!(backends, make_backend(realtype))
        catch err
            @debug "backend unavailable" make_backend realtype err
        end
    end
    return backends
end

function has_cuda_backend()
    try
        make_cuda_backend()
        return true
    catch err
        return false
    end
end

function has_metal_backend()
    try
        make_metal_backend()
        return true
    catch err
        return false
    end
end

"""
    kernel_arg(backend, x)

`x` as it must look to be passed into a kernel on `backend`.

A device may refuse a kernel argument struct that merely *contains* a Float64 field, even
when the kernel only reads integer fields out of it -- so a host-built
`CartesianGrid{..., Float64}` cannot be handed to such a kernel, although every one of
these loops uses only its integer cell counts. The high level API converts the grid once
in `ConservedSystem`/`Simulator`; doing it here as well makes the low level loops safe to
call directly. It is the identity, and therefore free, when the types already agree.

Deliberately narrow: only `Grid`s are converted. Converting everything would copy any
Float64 *output* array passed through `y...`, and writes would then land in the copy and be
silently lost. Other parameter structs are the caller's responsibility.
"""
kernel_arg(backend, x) = x
kernel_arg(backend, grid::Grid) = convert_realtype(paramtype(backend), grid)

kernel_args(backend, y::Tuple) = map(x -> kernel_arg(backend, x), y)

@kernel function for_each_inner_cell_kernel(f, grid, direction, ghostcells, y...)
    J = @index(Global, Cartesian)
    I = toint(J)
    f(left_cell(grid, I, direction, ghostcells), middle_cell(grid, I, direction, ghostcells), right_cell(grid, I, direction, ghostcells), y...)
end


function for_each_inner_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...; ghostcells=grid.ghostcells[direction]) where {T}
    ev = for_each_inner_cell_kernel(backend.backend)(f, kernel_arg(backend, grid), direction, ghostcells, kernel_args(backend, y)..., ndrange=inner_cells(grid, direction, ghostcells))
end

@kernel function for_each_ghost_cell_kernel(f, grid, direction, y...)
    I = @index(Global, Cartesian)
    f(toint(middle_cell(grid, I, direction, 0)), y...)
end


function for_each_ghost_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...) where {T}
    ev = for_each_ghost_cell_kernel(backend.backend)(f, kernel_arg(backend, grid), direction, kernel_args(backend, y)..., ndrange=ghost_cells(grid, direction))
end


@kernel function for_each_index_value_kernel(f, values, y...)
    I = @index(Global, Cartesian)
    f(toint(I), values[I], y...)
end


function for_each_index_value(f, backend::KernelAbstractionBackend{T}, values, y...) where {T}
    # Make sure we don't have weird indexing. If we do get weird indexing, we would have to 
    # do the ndrange and @index slightly differently.
    @assert firstindex(values) == 1
    @assert lastindex(values) == length(values)
    ev = for_each_index_value_kernel(backend.backend)(f, values, kernel_args(backend, y)..., ndrange=length(values))
end


@kernel function for_each_index_value_2d_kernel(f, values1, values2, y...)
    I = @index(Global, Cartesian)
    f(Tuple(I)..., values1[I[1]], values2[I[2]], y...)
end


function for_each_index_value_2d(f, backend::KernelAbstractionBackend{T}, values1, values2, y...) where {T}
    #TODO: This could be made general by just taking a tuple of values...
    # Make sure we don't have weird indexing. If we do get weird indexing, we would have to 
    # do the ndrange and @index slightly differently.
    @assert firstindex(values1) == 1
    @assert lastindex(values1) == length(values1)
    @assert firstindex(values2) == 1
    @assert lastindex(values2) == length(values2)

    ev = for_each_index_value_2d_kernel(backend.backend)(f, values1, values2, kernel_args(backend, y)..., ndrange=(length(values1), length(values2)))
end



@kernel function for_each_cell_kernel(f, y...)
    I = @index(Global, Cartesian)
    f(toint(I), y...)
end


function for_each_cell(f, backend::KernelAbstractionBackend{T}, grid, y...;) where {T}
    # `grid` is only needed host-side, for the ndrange.
    ev = for_each_cell_kernel(backend.backend)(f, kernel_args(backend, y)..., ndrange=size(grid))
end
