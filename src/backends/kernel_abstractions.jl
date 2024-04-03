using KernelAbstractions
import CUDA

abstract type Backend end

toint(x) = x
toint(x::CartesianIndex{1}) = x[1]

struct KernelAbstractionBackend{KABackendType} <: Backend
    backend::KABackendType
end

make_cuda_backend() = KernelAbstractionBackend(get_backend(CUDA.cu(ones(3))))
make_cpu_backend() = KernelAbstractionBackend(get_backend(ones(3)))
const CUDABackend = KernelAbstractionBackend{CUDA.CUDAKernels.CUDABackend}
const CPUBackend = KernelAbstractionBackend{KernelAbstractions.CPU}

name(::CUDABackend) = "CUDA"
name(::CPUBackend) = "CPU"

function get_available_backends()
    backends = Any[make_cpu_backend()]

    try
        cuda_backend = make_cuda_backend()
        push!(backends, cuda_backend)
    catch err
        @show err
    end
    return backends
end

@kernel function for_each_inner_cell_kernel(f, grid, direction, ghostcells, y...)
    J = @index(Global, Cartesian)
    I = toint(J)
    f(left_cell(grid, I, direction, ghostcells), middle_cell(grid, I, direction, ghostcells), right_cell(grid, I, direction, ghostcells), y...)
end


function for_each_inner_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...; ghostcells=grid.ghostcells[direction]) where {T}
    ev = for_each_inner_cell_kernel(backend.backend, 1024)(f, grid, direction, ghostcells, y..., ndrange=inner_cells(grid, direction, ghostcells))
end

@kernel function for_each_ghost_cell_kernel(f, grid, direction, y...)
    I = @index(Global, Cartesian)
    f(toint(middle_cell(grid, I, direction, 0)), y...)
end


function for_each_ghost_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...) where {T}
    ev = for_each_ghost_cell_kernel(backend.backend, 1024)(f, grid, direction, y..., ndrange=ghost_cells(grid, direction))
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
    ev = for_each_index_value_kernel(backend.backend, 1024)(f, values, y..., ndrange=length(values))
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

    ev = for_each_index_value_2d_kernel(backend.backend, 1024)(f, values1, values2, y..., ndrange=(length(values1), length(values2)))
end



@kernel function for_each_cell_kernel(f, grid, y...)
    I = @index(Global, Cartesian)
    f(I, y...)
end


function for_each_cell(f, backend::KernelAbstractionBackend{T}, grid, y...;) where {T}
    ev = for_each_cell_kernel(backend.backend, 1024)(f, grid, y..., ndrange=size(grid))
end