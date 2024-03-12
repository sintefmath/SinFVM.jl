using KernelAbstractions
import CUDA


struct KernelAbstractionBackend{KABackendType}
    backend::KABackendType
end

make_cuda_backend() = KernelAbstractionBackend(get_backend(CUDA.cu(ones(3))))
make_cpu_backend() = KernelAbstractionBackend(get_backend(ones(3)))
const CUDABackend = KernelAbstractionBackend{CUDA.CUDAKernels.CUDABackend}
const CPUBackend = KernelAbstractionBackend{KernelAbstractions.CPU}

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
    I = @index(Global)
    f(left_cell(grid, I, direction, ghostcells), middle_cell(grid, I, direction, ghostcells), right_cell(grid, I, direction, ghostcells), y...)
end


function for_each_inner_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...; ghostcells=grid.ghostcells[direction]) where {T}
    ev = for_each_inner_cell_kernel(backend.backend, 1024)(f, grid, direction, ghostcells, y..., ndrange=inner_cells(grid, direction, ghostcells))
end

@kernel function for_each_ghost_cell_kernel(f, grid, direction, y...)
    I = @index(Global)
    f(I, y...)
end


function for_each_ghost_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...) where {T}
    ev = for_each_ghost_cell_kernel(backend.backend, 1024)(f, grid, direction, y..., ndrange=ghost_cells(grid, direction))
end


@kernel function for_each_index_value_kernel(f, values, y...)
    I = @index(Global)
    f(I, values[I], y...)
end


function for_each_index_value(f, backend::KernelAbstractionBackend{T}, values, y...) where {T}
    # Make sure we don't have weird indexing. If we do get weird indexing, we would have to 
    # do the ndrange and @index slightly differently.
    @assert firstindex(values) == 1
    @assert lastindex(values) == length(values)
    ev = for_each_index_value_kernel(backend.backend, 1024)(f, values, y..., ndrange=length(values))
end
