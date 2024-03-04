using KernelAbstractions
import CUDA


struct KernelAbstractionBackend{KABackendType}
    backend::KABackendType
end

make_cuda_backend() = KernelAbstractionBackend(get_backend(CUDA.cu(ones(3))))
make_cpu_backend() = KernelAbstractionBackend(get_backend(ones(3)))
const CUDABackend = KernelAbstractionBackend{CUDA.CUDAKernels.CUDABackend}



@kernel function for_each_inner_cell_kernel(f, grid, direction, y...)
    I =@index(Global)
    f(left_cell(grid, I, direction), middle_cell(grid, I, direction), right_cell(grid, I, direction), y...)
end


function for_each_inner_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...) where {T}
    ev = for_each_inner_cell_kernel(backend.backend, 1024)(f, grid, direction, y..., ndrange=inner_cells(grid, direction))
end

@kernel function for_each_ghost_cell_kernel(f, grid, direction, y...)
    I =@index(Global)
    f(I, y...)
end


function for_each_ghost_cell(f, backend::KernelAbstractionBackend{T}, grid, direction, y...) where {T}
    ev = for_each_ghost_cell_kernel(backend.backend, 1024)(f, grid, direction, y..., ndrange=ghost_cells(grid, direction))
end