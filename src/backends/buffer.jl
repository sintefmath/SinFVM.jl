import CUDA

convert_to_backend(backend, array::AbstractArray) = array
convert_to_backend(backend::CUDABackend, array::AbstractArray) = CUDA.CuArray(array)
convert_to_backend(backend::CPUBackend, array::CUDA.CuArray) = collect(array)

# TODO: Do one for KA?

function create_buffer(backend, number_of_variables::Int64, spatial_resolution)
    zeros(backend.realtype, spatial_resolution..., number_of_variables)
end

function create_buffer(backend::CPUBackend, number_of_variables::Int64, spatial_resolution)
    # TODO: Fixme
    # buffer = KernelAbstractions.zeros(backend.backend, prod(spatial_resolution), number_of_variables)
    buffer = zeros(backend.realtype, spatial_resolution..., number_of_variables)
    buffer
end

function create_buffer(backend::CUDABackend, number_of_variables::Int64, spatial_resolution)
    CUDA.CuArray(zeros(backend.realtype, spatial_resolution..., number_of_variables))
end