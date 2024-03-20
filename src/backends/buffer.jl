import CUDA

convert_to_backend(backend, array) = array
convert_to_backend(backend::CUDABackend, array) = CUDA.CuArray(array)
# TODO: Do one for KA?

function create_buffer(backend, number_of_variables::Int64, spatial_resolution)
    zeros(prod(spatial_resolution), number_of_variables)
end

function create_buffer(backend::CPUBackend, number_of_variables::Int64, spatial_resolution)
    # TODO: Fixme
    # buffer = KernelAbstractions.zeros(backend.backend, prod(spatial_resolution), number_of_variables)
    buffer = zeros(prod(spatial_resolution), number_of_variables)
    buffer
end

function create_buffer(backend::CUDABackend, number_of_variables::Int64, spatial_resolution)
    CUDA.CuArray(zeros(prod(spatial_resolution), number_of_variables))
end