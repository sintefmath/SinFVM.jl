import CUDA

convert_to_backend(backend, array) = array
convert_to_backend(backend::CUDABackend, array) = CUDA.cu(array)
# TODO: Do one for KA?

function create_buffer(backend, number_of_variables::Int64, spatial_resolution)
    @info "In dummy create_buffer"
    zeros(prod(spatial_resolution), number_of_variables)
end

function create_buffer(backend::CPUBackend, number_of_variables::Int64, spatial_resolution)
    @info "In KA create_buffer" prod(spatial_resolution) number_of_variables
    # TODO: Fixme
    # buffer = KernelAbstractions.zeros(backend.backend, prod(spatial_resolution), number_of_variables)
    buffer = zeros(prod(spatial_resolution), number_of_variables)
    @info "Created buffer" buffer
    buffer
end

function create_buffer(backend::CUDABackend, number_of_variables::Int64, spatial_resolution)
    @info "In CUDA create_buffer"
    CUDA.cu(zeros(prod(spatial_resolution), number_of_variables))
end