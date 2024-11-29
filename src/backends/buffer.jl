# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
