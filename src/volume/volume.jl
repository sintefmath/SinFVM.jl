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

function create_buffer(backend, grid::Grid, equation::Equation)
    create_buffer(backend, number_of_conserved_variables(equation), grid.totalcells)
end
struct Volume{
    EquationType<:Equation,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,
} <: AbstractArray{SVector{NumberOfConservedVariables,RealType},Dimension}
    _data::MatrixType
    _grid::GridType
    _backend::BackendType
    function Volume(backend, equation::Equation, grid::Grid)
        buffer = create_buffer(backend, grid, equation)
        new{
            typeof(equation),
            typeof(grid),
            eltype(buffer),
            typeof(buffer),
            typeof(backend),
            number_of_conserved_variables(equation),
            dimension(grid),
        }(
            buffer,
            grid,
            backend,
        )
    end

    function Volume(EquationType::Type{<:Equation}, data::AbstractArray, grid::Grid)
        # Only for internal use. Note that this is only used when
        # transferring this struct to the GPU, and there we do not need access to the backend.
        # Hence, to avoid headache, we simply leave that at nothing.
        new{
            EquationType,
            typeof(grid),
            eltype(data),
            typeof(data),
            Nothing,
            number_of_conserved_variables(EquationType),
            dimension(grid),
        }(
            data,
            grid,
            nothing,
        )
    end
end

linear2cartesian(::Tuple{Int64}, index) = index
linear2cartesian(s::Tuple{Int64, Int64}, index) = CartesianIndex((index - 1) % s[1] + 1, (index - 1) ÷ s[2] + 1)
linear2cartesian(vol::Volume, index) = linear2cartesian(size(vol), index)

import Adapt
function Adapt.adapt_structure(
    to,
    volume::Volume{EquationType,S,T,M,B,N,D},
) where {EquationType,S,T,M,B,N,D}
    data = Adapt.adapt_structure(to, volume._data)
    grid = Adapt.adapt_structure(to, volume._grid)
    Volume(EquationType, data, grid)
end

number_of_variables(
    ::Type{Volume{EquationType,S,T,M,B,N,D}},
) where {EquationType,S,T,M,B,N,D} = number_of_conserved_variables(EquationType)

number_of_variables(::T) where {T<:Volume} = number_of_variables(T)

variable_names(::Type{Volume{EquationType,S,T,M,B,N,D}}) where {EquationType,S,T,M,B,N,D} =
    conserved_variable_names(EquationType)

realtype(::Type{Volume{S,T,RealType,M,B,N,D}}) where {S,T,RealType,M,B,N,D} = RealType

@inline function Base.getindex(vol::T, index::Int64) where {T<:Volume}
    SinFVM.extract_vector(Val(number_of_variables(T)), vol._data, linear2cartesian(vol, index))
end
@inline function Base.setindex!(vol::T, value, index::Int64) where {T<:Volume}
    SinFVM.set_vector!(Val(number_of_variables(T)), vol._data, value, linear2cartesian(vol, index))
end

@inline Base.getindex(vol::T, i::Int64, j::Int64) where {T<:Volume} =
    SinFVM.extract_vector(Val(number_of_variables(T)), vol._data, CartesianIndex(i, j))
@inline Base.setindex!(vol::T, value, i::Int64, j::Int64) where {T<:Volume} =
    SinFVM.set_vector!(Val(number_of_variables(T)), vol._data, value, CartesianIndex(i, j))


Base.firstindex(vol::Volume) = 1
Base.lastindex(vol::Volume) = Base.length(vol)

function Base.iterate(vol::Volume, index = 1)
    if index > length(vol)
        return nothing
    end
    nextindex = index + 1
    
    return (vol[index], nextindex)
end

@inline Base.IndexStyle(::Type{T}) where {T<:Volume} = Base.IndexCartesian()
@inline Base.eltype(::Type{T}) where {T<:Volume} =
    SVector{number_of_variables(T),realtype(T)}

@inline Base.length(vol::Volume) = prod(Base.size(vol._grid))
@inline Base.size(vol::Volume) = Base.size(vol._grid)
@inline Base.size(vol::Volume, i::Int64) = (Base.size(vol._grid)[i],)

Base.similar(vol::Volume) = convert_to_backend(vol._backend, similar(vol._data))
Base.similar(vol::Volume, type::Type{S}) where {S} =
    convert_to_backend(vol._backend, similar(vol._data, type))
Base.similar(vol::Volume, type::Type{S}, dims::Dims) where {S} =
    convert_to_backend(vol._backend, similar(vol._data, type, dims))
Base.similar(vol::Volume, dims::Dims) =
    convert_to_backend(vol._backend, similar(vol._data, dims))

function Base.setindex!(
    vol::T,
    values::Container,
    indices::UnitRange{Int64},
) where {T<:Volume,Container<:AbstractVector{<:AbstractVector}}
    # TODO: Move this for loop to the inner kernel... Current limitation in the @fvmloop makes this hard.
    values_backend = convert_to_backend(vol._backend, values)
    for j = 1:number_of_variables(T)
        @fvmloop for_each_index_value(vol._backend, indices) do index_source, index_target
            vol._data[index_target, j] = values_backend[index_source][j]
        end
    end
end

function Base.setindex!(
    vol::T,
    values::Container,
    indices1::UnitRange{Int64},
    indices2::UnitRange{Int64},
) where {T<:Volume,Container<:AbstractMatrix{<:AbstractVector}}
    # TODO: Move this for loop to the inner kernel... Current limitation in the @fvmloop makes this hard.
    values_backend = convert_to_backend(vol._backend, values)
    
    for variable_index = 1:number_of_variables(T)
        @fvmloop for_each_index_value_2d(vol._backend, indices1, indices2) do i_source, j_source, i_target, j_target
            vol._data[i_target, j_target, variable_index] = values_backend[i_source, j_source][variable_index]
        end
    end
end

Base.collect(vol::Volume) = Base.collect(vol._data)


include("volume_variable.jl")
include("interior_volume.jl")
include("interior_volume_variable.jl")

convert_to_backend(::CUDABackend, vol::Volume{A, B, C, D, <: CUDABackend, F, G}) where {A, B, C, D, F, G} = vol
convert_to_backend(::CPUBackend, vol::Volume{A, B, C, D, <: CPUBackend, F, G}) where {A, B, C, D, F, G} = vol

convert_to_backend(::CUDABackend, vol::InteriorVolume{A, B, C, D, <: CUDABackend, F, G}) where {A, B, C, D, F, G} = vol
convert_to_backend(::CPUBackend, vol::InteriorVolume{A, B, C, D, <: CPUBackend, F, G}) where {A, B, C, D, F, G} = vol

convert_to_backend(::CUDABackend, vol::VolumeVariable{A, B, C, D, <: CUDABackend, F, G}) where {A, B, C, D, F, G} = vol
convert_to_backend(::CPUBackend, vol::VolumeVariable{A, B, C, D, <: CPUBackend, F, G}) where {A, B, C, D, F, G} = vol

convert_to_backend(::CUDABackend, vol::InteriorVolumeVariable{A, B, C, D, <: CUDABackend, F, G}) where {A, B, C, D, F, G} = vol
convert_to_backend(::CPUBackend, vol::InteriorVolumeVariable{A, B, C, D, <: CPUBackend, F, G}) where {A, B, C, D, F, G} = vol
