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


struct VolumeVariable{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,} <: AbstractArray{RealType,Dimension}

    _volume::Volume{EquationType,
        GridType,
        RealType,
        MatrixType,
        BackendType,
        NumberOfConservedVariables,
        Dimension,}

    _index::Int64

    function VolumeVariable(volume::Volume{A,B,C,D,E,F,G}, index) where {A,B,C,D,E,F,G}
        return new{A,B,C,D,E,F,G}(volume, index)
    end

end
linear2cartesian(vol::VolumeVariable, index) = linear2cartesian(vol._volume, index)

import Adapt
function Adapt.adapt_structure(
    to,
    volumevar::VolumeVariable{EquationType,S,T,M,B,N,D},
) where {EquationType,S,T,M,B,N,D}
    volume = Adapt.adapt_structure(to, volumevar._volume)
    #index = Adapt.adapt_structure(to, volume._index)
    VolumeVariable(volume, volumevar._index)
end


@inline function Base.propertynames(::Type{T}) where {T<:Volume}
    return variable_names(T)
end

@inline function Base.getproperty(volume::T, variable::Symbol) where {T<:Volume}
    if variable == :_data || variable == :_grid || variable == :_backend
        return getfield(volume, variable)
    end
    variable_index = findfirst(x -> x == variable, variable_names(T))
    return VolumeVariable(volume, variable_index)
end

Base.getindex(volumevariable::VolumeVariable, index::Int64) =
    volumevariable._volume._data[linear2cartesian(volumevariable, index), volumevariable._index]
function Base.setindex!(volumevariable::VolumeVariable, value, index::Int64)
    volumevariable._volume._data[linear2cartesian(volumevariable, index), volumevariable._index] = value
end

Base.getindex(volumevariable::VolumeVariable, i::Int64, j::Int64) =
    volumevariable._volume._data[i, j, volumevariable._index]
function Base.setindex!(volumevariable::VolumeVariable, value, i::Int64, j::Int64)
    volumevariable._volume._data[i, j, volumevariable._index] = value
end

function Base.setindex!(
    volumevariable::VolumeVariable,
    value::AbstractVector{S},
    indexrange::UnitRange{Int64},
) where {S}
    volumevariable._volume._data[indexrange, volumevariable._index] = value
end
function Base.setindex!(
    volumevariable::VolumeVariable,
    value::AbstractMatrix{S},
    indexrange1::UnitRange{Int64},
    indexrange2::UnitRange{Int64},
) where {S}
    value_backend = convert_to_backend(volumevariable._volume._backend, value)
    @fvmloop for_each_index_value_2d(volumevariable._volume._backend, indexrange1, indexrange2) do i_source, j_source, i_target, j_target
        volumevariable._volume._data[i_target, j_target, volumevariable._index] = value_backend[i_source, j_source]
    end
end


function Base.getindex(volumevariable::VolumeVariable, indexrange::UnitRange{Int64})
    volumevariable._volume._data[indexrange, volumevariable._index]
end

function Base.getindex(volumevariable::VolumeVariable, indexrange1::UnitRange{Int64}, indexrange2::UnitRange{Int64})
    volumevariable._volume._data[indexrange1, indexrange2, volumevariable._index]
end

Base.firstindex(volumevariable::VolumeVariable) = Base.firstindex(volumevariable._volume)
Base.lastindex(volumevariable::VolumeVariable) = Base.lastindex(volumevariable._volume)


@inline Base.similar(vol::T) where {T<:VolumeVariable} = convert_to_backend(vol._volume._backend, zeros(eltype(T), size(vol)))

@inline Base.similar(vol::VolumeVariable, type::Type{S}) where {S} =
    convert_to_backend(vol._volume._backend, zeros(type, size(vol)))

@inline Base.similar(vol::VolumeVariable, type::Type{S}, dims::Dims) where {S} =
    convert_to_backend(vol._volume._backend, zeros(type, dims))

@inline Base.similar(vol::T, dims::Dims) where {T<:VolumeVariable} =
    convert_to_backend(vol._volume._backend, zeros(eltype(T), dims))

function Base.iterate(volumevariable::VolumeVariable, index = 1)
    if index > length(volumevariable)
        return nothing
    end
    return (volumevariable[index], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{T}) where {T<:VolumeVariable} = Base.IndexCartesian()
Base.eltype(::Type{VolumeVariable{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,}}) where {EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,} = RealType
Base.length(volumevariable::VolumeVariable) = Base.length(volumevariable._volume)
Base.size(volumevariable::VolumeVariable) = size(volumevariable._volume)

collect_at_index(d::AbstractArray{T, 2}, index) where {T} = collect(d[:, index])
collect_at_index(d::AbstractArray{T, 3}, index) where {T} = collect(d[:, :, index])
Base.collect(vol::VolumeVariable) = collect_at_index(vol._volume._data, vol._index)
