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


struct InteriorVolumeVariable{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,} <: AbstractArray{RealType, Dimension}

    _volume::InteriorVolume{EquationType,
        GridType,
        RealType,
        MatrixType,
        BackendType,
        NumberOfConservedVariables,
        Dimension,}

    _index::Int64

    function InteriorVolumeVariable(volume::InteriorVolume{A, B, C, D, E, F, G}, index) where {A, B, C, D, E, F, G}
        return new{A, B, C, D, E, F, G}(volume, index)
    end

end

Adapt.@adapt_structure InteriorVolumeVariable

@inline function Base.propertynames(::Type{T}) where {T<:InteriorVolume}
    return variable_names(T)
end

@inline function Base.getproperty(volume::T, variable::Symbol) where {T<:InteriorVolume}
    if variable == :_volume || variable == :_index
        return getfield(volume, variable)
    end
    variable_index = findfirst(x -> x == variable, variable_names(T))
    return InteriorVolumeVariable(volume, variable_index)
end


Base.getindex(vol::InteriorVolumeVariable, index::Int64) =
    vol._volume._volume._data[linear2cartesian(vol._volume, interior2full(vol._volume._volume, index)), vol._index]

function Base.setindex!(vol::InteriorVolumeVariable, value, index::Int64)
    vol._volume._volume._data[linear2cartesian(vol._volume, interior2full(vol._volume._volume, index)), vol._index] = value
end

Base.getindex(vol::InteriorVolumeVariable, i::Int64, j::Int64) =
    vol._volume._volume._data[interior2full(vol._volume._volume, i, j), vol._index]

function Base.setindex!(vol::InteriorVolumeVariable, value, i::Int64, j::Int64)
    vol._volume._volume._data[interior2full(vol._volume._volume, i, j), vol._index] = value
end

function Base.setindex!(
    vol::InteriorVolumeVariable,
    value::AbstractVector{S},
    indexrange::UnitRange{Int64},
) where {S}
    conversion = let volume = vol._volume._volume
        index -> interior2full(volume, index)
    end
    converted_range = conversion.(indexrange)
    vol._volume._volume._data[converted_range, vol._index] = value
end

function Base.getindex(vol::InteriorVolumeVariable, indexrange::UnitRange{Int64})
    conversion = let volume = vol._volume._volume
        index -> interior2full(volume, index)
    end
    converted_range = conversion.(indexrange)
    vol._volume._volume._data[converted_range, vol._index]
end


function Base.setindex!(
    vol::InteriorVolumeVariable,
    value::AbstractMatrix{S},
    indexrange1::UnitRange{Int64},
    indexrange2::UnitRange{Int64},
) where {S}
    conversion1 = let grid = vol._volume._volume._grid
        index -> index + grid.ghostcells[1]
    end
    converted_range1 = conversion1.(indexrange1)

    conversion2 = let grid = vol._volume._volume._grid
        index -> index + grid.ghostcells[2]
    end
    converted_range2 = conversion2.(indexrange2)
    vol._volume._volume._data[converted_range1, converted_range2, vol._index] = value
end

function Base.getindex(
    vol::InteriorVolumeVariable, 
    indexrange1::UnitRange{Int64},
    indexrange2::UnitRange{Int64},)
    conversion1 = let grid = vol._volume._volume._grid
        index -> index + grid.ghostcells[1]
    end
    converted_range1 = conversion1.(indexrange1)

    conversion2 = let grid = vol._volume._volume._grid
        index -> index + grid.ghostcells[2]
    end
    converted_range2 = conversion2.(indexrange2)
    vol._volume._data[converted_range1, converted_range2, vol._index]
end

Base.firstindex(vol::InteriorVolumeVariable) = Base.firstindex(vol._volume)
Base.lastindex(vol::InteriorVolumeVariable) = Base.lastindex(vol._volume)


@inline Base.similar(vol::T) where {T<: InteriorVolumeVariable} = convert_to_backend(vol._volume._volume._backend, zeros(eltype(T), size(vol)))

@inline Base.similar(vol::InteriorVolumeVariable, type::Type{S}) where {S} =
    convert_to_backend(vol._volume._volume._backend, zeros(type, size(vol)))

@inline Base.similar(vol::InteriorVolumeVariable, type::Type{S}, dims::Dims) where {S} =
    convert_to_backend(vol._volume._volume._backend, zeros(type, dims))

@inline Base.similar(vol::T, dims::Dims) where {T<: InteriorVolumeVariable}=
    convert_to_backend(vol._volume._volume._backend, zeros(eltype(T), dims))

function Base.iterate(vol::InteriorVolumeVariable, index::Int64 = 1)
    if index > length(vol)
        return nothing
    end
    return (vol[index], index + 1)
end

Base.IndexStyle(::Type{T}) where {T<:InteriorVolumeVariable} = Base.IndexLinear()
Base.eltype(::Type{InteriorVolumeVariable{EquationType,
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
Base.length(vol::InteriorVolumeVariable) = length(vol._volume)
Base.size(vol::InteriorVolumeVariable) = size(vol._volume)
collect_interior(d::AbstractArray{T, 2}, grid, index) where {T} = collect(d[(grid.ghostcells[1]+1):(end-grid.ghostcells[1]), index])
function collect_interior(d::AbstractArray{T, 3}, grid, index) where {T}
    start_x = grid.ghostcells[1] + 1
    end_x = grid.ghostcells[1]
    start_y = grid.ghostcells[2] + 1
    end_y = grid.ghostcells[2]
    collect(d[start_x:(end - end_x), start_y:(end-end_y), index])
end
Base.collect(vol::InteriorVolumeVariable) = collect_interior(vol._volume._volume._data, vol._volume._volume._grid, vol._index)
