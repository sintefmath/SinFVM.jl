
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


function Base.iterate(volumevariable::VolumeVariable)
    if length(volumevariable) == 0
        return nothing
    end
    return (volumevariable[1], 1)
end

Base.getindex(volumevariable::VolumeVariable, index::Int64) =
    volumevariable._volume._data[index, volumevariable._index]
function Base.setindex!(volumevariable::VolumeVariable, value, index::Int64)
    volumevariable._volume._data[index, volumevariable._index] = value
end

function Base.setindex!(
    volumevariable::VolumeVariable,
    value::AbstractVector{S},
    indexrange::UnitRange{Int64},
) where {S}
    volumevariable._volume._data[indexrange, volumevariable._index] = value
end

function Base.getindex(volumevariable::VolumeVariable, indexrange::UnitRange{Int64})
    volumevariable._volume._data[indexrange, volumevariable._index]
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

function Base.iterate(volumevariable::VolumeVariable, index::Int64)
    if index > length(volumevariable)
        return nothing
    end
    return (volumevariable[index], index + 1)
end


function Base.iterate(volumevariable::VolumeVariable, state)
    index = state[2]
    if index > length(volumevariable)
        return nothing
    end
    return (volumevariable[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{T}) where {T<:VolumeVariable} = Base.IndexLinear()
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
Base.length(volumevariable::VolumeVariable) = Base.size(volumevariable._volume, 1)
Base.size(volumevariable::VolumeVariable) = size(volumevariable._volume)

Base.collect(vol::VolumeVariable) = Base.collect(vol._volume._data[:, vol._index])
