
struct VolumeVariable{VolumeType}
    _volume::VolumeType
    _index::Int64
end

function Base.iterate(volumevariable::VolumeVariable)
    if length(volumevariable) == 0
        return nothing
    end
    return (volumevariable[1], 1)
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



function Base.iterate(volumevariable::VolumeVariable, state)
    index = state[2]
    if index > length(volumevariable)
        return nothing
    end
    return (volumevariable[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{T}) where {T<:VolumeVariable} = Base.IndexLinear()
Base.eltype(::Type{VolumeVariable{T}}) where {T<:Volume} = realtype(T)
Base.length(volumevariable::VolumeVariable) = Base.size(volumevariable._volume, 1)
Base.size(volumevariable::VolumeVariable) = size(volumevariable._volume)
