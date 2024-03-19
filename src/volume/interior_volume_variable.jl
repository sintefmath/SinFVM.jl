
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


function Base.iterate(interiorVolumeVariable::InteriorVolumeVariable)
    if length(interiorVolumeVariable) == 0
        return nothing
    end
    return (interiorVolumeVariable[1], 1)
end

Base.getindex(vol::InteriorVolumeVariable, index::Int64) =
    vol._volume._volume._data[interior2full(vol._volume._volume, index), vol._index]

function Base.setindex!(vol::InteriorVolumeVariable, value, index::Int64)
    vol._volume._volume._data[interior2full(vol._volume._volume, index), vol._index] = value
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
    vol._volume._data[converted_range, vol._index] = value
end

function Base.getindex(vol::InteriorVolumeVariable, indexrange::UnitRange{Int64})
    conversion = let volume = vol._volume._volume
        index -> interior2full(volume, index)
    end
    converted_range = conversion.(indexrange)
    vol._volume._data[converted_range, vol._index]
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

function Base.iterate(vol::InteriorVolumeVariable, index::Int64)
    if index > length(vol)
        return nothing
    end
    return (vol[index], index + 1)
end


function Base.iterate(vol::InteriorVolumeVariable, state)
    index = state[2]
    if index > length(vol)
        return nothing
    end
    return (vol[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
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
Base.length(vol::InteriorVolumeVariable) = Base.size(vol._volume, 1)
Base.size(vol::InteriorVolumeVariable) = size(vol._volume)
