function create_buffer(backend, grid::Grid, equation::Equation)
    @info "In create buffer in volume.jl"
    create_buffer(backend, number_of_conserved_variables(equation), grid.totalcells)
end


struct Volume{EquationType<:Equation,GridType,RealType,MatrixType}
    _data::MatrixType
    _grid::GridType
    function Volume(backend, equation, grid)
        @info "Creating buffer"
        buffer = create_buffer(backend, grid, equation)
        @info "Created buffer"
        new{typeof(equation),typeof(grid),eltype(buffer),typeof(buffer)}(buffer, grid)
    end
end

struct VolumeVariable{VolumeType}
    _volume::VolumeType
    _index::Int64
end

number_of_variables(::Type{Volume{EquationType,S,T,M}}) where {EquationType,S,T,M} = number_of_conserved_variables(EquationType)
variable_names(::Type{Volume{EquationType,S,T,M}}) where {EquationType,S,T,M} = conserved_variable_names(EquationType)

realtype(::Type{Volume{S,T,RealType,M}}) where {S,T,RealType,M} = RealType

Base.getindex(vol::T, index::Int64) where {T<:Volume} = SinSWE.extract_vector(Val(number_of_variables(T)), vol._data, index)
Base.setindex!(vol::T, value, index::Int64) where {T<:Volume} = SinSWE.set_vector!(Val(number_of_variables(T)), vol._data, value, index)

Base.firstindex(vol::Volume) = 1
Base.lastindex(vol::Volume) = size(vol._data, 1)


function Base.iterate(vol::Volume)
    if length(vol) == 0
        return nothing
    end
    return (vol[1], 1)
end

function Base.iterate(vol::Volume, state)
    index = state[2]
    if index > length(vol)
        return nothing
    end
    return (vol[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{T}) where {T<:Volume} = Base.IndexLinear()
Base.eltype(::Type{T}) where {T<:Volume} = SVector{number_of_variables(T),realtype(T)}
Base.length(vol::Volume) = Base.size(vol._data, 1)
Base.size(vol::Volume) = size(vol._grid)

function Base.propertynames(::Type{T}) where {T<:Volume}
    return variable_names(T)
end

function Base.getproperty(volume::T, variable::Symbol) where {T<:Volume}
    if variable == :_data || variable == :_grid
        return getfield(volume, variable)
    end
    variable_index = findfirst(x -> x == variable, variable_names(T))
    return VolumeVariable(volume, variable_index)
end




Base.getindex(volumevariable::VolumeVariable, index::Int64) = volumevariable._volume._data[index, volumevariable._index]
function Base.setindex!(volumevariable::VolumeVariable, value, index::Int64)
    volumevariable._volume._data[index, volumevariable._index] = value
end

function Base.setindex!(volumevariable::VolumeVariable, value::Vector{S}, ::UnitRange{Int64}) where {T<:Volume,S}

end

Base.firstindex(volumevariable::VolumeVariable) = Base.firstindex(volumevariable._volume)
Base.lastindex(volumevariable::VolumeVariable) = Base.lastindex(volumevariable._volume)


function Base.iterate(volumevariable::VolumeVariable)
    if length(volumevariable) == 0
        return nothing
    end
    return (volumevariable[1], 1)
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
Base.eltype(::Type{VolumeVariable{T}}) where {T<:Volume} = realtype(T)
Base.length(volumevariable::VolumeVariable) = Base.size(volumevariable._volume, 1)
Base.size(volumevariable::VolumeVariable) = size(volumevariable._volume)

