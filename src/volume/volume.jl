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
            dimension(grid),
            number_of_conserved_variables(equation),
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
            dimension(grid),
            number_of_conserved_variables(EquationType),
        }(
            data,
            grid,
            nothing,
        )
    end
end


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

@inline Base.getindex(vol::T, index::Int64) where {T<:Volume} =
    SinSWE.extract_vector(Val(number_of_variables(T)), vol._data, index)
@inline Base.setindex!(vol::T, value, index::Int64) where {T<:Volume} =
    SinSWE.set_vector!(Val(number_of_variables(T)), vol._data, value, index)

Base.firstindex(vol::Volume) = 1
Base.lastindex(vol::Volume) = Base.size(vol._data, 1)


function Base.iterate(vol::Volume)
    if length(vol) == 0
        return nothing
    end
    return (vol[1], 1)
end

function Base.iterate(vol::Volume, index::Int64)
    if index > length(vol)
        return nothing
    end
    return (vol[index], index + 1)
end

function Base.iterate(vol::Volume, state)
    index = state[2]
    if index > length(vol)
        return nothing
    end
    return (vol[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
@inline Base.IndexStyle(::Type{T}) where {T<:Volume} = Base.IndexLinear()
@inline Base.eltype(::Type{T}) where {T<:Volume} =
    SVector{number_of_variables(T),realtype(T)}
@inline Base.length(vol::Volume) = Base.size(vol._data, 1)
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

Base.collect(vol::Volume) = Base.collect(vol._data)


include("volume_variable.jl")
include("interior_volume.jl")
include("interior_volume_variable.jl")