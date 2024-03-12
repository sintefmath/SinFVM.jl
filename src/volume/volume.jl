function create_buffer(backend, grid::Grid, equation::Equation)
    @info "In create buffer in volume.jl"
    create_buffer(backend, number_of_conserved_variables(equation), grid.totalcells)
end


struct Volume{EquationType<:Equation,GridType,RealType,MatrixType,BackendType}
    _data::MatrixType
    _grid::GridType
    _backend::BackendType
    function Volume(backend, equation::Equation, grid::Grid)
        @info "Creating buffer"
        buffer = create_buffer(backend, grid, equation)
        @info "Created buffer"
        new{typeof(equation),typeof(grid),eltype(buffer),typeof(buffer),typeof(backend)}(
            buffer,
            grid,
            backend,
        )
    end

    function Volume(EquationType::Type{<:Equation}, data::AbstractArray, grid::Grid)
        # Only for internal use. Note that this is only used when
        # transferring this struct to the GPU, and there we do not need access to the backend.
        # Hence, to avoid headache, we simply leave that at nothing.
        new{EquationType,typeof(grid),eltype(data),typeof(data),Nothing}(
            data,
            grid,
            nothing,
        )
    end
end


import Adapt
function Adapt.adapt_structure(to, volume::Volume{EquationType,S,T,M,B}) where {EquationType,S,T,M,B}
    data = Adapt.adapt_structure(to, volume._data)
    grid = Adapt.adapt_structure(to, volume._grid)
    Volume(EquationType, data, grid)
end

number_of_variables(::Type{Volume{EquationType,S,T,M,B}}) where {EquationType,S,T,M,B} =
    number_of_conserved_variables(EquationType)
variable_names(::Type{Volume{EquationType,S,T,M,B}}) where {EquationType,S,T,M,B} =
    conserved_variable_names(EquationType)

realtype(::Type{Volume{S,T,RealType,M,B}}) where {S,T,RealType,M,B} = RealType

Base.getindex(vol::T, index::Int64) where {T<:Volume} =
    SinSWE.extract_vector(Val(number_of_variables(T)), vol._data, index)
Base.setindex!(vol::T, value, index::Int64) where {T<:Volume} =
    SinSWE.set_vector!(Val(number_of_variables(T)), vol._data, value, index)

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


function Base.setindex!(vol::T, values::Container, indices::UnitRange{Int64}) where {T<:Volume,Container<:AbstractVector{<:AbstractVector}}
    for j in 1:number_of_variables(T)

        @fvmloop for_each_index_value(vol._backend, indices) do index_source, index_target
            vol._data[index_target, j] = values[index_source][j]
        end
    end
end

include("volume_variable.jl")
include("interior_volume.jl")