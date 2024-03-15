
struct InteriorVolume{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,} <: AbstractArray{SVector{NumberOfConservedVariables, RealType}, NumberOfConservedVariables}
    _volume::Volume{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,}
end

function interior2full(grid::CartesianGrid{1}, index)
    return index + grid.ghostcells[1]
end


function interior2full(grid::CartesianGrid{2}, index)
    nx = grid.totalcells[1]
    nx_without_ghostcells = nx - grid.ghostcells[1]
    i = (index - 1) % nx_without_ghostcells
    j = (index - 1) ÷ nx_without_ghostcells


    return i + ghost.ghostcells[1] + (j + grid.ghostcells[2]) * nx + 1
end

function interior2full(volume::Volume, index)
    interior2full(volume._grid, index)
end

function number_of_interior_cells(volume::Volume)
    number_of_interior_cells(volume._grid)
end



Base.getindex(vol::InteriorVolume, index::Int64) =
    vol._volume[interior2full(vol._volume, index)]
function Base.setindex!(vol::InteriorVolume, value, index::Int64)
    vol._volume[interior2full(vol._volume, index)] = value
end

Base.firstindex(vol::InteriorVolume) = Base.firstindex(vol._volume)
Base.lastindex(vol::InteriorVolume) = Base.lastindex(vol._volume)


function Base.iterate(vol::InteriorVolume)
    if length(vol) == 0
        return nothing
    end
    return (vol[1], 1)
end


function Base.iterate(vol::InteriorVolume, index::Int64)
    if index > length(vol)
        return nothing
    end
    return (vol[index], index + 1)
end
function Base.iterate(vol::InteriorVolume, state)
    index = state[2]
    if index > length(vol)
        return nothing
    end
    return (vol[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{InteriorVolume{EquationType,
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
Dimension,} = Base.IndexStyle(Volume{EquationType,
GridType,
RealType,
MatrixType,
BackendType,
NumberOfConservedVariables,
Dimension,})

Base.eltype(::Type{InteriorVolume{EquationType,
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
Dimension,} = Base.eltype(Volume{EquationType,
GridType,
RealType,
MatrixType,
BackendType,
NumberOfConservedVariables,
Dimension,})


Base.length(vol::InteriorVolume) = number_of_interior_cells(vol._volume)
Base.size(vol::InteriorVolume) = interior_size(vol._volume._grid)
function Base.setindex!(vol::InteriorVolume, values::Container, indices::UnitRange{Int64}) where {Container<:AbstractVector{<:AbstractVector}}
    # TODO: Move this for loop to the inner kernel...
    for j in 1:number_of_variables(vol._volume)
        proper_volume = vol._volume
        @fvmloop for_each_index_value(vol._volume._backend, indices) do index_source, index_target
            proper_volume._data[interior2full(proper_volume, index_target), j] = values[index_source][j]
        end
    end
end


@inline Base.similar(vol::InteriorVolume) = similar(vol._volume._data, size(vol))

@inline Base.similar(vol::InteriorVolume, type::Type{S}) where {S} =
    similar(vol._volume._data, type, size(vol))

@inline Base.similar(vol::InteriorVolume, type::Type{S}, dims::Dims) where {S} =
    similar(vol._volume, type, dims)

@inline Base.similar(vol::InteriorVolume, dims::Dims) =
    similar(vol._volume, dims)
    
# function Base.propertynames(::Type{InteriorVolume{T}}) where {T<:Volume}
#     return variable_names(T)
# end

# function Base.getproperty(volume::InteriorVolume{T}, variable::Symbol) where {T<:Volume}
#     if variable == :_data || variable == :_grid
#         return getfield(volume, variable)
#     end
#     variable_index = findfirst(x -> x == variable, variable_names(T))
#     return VolumeVariable(volume, variable_index)
# end