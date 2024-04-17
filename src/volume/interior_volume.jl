
struct InteriorVolume{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,} <: AbstractArray{SVector{NumberOfConservedVariables, RealType}, Dimension}
    _volume::Volume{EquationType,
    GridType,
    RealType,
    MatrixType,
    BackendType,
    NumberOfConservedVariables,
    Dimension,}
end
Adapt.@adapt_structure InteriorVolume

variable_names(::Type{InteriorVolume{EquationType,
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
Dimension,} = variable_names(Volume{EquationType,
GridType,
RealType,
MatrixType,
BackendType,
NumberOfConservedVariables,
Dimension,})

linear2cartesian(vol::InteriorVolume, index) = linear2cartesian(vol._volume, index)

function interior2full(grid::CartesianGrid{1}, index)
    return index + grid.ghostcells[1]
end


function interior2full(grid::CartesianGrid{2}, index)
    nx = grid.totalcells[1]
    nx_without_ghostcells = nx - 2 * grid.ghostcells[1]
    i = (index - 1) % nx_without_ghostcells
    j = (index - 1) ÷ nx_without_ghostcells


    return i + grid.ghostcells[1] + (j + grid.ghostcells[2]) * nx + 1
end



function interior2full(volume::Volume, index)
    interior2full(volume._grid, index)
end

function interior2full(grid::CartesianGrid{2}, i_inner, j_inner)
    return CartesianIndex(i_inner + grid.ghostcells[1], j_inner +  grid.ghostcells[2])
end


function interior2full(volume::Volume, i, j)
    interior2full(volume._grid, i, j)
end


function number_of_interior_cells(volume::Volume)
    number_of_interior_cells(volume._grid)
end



# TODO: One could probably combine linear2cartesian and interior2full
Base.getindex(vol::InteriorVolume, index::Int64) =
    vol._volume[Tuple(linear2cartesian(vol, interior2full(vol._volume, index)))...]
function Base.setindex!(vol::InteriorVolume, value, index::Int64)
    vol._volume[Tuple(linear2cartesian(vol, interior2full(vol._volume, index)))...] = value
end
Base.getindex(vol::InteriorVolume, i::Int64, j::Int64) =
    vol._volume[Tuple(interior2full(vol._volume, i, j))...]

function Base.setindex!(vol::InteriorVolume, value, i::Int64, j::Int64)
    vol._volume[Tuple(interior2full(vol._volume, i, j))...] = value
end


Base.firstindex(vol::InteriorVolume) = Base.firstindex(vol._volume)
Base.lastindex(vol::InteriorVolume) = Base.lastindex(vol._volume)



function Base.iterate(vol::InteriorVolume, index::Int64 = 1)
    if index > length(vol)
        return nothing
    end
    return (vol[index], index + 1)
end

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

function Base.setindex!(vol::InteriorVolume, values::Container, indices1::UnitRange{Int64}, indices2::UnitRange{Int64}) where {Container<:Union{Volume, InteriorVolume, AbstractMatrix{<:AbstractVector}}}
    # TODO: Move this for loop to the inner kernel...
    for variable_index in 1:number_of_variables(vol._volume)
        proper_volume = vol._volume
        @fvmloop for_each_index_value_2d(vol._volume._backend, indices1, indices2) do i_source, j_source, i_target, j_target
            proper_volume._data[interior2full(proper_volume, i_target, j_target), variable_index] = values[i_source, j_source][variable_index]
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

collect_interior(d::AbstractArray{T, 2}, grid) where {T} = collect(d[grid.ghostcells[1]:(end-grid.ghostcells[1]), :])
function collect_interior(d::AbstractArray{T, 3}, grid) where {T}
    start_x = grid.ghostcells[1]
    end_x = grid.ghostcells[1]
    start_y = grid.ghostcells[2]
    end_y = grid.ghostcells[2]
    collect(d[start_x:(end - end_x), start_y:(end-end_y), :])
end
Base.collect(vol::InteriorVolume) = collect_interior(vol._volume._data, vol._volume._grid)