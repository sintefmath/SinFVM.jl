
struct InteriorVolume{VolumeType<:Volume}
    _volume::VolumeType
end

function interior2full(grid::CartesianGrid{1}, index)
    return index + grid.ghostcells[1]
end


function interior2full(grid::CartesianGrid{2}, index)
    nx = grid.totalcells[1]
    nx_without_ghostcells = nx - grid.ghostcells[1]
    i = index % nx_without_ghostcells
    j = index ÷ nx_without_ghostcells


    return i + ghost.ghostcells[1] + (j + grid.ghostcells[2]) * nx
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

function Base.iterate(vol::InteriorVolume, state)
    index = state[2]
    if index > length(vol)
        return nothing
    end
    return (vol[state[2]], index + 1)
end

# TODO: Support Cartesian indexing
Base.IndexStyle(::Type{InteriorVolume{T}}) where {T<:Volume} = Base.IndexStyle(T)
Base.eltype(::Type{InteriorVolume{T}}) where {T<:Volume} = Base.eltype(T)
Base.length(vol::InteriorVolume) = number_of_interior_cells(vol._volume)
Base.size(vol::InteriorVolume) = interior_size(vol._volume)

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