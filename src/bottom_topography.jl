abstract type AbstractBottomTopography end


struct ConstantBottomTopography{T} <: AbstractBottomTopography
    B::T
    ConstantBottomTopography(B=0.0) = new{typeof(B)}(B)
end
Adapt.@adapt_structure ConstantBottomTopography


struct BottomTopography1D{T} <: AbstractBottomTopography
    B::T
    function BottomTopography1D(B, backend, grid)
        # TODO: If B is only defined for interior, should we then extend B?
        validate_bottom_topography(B, grid)
        B = convert_to_backend(backend, B)
        return new{typeof(B)}(B)
    end
end
Adapt.@adapt_structure BottomTopography1D

struct BottomTopography2D{T} <: AbstractBottomTopography
    B::T
    function BottomTopography2D(B, backend, grid)
        validate_bottom_topography(B, grid)
        B = convert_to_backend(backend, B)
        return new{typeof(B)}(B)
    end
end
Adapt.@adapt_structure BottomTopography2D

## Validate
function validate(::ConstantBottomTopography, ::Grid)
    nothing
end

function validate(bottom::AbstractBottomTopography, grid::Grid)
    validate_bottom_topography(bottom.B, grid)
end

function validate_bottom_topography(B, grid::Grid)
    if size(B) != size(grid) .+1
        throw(DomainError("Bottom topography should be of size $(size(grid) .+ 1) but got $(size(B))"))
    end
end

# Lookup for ConstantBottomTopography
B_cell(B::ConstantBottomTopography, i...) = B.B
B_face_minus(B::ConstantBottomTopography, i...) = B.B
B_face_plus(B::ConstantBottomTopography, i...) = B.B

# Lookup for BottomTopography1D
B_cell(B::BottomTopography1D, index) = 0.5*(B.B[index] + B.B[index + 1])
B_face_minus(B::BottomTopography1D, index) = B.B[index]
B_face_plus(B::BottomTopography1D, index) = B.B[index + 1]

# Lookup for BottomTopography2D
B_cell(B::BottomTopography2D, x, y) = 0.25*(B.B[x, y] + B.B[x+1, y] + B.B[x, y + 1] + B.B[x + 1, y + 1])
B_face_minus(B::BottomTopography2D, x, y, ::XDIRT) = 0.5*(B.B[x, y] + B.B[x, y + 1])
B_face_plus(B::BottomTopography2D, x, y, ::XDIRT) = 0.5*(B.B[x + 1, y] + B.B[x + 1, y + 1])
B_face_minus(B::BottomTopography2D, x, y, ::YDIRT) = 0.5*(B.B[x, y] + B.B[x + 1, y])
B_face_plus(B::BottomTopography2D, x, y, ::YDIRT) = 0.5*(B.B[x, y + 1] + B.B[x + 1, y + 1])

