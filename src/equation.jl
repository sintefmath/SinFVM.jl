number_of_conserved_variables(::Type{T}) where {T} = error("This is not an equation type.")
number_of_conserved_variables(::T) where {T<:Equation} = number_of_conserved_variables(T)
number_of_conserved_variables(::Type{T}) where {T<:Equation} = length(conserved_variable_names(T))
struct ShallowWaterEquations1D{T, S} <: Equation
    B::S
    ρ::T
    g::T
    depth_cutoff::T
    flux_slope_eps::T
    ShallowWaterEquations1D(B::BottomType; ρ=1.0, g=9.81, depth_cutoff=10^-5, flux_slope_eps=10^-4) where {BottomType <: AbstractArray} = new{typeof(g), typeof(B)}(B, ρ, g, depth_cutoff, flux_slope_eps)
end
Adapt.@adapt_structure ShallowWaterEquations1D

# ShallowWaterEquations1D(B::AbstractArray) = ShallowWaterEquations1D(B)
ShallowWaterEquations1D(grid::Grid) = ShallowWaterEquations1D(constant_bottom_topography(grid, 0.0))
ShallowWaterEquations1D(backend::Backend, grid::Grid) = ShallowWaterEquations1D(convert_to_backend(backend, constant_bottom_topography(grid, 0.0)))

function (eq::ShallowWaterEquations1D)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
    ]
end

function compute_eigenvalues(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    g = eq.g
    u = hu / h
    return @SVector [u + sqrt(g * h), u - sqrt(g * h)]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    # TODO: Use compute_eigenvalues
    g = eq.g
    u = hu / h
    return max(abs(u + sqrt(g * h)), abs(u - sqrt(g * h)))
end
conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations1D} = (:h, :hu)

struct ShallowWaterEquations{T} <: Equation
    ρ::T
    g::T
end

ShallowWaterEquations() = ShallowWaterEquations(1.0, 9.81)

function (eq::ShallowWaterEquations)(::XDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hu,
        ρ * hu * hu / h + 0.5 * ρ * g * h^2,
        ρ * hu * hv / h
    ]
end

function (eq::ShallowWaterEquations)(::YDIRT, h, hu, hv)
    ρ = eq.ρ
    g = eq.g
    return @SVector [
        ρ * hv,
        ρ * hu * hv / h,
        ρ * hv * hv / h + 0.5 * ρ * g * h^2,
    ]
end

struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = @SVector [0.5 * u .^ 2]

compute_max_abs_eigenvalue(::Burgers, ::XDIRT, u) = abs(first(u))
conserved_variable_names(::Type{Burgers}) = (:u,)
