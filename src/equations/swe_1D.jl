struct ShallowWaterEquations1D{T, S} <: Equation
    B::S
    ρ::T
    g::T
    depth_cutoff::T
    desingularizing_kappa::T
    ShallowWaterEquations1D(B::BottomType; ρ=1.0, g=9.81, depth_cutoff=10^-5, desingularizing_kappa=10^-5) where {BottomType <: AbstractArray} = new{typeof(g), typeof(B)}(B, ρ, g, depth_cutoff, desingularizing_kappa)
end

function Adapt.adapt_structure(
    to,
    swe::ShallowWaterEquations1D{T, S}
) where {T, S}
    B = Adapt.adapt_structure(to, swe.B)
    ρ = Adapt.adapt_structure(to, swe.ρ)
    g = Adapt.adapt_structure(to, swe.g)
    depth_cutoff = Adapt.adapt_structure(to, swe.depth_cutoff)
    desingularizing_kappa = Adapt.adapt_structure(to, swe.desingularizing_kappa)
    
    ShallowWaterEquations1D(B; ρ=ρ, g=g, depth_cutoff=depth_cutoff, desingularizing_kappa=desingularizing_kappa)
end

ShallowWaterEquations1D(backend::Backend, grid::Grid; B=0.0, kwargs...) = ShallowWaterEquations1D(convert_to_backend(backend, constant_bottom_topography(grid, B)); kwargs...)

function desingularize(eq::ShallowWaterEquations1D, h, hu)
    # The different desingularizations are taken from 
    # Brodtkorb and Holm (2021), Coastal ocean forecasting on the GPU using a two-dimensional finite-volume scheme.  
    # Tellus A: Dynamic Meteorology and Oceanography,  73(1), p.1876341.DOI: https://doi.org/10.1080/16000870.2021.1876341
    # and the equation numbers refere to that paper

    # Eq (23):
    # h_star = (sqrt(h^4 + max(h^4, eq.desingularizing_kappa^4)))/(sqrt(2)*h)

    # Eq (24):
    # h_star = (h^2 + eq.desingularizing_kappa^2)/h

    # Eq (25):
    # h_star = (h^2 + max(h^2, eq.desingularizing_kappa^2))/(2*h)

    # Eq (26):
    h_star = sign(h)*max(abs(h), min(h^2/(2*eq.desingularizing_kappa) + eq.desingularizing_kappa/2.0, eq.desingularizing_kappa))

    return hu/h_star
end


function (eq::ShallowWaterEquations1D)(::XDIRT, h, hu)
    ρ = eq.ρ
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [
        ρ * h * u,
        ρ * h * u * u + 0.5 * ρ * g * h^2,
    ]
end

function compute_eigenvalues(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    g = eq.g
    u = desingularize(eq, h, hu)
    return @SVector [u + sqrt(g * h), u - sqrt(g * h)]
end

function compute_max_abs_eigenvalue(eq::ShallowWaterEquations1D, ::XDIRT, h, hu)
    # TODO: Use compute_eigenvalues
    g = eq.g
    u = desingularize(eq, h, hu)
    return max(abs(u + sqrt(g * h)), abs(u - sqrt(g * h)))
end
conserved_variable_names(::Type{T}) where {T<:ShallowWaterEquations1D} = (:h, :hu)
