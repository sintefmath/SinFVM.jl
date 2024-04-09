number_of_conserved_variables(::Type{T}) where {T} = error("This is not an equation type.")
number_of_conserved_variables(::T) where {T<:Equation} = number_of_conserved_variables(T)
number_of_conserved_variables(::Type{T}) where {T<:Equation} = length(conserved_variable_names(T))


include("burgers.jl")
include("swe_1D_pure.jl")
include("swe_1D.jl")
include("swe_2D_pure.jl")
include("swe_2D.jl")

AllSWE = Union{ShallowWaterEquations1D,ShallowWaterEquations1DPure,ShallowWaterEquationsPure, ShallowWaterEquations}
AllPracticalSWE = Union{ShallowWaterEquations1D, ShallowWaterEquations}
AllSWE1D = Union{ShallowWaterEquations1D, ShallowWaterEquations1DPure}
AllSWE2D = Union{ShallowWaterEquations, ShallowWaterEquationsPure}

function desingularize(eq::AllPracticalSWE, h, momentum)
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
    h_star = copysign(1, h)*max(abs(h), min(h^2/(2*eq.desingularizing_kappa) + eq.desingularizing_kappa/2.0, eq.desingularizing_kappa))
    # h_star = sign(h)*max(abs(h), min(h^2/(2*eq.desingularizing_kappa) + eq.desingularizing_kappa/2.0, eq.desingularizing_kappa))
    # if h < 0.0
    #     h_star = 0.5*eq.desingularizing_kappa
    # end
    return momentum/h_star
end