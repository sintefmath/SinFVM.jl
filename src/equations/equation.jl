number_of_conserved_variables(::Type{T}) where {T} = error("This is not an equation type.")
number_of_conserved_variables(::T) where {T<:Equation} = number_of_conserved_variables(T)
number_of_conserved_variables(::Type{T}) where {T<:Equation} = length(conserved_variable_names(T))


include("burgers.jl")
include("swe_1D_pure.jl")
include("swe_1D.jl")
include("swe_2D_pure.jl")
include("swe_2D.jl")

const AllSWE = Union{ShallowWaterEquations1D, ShallowWaterEquations1DPure} #, ShallowWaterEquations2D, ShallowWaterEquations2DPure}
const AllPracticalSWE = Union{ShallowWaterEquations1D} #, ShallowWaterEquations2D}