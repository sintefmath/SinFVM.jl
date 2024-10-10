include("swe1d.jl")
include("swe2d.jl")
const AllPracticalSWEStable = Union{ShallowWaterEquations1DStable, ShallowWaterEquationsStable} 
desingularize(::AllPracticalSWEStable, h) = h # TODO: Do we want to something more here?
