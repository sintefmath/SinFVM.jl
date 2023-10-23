module Rumpetroll
    include("SwimTypeMacros.jl")
    include("conserved_variables.jl")
    include("bathymetry.jl")
    include("Friction.jl")
    include("SWEPlottingNoMakie.jl")
    include("SWEUtils.jl")
    include("double_swe_kp07_pure.jl")
    include("ValidationUtils.jl")
    include("run_swe.jl")
    export ConservedVariables, run_swe, Bathymetry
end