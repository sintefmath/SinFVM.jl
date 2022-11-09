module Rumpetroll
    include("SwimTypeMacros.jl")
    include("Friction.jl")
    include("swe_kp07_pure.jl")
    include("double_swe_kp07_pure.jl")
    include("run_swe.jl")
end