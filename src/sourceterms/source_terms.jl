
# Source terms are evaluated either per direction or simply non-directional but per cell
# Each source term should overload one and only one of these functions
function evaluate_directional_source_term!(::SourceTerm, output, current_state, ::ConservedSystem, ::Direction)
    nothing
end 

function evaluate_source_term!(::SourceTerm, output, current_state, ::ConservedSystem)
    nothing
end


include("source_term_bottom.jl")
include("source_term_rain.jl")
include("source_term_infiltration.jl")
