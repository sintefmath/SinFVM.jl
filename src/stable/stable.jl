include("swe.jl")
include("numflux.jl")
include("hll.jl")
include("reconstruct.jl")


function post_proc_substep!(output, system::System, eq::ShallowWaterEquations1DStable)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        if output[index][1] < 0.0
            output[index] = typeof(output[index])(0.0, 0.0)
        end
    end
    return nothing
end