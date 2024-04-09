include("forwardeuler.jl")
include("rungekutta.jl")


function post_proc_substep!(output, ::System, ::Equation)
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations1D)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(output[index][1], 0.0)
            output[index] = typeof(output[index])(output[index][1], 0.0) 
        end
    end
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(output[index][1], 0.0, 0.0)
            output[index] = typeof(output[index])(output[index][1], 0.0, 0.0) 
        end
    end
    return nothing
end
