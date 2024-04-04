include("forwardeuler.jl")
include("rungekutta.jl")


function do_substep!(output, ts::TimeStepper, system::System, ::Equation, current_state, dt, timestep_computer, substep_number)
    return do_substep!(output, ts, system, current_state, dt, timestep_computer, substep_number)
end


function do_substep!(output, ts::TimeStepper, system::System, eq::ShallowWaterEquations1D, current_state, dt, timestep_computer, substep_number)
    # TODO: Change type of eq to AllPracticalSWE? If so, change also the content of the if statement
    # TODO: Need to support accumulation of rain... Move this functionality to postproc_substep function?
    dt = do_substep!(output, ts, system, current_state, dt, timestep_computer, substep_number)
    @fvmloop for_each_cell(system.backend, system.grid) do index
        
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(b_in_cell, 0.0)
        end
    
    end
    return dt
end

