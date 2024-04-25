struct RungeKutta2 <: TimeStepper
end

number_of_substeps(::RungeKutta2) = 2

function do_substep!(output, ::RungeKutta2, system::System, states, dt, timestep_computer, substep_number, t)
    # Reset to zero
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = zero(output[index])
    end
    
    wavespeed = add_time_derivative!(output, system, states[substep_number], t)

    if substep_number == 1
        dt = timestep_computer(wavespeed)
    end

    current_state = states[substep_number]
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = current_state[index] + dt * output[index]
    end
   
    
    if substep_number == 2
        first_state = states[1]
        @fvmloop for_each_cell(system.backend, system.grid) do index
            output[index] = 0.5 * (first_state[index] + output[index])
        end
    end

    return dt
end