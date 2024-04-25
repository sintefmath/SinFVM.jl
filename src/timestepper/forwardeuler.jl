

struct ForwardEulerStepper <: TimeStepper
end

number_of_substeps(::ForwardEulerStepper) = 1

function do_substep!(output, ::ForwardEulerStepper, system::System, states, dt, timestep_computer, substep_number, t)
    # Reset to zero

    current_state = states[substep_number]
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = zero(output[index])
    end

    wavespeed = add_time_derivative!(output, system, current_state, t)

    if substep_number == 1
        dt = timestep_computer(wavespeed)
    end

    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = current_state[index] + dt * output[index]
    end
    return dt
    ##@info "End of substep" output current_state
end
