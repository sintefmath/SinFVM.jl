

struct ForwardEulerStepper <: TimeStepper
end

number_of_substeps(::ForwardEulerStepper) = 1

function do_substep!(output, ::ForwardEulerStepper, system::System, current_state, dt, timestep_computer, substep_number)
    # Reset to zero
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = zero(output[index])
    end

    wavespeed = add_time_derivative!(output, system, current_state)

    if substep_number == 1
        dt = timestep_computer(wavespeed)
    end

    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = current_state[index] + dt * output[index]
    end
    return dt
    ##@info "End of substep" output current_state
end
