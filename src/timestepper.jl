


abstract type TimeStepper end

struct ForwardEulerStepper <: TimeStepper
end

number_of_substeps(::ForwardEulerStepper) = 1


function do_substep!(output, ::ForwardEulerStepper, system::System, current_state, dt, timestep_computer, substep_number)
    # Reset to zero
    # TODO: Remove allowscalar
    CUDA.@allowscalar output .= @SVector [zero(first(output))]#0.0#zero(output)

    wavespeed = add_time_derivative!(output, system, current_state)

    if substep_number == 1
        dt = timestep_computer(wavespeed)
    end
    output .*= dt
    output .+= current_state

    return dt
    ##@info "End of substep" output current_state
end
