


struct ForwardEulerStepper <: TimeStepper
end

number_of_substeps(::ForwardEulerStepper) = 1


function do_substep!(output, ::ForwardEulerStepper, system::System, current_state, dt)
    # Reset to zero
    # TODO: Remove allowscalar
    CUDA.@allowscalar output .= @SVector [zero(first(output))]#0.0#zero(output)

    add_time_derivative!(output, system, current_state)
    output .*= dt
    output .+= current_state
    ##@info "End of substep" output current_state
end
