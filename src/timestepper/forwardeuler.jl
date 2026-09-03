# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



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

    # `dt` arrives in host precision (Float64); narrow it before it enters a kernel,
    # otherwise a Float32 state gets promoted to Float64 and Metal refuses to compile.
    # `convert_realtype` rather than a plain `paramtype(...)(dt)` because under AD `dt` is
    # a `ForwardDiff.Dual` (the timestep depends on the wave speeds, which are being
    # differentiated) and must be passed through untouched.
    dt_kernel = convert_realtype(paramtype(system.backend), dt)
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = current_state[index] + dt_kernel * output[index]
    end
    return dt
    ##@info "End of substep" output current_state
end
