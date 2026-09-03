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
    # `dt` arrives in host precision; narrow it before it enters a kernel, otherwise a
    # Float32 state gets promoted back to Float64. `convert_realtype` rather than a plain
    # `paramtype(...)(dt)` because under AD `dt` is a `ForwardDiff.Dual` -- the timestep
    # depends on the wave speeds, which are being differentiated -- and must pass through.
    dt_kernel = convert_realtype(paramtype(system.backend), dt)
    @fvmloop for_each_cell(system.backend, system.grid) do index
        output[index] = current_state[index] + dt_kernel * output[index]
    end
   
    
    if substep_number == 2
        first_state = states[1]
        @fvmloop for_each_cell(system.backend, system.grid) do index
            output[index] = 0.5 * (first_state[index] + output[index])
        end
    end

    return dt
end
