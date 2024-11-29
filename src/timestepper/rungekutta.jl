# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
