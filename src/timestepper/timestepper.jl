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

include("forwardeuler.jl")
include("rungekutta.jl")


function post_proc_substep!(output, ::System, ::Equation)
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations1D)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(max(output[index][1], b_in_cell), 0.0)
            # output[index] = typeof(output[index])(output[index][1], 0.0) 
        end
    end
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(max(output[index][1], b_in_cell), 0.0, 0.0)
            # output[index] = typeof(output[index])(output[index][1], 0.0, 0.0) 
        end
    end
    return nothing
end

function implicit_substep!(output, previous_state, system, dt)
    return nothing
end
