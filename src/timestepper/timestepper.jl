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

include("forwardeuler.jl")
include("rungekutta.jl")


function post_proc_substep!(output, ::System, ::Equation)
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations1D)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(max(output[index][1], b_in_cell), 0)
            # output[index] = typeof(output[index])(output[index][1], 0.0) 
        end
    end
    return nothing
end

function post_proc_substep!(output, system::System, eq::ShallowWaterEquations)
 
    @fvmloop for_each_cell(system.backend, system.grid) do index       
        b_in_cell = B_cell(eq.B, index)
        if output[index][1] - b_in_cell < eq.depth_cutoff
            output[index] = typeof(output[index])(max(output[index][1], b_in_cell), 0, 0)
            # output[index] = typeof(output[index])(output[index][1], 0.0, 0.0) 
        end
    end
    return nothing
end

function implicit_substep!(output, previous_state, system, dt)
    return nothing
end
