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




function evaluate_directional_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem, dir::Direction)

    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid, dir)
    output_momentum = (dir == XDIR) ? output.hu : output.hv
    B = cs.equation.B 
    g = cs.equation.g
    h_right = cs.right_buffer.h
    h_left  = cs.left_buffer.h
    @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right( B, imiddle, dir)
        B_left  = B_face_left(B, imiddle, dir)

        output_momentum[imiddle] +=-g*((B_right - B_left)/dx)*((h_right[imiddle] + h_left[imiddle])/2.0)
        nothing
    end
end


function evaluate_directional_source_term!(
    ::SourceTermBottom, output, current_state,
    cs::ConservedSystem{<:Any,<:Any,<:Any,<:TwoLayerShallowWaterEquations1D},
    dir::Direction
)
    dx = SinFVM.compute_dx(cs.grid, dir)
    B  = cs.equation.B
    g  = cs.equation.g
    r  = cs.equation.ρ1 / cs.equation.ρ2

    h1R = cs.right_buffer.h1
    h1L = cs.left_buffer.h1
    out_q2 = output.q2 

    # Use w if buffers have it; otherwise compute w = h2 + B_cell
    names = SinFVM.variable_names(typeof(cs.left_buffer))
    has_w = (:w in names)

    if has_w
        wR = cs.right_buffer.w
        wL = cs.left_buffer.w

        @fvmloop SinFVM.for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
            B_right = SinFVM.B_face_right(B, imiddle, dir)
            B_left  = SinFVM.B_face_left( B, imiddle, dir)
            Bx = (B_right - B_left)/dx

            avg = 0.5*(wR[imiddle] + wL[imiddle] + r*h1R[imiddle] + r*h1L[imiddle])
            out_q2[imiddle] += -g * avg * Bx
            nothing
        end
    else
        h2R = cs.right_buffer.h2
        h2L = cs.left_buffer.h2

        @fvmloop SinFVM.for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
            B_right = SinFVM.B_face_right(B, imiddle, dir)
            B_left  = SinFVM.B_face_left(B, imiddle, dir)
            Bmid    = SinFVM.B_cell(B, imiddle, dir)
            Bx = (B_right - B_left)/dx

            wR = h2R[imiddle] + Bmid
            wL = h2L[imiddle] + Bmid
            avg = 0.5*(wR + wL + r*h1R[imiddle] + r*h1L[imiddle])
            
            out_q2[imiddle] += -g * avg * Bx
            nothing
        end
    end

    return nothing
end