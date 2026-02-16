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


function evaluate_directional_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem{<:Any,<:Any,<:Any,<:AllTwoLayerSWE}, dir::Direction)
    println("DEBUG: Two-layer bottom source called, dir=", dir)
    dx = compute_dx(cs.grid, dir)
    B  = cs.equation.B
    g  = cs.equation.g
    r  = cs.equation.ρ1 / cs.equation.ρ2
    out_m2 = (dir == XDIR) ? output.q2 : output.p2
    h1R = cs.right_buffer.h1; h1L = cs.left_buffer.h1
    wR  = cs.right_buffer.w;  wL  = cs.left_buffer.w
    @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        B_right = B_face_right(B, imiddle, dir)
        B_left  = B_face_left( B, imiddle, dir)
        Bx = (B_right - B_left) / dx

        avg = 0.5 * (wR[imiddle] + wL[imiddle] + r*h1R[imiddle] + r*h1L[imiddle])
        out_m2[imiddle] += -g * avg * Bx
        nothing
    end
    return nothing
end