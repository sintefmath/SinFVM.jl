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


struct SourceTermNonConservative <: SinFVM.SourceTerm end

function SinFVM.evaluate_directional_source_term!(::SourceTermNonConservative, output, current_state,
    cs::ConservedSystem, dir::Direction)

    delta = SinFVM.compute_dx(cs.grid, dir)
    eq = cs.equation
    g  = eq.g
    r  = eq.ρ1 / eq.ρ2

    out_m1 = (dir == XDIR) ? output.q1 : output.p1
    out_m2 = (dir == XDIR) ? output.q2 : output.p2

    names = SinFVM.variable_names(typeof(cs.left_buffer))
    has_w = (:w in names)

    B = eq.B

    @fvmloop SinFVM.for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
        h1R = cs.right_buffer.h1[imiddle]
        h1L = cs.left_buffer.h1[imiddle]
        B_right = SinFVM.B_face_right(B, imiddle, dir)
        B_left  = SinFVM.B_face_left( B, imiddle, dir)

        if has_w
            wR = cs.right_buffer.w[imiddle]
            wL = cs.left_buffer.w[imiddle]
        else
            wR = cs.right_buffer.h2[imiddle] + B_right
            wL = cs.left_buffer.h2[imiddle]  + B_left
        end

        N = g * 0.5 * ((h1R + wR) + (h1L + wL)) * ((h1R - h1L) / delta)

        out_m1[imiddle] +=  N
        out_m2[imiddle] += -r * N
        nothing
    end
    return nothing
end
