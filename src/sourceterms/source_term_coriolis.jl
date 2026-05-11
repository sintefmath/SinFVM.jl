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


struct SourceTermCoriolis{T} <: SourceTerm
    f::T   # Coriolis parameter 
end

# Single-layer 2D
function evaluate_source_term!(st::SourceTermCoriolis, output, current_state, cs::ConservedSystem, _)
    f = st.f
    @fvmloop for_each_inner_cell(cs.backend, cs.grid) do imiddle
        hu = current_state.hu[imiddle]
        hv = current_state.hv[imiddle]
        output.hu[imiddle] +=  f * hv
        output.hv[imiddle] += -f * hu
        nothing
    end
end

# Two-layer 2D: non-directional coupling maintains physical accuracy
function evaluate_source_term!(
    st::SourceTermCoriolis,
    output,
    current_state,
    cs::ConservedSystem{<:Any,<:Any,<:Any,<:TwoLayerShallowWaterEquations2D},
    _,
)
    f = st.f
    @fvmloop for_each_inner_cell(cs.backend, cs.grid) do imiddle
        q1 = current_state.q1[imiddle]
        q2 = current_state.q2[imiddle]
        p1 = current_state.p1[imiddle]
        p2 = current_state.p2[imiddle]
        output.q1[imiddle] +=  f * p1
        output.q2[imiddle] +=  f * p2
        output.p1[imiddle] += -f * q1
        output.p2[imiddle] += -f * q2
        nothing
    end
end