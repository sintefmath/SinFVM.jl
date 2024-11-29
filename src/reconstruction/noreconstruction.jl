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

struct NoReconstruction <: Reconstruction end

function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction)
    @fvmloop for_each_cell(backend, grid) do middle
        output_left[middle] = input_conserved[middle]
        output_right[middle] = input_conserved[middle]
    end
end

function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::AllPracticalSWE, direction)
    @fvmloop for_each_cell(backend, grid) do middle
        output_left[middle] = input_conserved[middle]
        output_right[middle] = input_conserved[middle]
    end

    # TODO: Combine this with the above

    h_input = input_conserved.h
    h_left = output_left.h
    h_right = output_right.h

    @fvmloop for_each_cell(backend, grid) do middle
        h_left[middle] = h_input[middle] - B_cell(equation.B, middle)
        h_right[middle] = h_input[middle] - B_cell(equation.B, middle)
    end
end
