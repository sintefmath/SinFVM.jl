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

"""
    reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::TriangularGrid, equation, direction)

No-reconstruction specialisation for triangular grids.  Cell-averaged values
are copied into both `output_left` and `output_right`.
"""
function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::TriangularGrid, equation::Equation, direction::Direction)
    @fvmloop for_each_cell(backend, grid) do i
        output_left[i] = input_conserved[i]
        output_right[i] = input_conserved[i]
    end
end
