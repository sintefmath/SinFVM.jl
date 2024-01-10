abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end

function reconstruct!(::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, ::XDIRT)
    for_each_inner_cell(grid, 1) do ileft, imiddle, iright
        output_left[imiddle] = input_conserved[imiddle]
        output_right[imiddle] = input_conserved[imiddle]
    end
end