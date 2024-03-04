abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end

function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        output_left[imiddle] = input_conserved[imiddle]
        output_right[imiddle] = input_conserved[imiddle]
        nothing
    end
end