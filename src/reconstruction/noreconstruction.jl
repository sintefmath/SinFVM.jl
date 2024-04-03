struct NoReconstruction <: Reconstruction end

function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction)
    @fvmloop for_each_cell(backend, grid) do middle
        output_left[middle] = input_conserved[middle]
        output_right[middle] = input_conserved[middle]
    end
end