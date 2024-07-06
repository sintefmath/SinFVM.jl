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