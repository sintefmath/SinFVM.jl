abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end

function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    output_left .= input_conserved
    output_right .= input_conserved
end