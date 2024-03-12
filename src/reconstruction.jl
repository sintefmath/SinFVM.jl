abstract type Reconstruction end

struct NoReconstruction <: Reconstruction end
struct LinearReconstruction <: Reconstruction 
    theta::Float64
end


function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    output_left .= input_conserved
    output_right .= input_conserved
end

function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    @assert grid.ghostcells[1] > 1

    Δx = compute_dx(grid, direction)
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        forward_diff  = (input_conserved[iright]  .- input_conserved[imiddle])./Δx
        backward_diff = (input_conserved[imiddle] .- input_conserved[ileft])./Δx
        central_diff  = (forward_diff .+ backward_diff)./2.0
        slope = minmod.(linRec.theta.*forward_diff, central_diff, linRec.theta.*backward_diff)

        output_left[imiddle]  = input_conserved[imiddle] .- 0.5.*Δx.*slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5.*Δx.*slope
    end
end


function minmod(a, b, c)
    # TODO: avoid if statement
    if (a > 0) && (b > 0) && (c > 0)
        return min(a, b, c)
    elseif a < 0 && b < 0 && c < 0
        return max(a, b, c)
    end
    return 0.0
end