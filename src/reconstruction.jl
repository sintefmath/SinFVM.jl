
struct NoReconstruction <: Reconstruction end
struct LinearReconstruction <: Reconstruction 
    theta::Float64
    LinearReconstruction(theta=1.2) = new(theta)
end


function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    @fvmloop for_each_cell(backend, grid) do middle
        output_left[middle] = input_conserved[middle]
        output_right[middle] = input_conserved[middle]
    end
end

function minmod(a, b, c)
    if (a > 0) && (b > 0) && (c > 0)
        return min(a, b, c)
    elseif a < 0 && b < 0 && c < 0
        return max(a, b, c)
    end
    return zero(a)
end

function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, direction::XDIRT)
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
function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, ::Equation, direction::XDIRT)
    reconstruct!(backend, linRec, output_left, output_right, input_conserved, grid, direction)
end

function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, eq::ShallowWaterEquations1D, direction::XDIRT)
    # TODO: Might need to put all these steps in one fvmloop for performance

    # input is (w, hu)
    # First, reconstruct (w, hu)
    reconstruct!(backend, linRec, output_left, output_right, input_conserved, grid, direction)

    # TODO: Second, avoid negative water depth
    
    h_left = output_left.h
    h_right = output_right.h
    # Third, map from (w, hu) to (h, hu), where h = w - B
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        h_left[imiddle] = h_left[imiddle] - eq.B[imiddle]
        h_right[imiddle] = h_right[imiddle] - eq.B[imiddle + 1]
    end
end

