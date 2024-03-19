
struct NoReconstruction <: Reconstruction end
struct LinearReconstruction <: Reconstruction 
    theta::Float64
    # LinearReconstruction() = new(1.2)
end


function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction::XDIRT)
    output_left .= input_conserved
    output_right .= input_conserved
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

    # Third, map from (w, hu) to (h, hu), where h = w - B
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        # TODO: How do we easily update values within output_left and right?
        #       This does not feel correct... See also update_bc!(::WallBC) 
        output_left[imiddle]  = typeof(output_left[imiddle])(output_left[imiddle][1]   - eq.B[imiddle    ], output_left[imiddle][2] )
        output_right[imiddle] = typeof(output_right[imiddle])(output_right[imiddle][1] - eq.B[imiddle + 1], output_right[imiddle][2])
        # output_right[imiddle][1] = output_right[imiddle][1] - eq.B[imiddle +1 ]
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