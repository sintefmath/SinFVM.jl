
struct NoReconstruction <: Reconstruction end
struct LinearReconstruction <: Reconstruction
    theta::Float64
    LinearReconstruction(theta=1.2) = new(theta)
end


function reconstruct!(backend, ::NoReconstruction, output_left, output_right, input_conserved, grid::Grid, equation::Equation, direction)
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

function minmod_slope(left, center, right, theta)
    forward_diff = right .- center
    backward_diff = center .- left
    central_diff = (forward_diff .+ backward_diff) ./ 2.0
    return minmod.(theta .* forward_diff, central_diff, theta .* backward_diff)

end
function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, direction::Direction)
    @assert grid.ghostcells[1] > 1

    # NOTE: dx cancel, as the slope depends on 1/dx and face values depend on dx*slope
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        slope = minmod_slope.(input_conserved[ileft], input_conserved[imiddle], input_conserved[iright], linRec.theta)
        output_left[imiddle] = input_conserved[imiddle] .- 0.5 .* slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* slope
    end
end
function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, ::Equation, direction::Direction)
    reconstruct!(backend, linRec, output_left, output_right, input_conserved, grid, direction)
end

function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, eq::AllPracticalSWE, direction::Direction)
    @assert grid.ghostcells[1] > 1

    w_input = input_conserved.h
    h_left = output_left.h
    h_right = output_right.h

    # input_conserved is (w, hu)
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        # 1) Obtain slope of (w, hu)
        slope = minmod_slope.(input_conserved[ileft], input_conserved[imiddle], input_conserved[iright], linRec.theta)

        # 2) Adjust slope of water
        if (w_input[imiddle] - 0.5 * slope[1] < B_face_left(eq.B, imiddle))
            # Negative h on left face
            #TODO: uncomment and fix
            slope = typeof(slope)(2.0 * (w_input[imiddle] - B_face_left(eq.B, imiddle)), slope[2])
            #slope[1] = 2.0*(w_input[imiddle] - eq.B[imiddle])
        elseif (w_input[imiddle] + 0.5 * slope[1] < B_face_right(eq.B, imiddle))
            # Negative h on right face
            #TODO:uncomment and fix
            slope = typeof(slope)(2.0 * (B_face_right(eq.B, imiddle) - w_input[imiddle]), slope[2])
            #slope[1] = 2.0*(eq.B[imiddle] - w_input[imiddle])
        end

        # 3) Reconstruct face values (w, hu)
        output_left[imiddle] = input_conserved[imiddle] .- 0.5 .* slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* slope

        # 4) Return face values (h, hu)
        h_left[imiddle] -= B_face_left(eq.B, imiddle)
        h_right[imiddle] -= B_face_right(eq.B, imiddle)
    end
    nothing
end

