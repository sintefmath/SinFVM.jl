function reconstruct!(
    backend,
    linRec::LinearReconstruction,
    output_left,
    output_right,
    input_conserved,
    grid::Grid,
    eq::AllPracticalSWEStable,
    direction::Direction,
)
    if grid isa CartesianGrid
        @assert grid.ghostcells[1] > 1
    end

    h_input = input_conserved.h
    h_left = output_left.h
    h_right = output_right.h

    function fix_slope(slope, fix_val, ::ShallowWaterEquations1DStable)
        return typeof(slope)(fix_val, slope[2])
    end
    function fix_slope(slope, fix_val, ::ShallowWaterEquationsStable)
        return typeof(slope)(fix_val, slope[2], slope[3])
    end

    @fvmloop for_each_inner_cell(
        backend,
        grid,
        direction;
        ghostcells = 1,
    ) do ileft, imiddle, iright
        # 1) Obtain slope of (w, hu)
        slope =
            minmod_slope.(
                input_conserved[ileft],
                input_conserved[imiddle],
                input_conserved[iright],
                linRec.theta,
            )
        h = h_input[imiddle]

        # 2) Adjust slope of water
        if (h - 0.5 * slope[1] < 0.0)
            # Negative h on left face
            #TODO: uncomment and fix
            slope = fix_slope(slope, 2.0 * h, eq)
            #slope[1] = 2.0*(w_input[imiddle] - eq.B[imiddle])
        elseif (h + 0.5 * slope[1] < 0.0)
            # Negative h on right face
            #TODO:uncomment and fix
            slope = fix_slope(slope, 2.0 * h, eq)
        end

        # 3) Reconstruct face values (h, hu)
        output_left[imiddle] = input_conserved[imiddle] .- 0.5 .* slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* slope
    end
    nothing
end

