function evaluate_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem)

    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid)
    
    output_hu = output.hu
    B = cs.equation.B 
    g = cs.equation.g
    h_right = cs.right_buffer.h
    h_left  = cs.left_buffer.h
    @fvmloop for_each_inner_cell(cs.backend, cs.grid, XDIR) do ileft, imiddle, iright
        B_right = B_face_right( B, imiddle)
        B_left  = B_face_left(B, imiddle)

        output_hu[imiddle] +=-g*((B_right - B_left)/dx)*((h_right[imiddle] + h_left[imiddle])/2.0)
        nothing
    end
    

end

