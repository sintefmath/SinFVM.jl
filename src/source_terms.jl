function evaluate_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem)

    # {right, left}_buffer is (h, hu)
    # output and current_state is (w, hu)
    dx = compute_dx(cs.grid)
    
    output_hu = output.hu
    for_each_inner_cell(cs.grid) do ileft, imiddle, iright
        B_right = cs.equation.B[imiddle+1]
        B_left  = cs.equation.B[imiddle  ]
        h_right = cs.right_buffer[imiddle][1] #- B_right
        h_left  = cs.left_buffer[ imiddle][1] #- B_left

        output_hu[imiddle] +=-cs.equation.g*((B_right - B_left)/dx)*((h_right + h_left)/2.0)
        nothing
    end
    

end

