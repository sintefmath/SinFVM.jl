


struct ConstantRain{R} <: SourceTermRain
    rain_rate::R
    ConstantRain(rain_rate) = new{typeof(rain_rate)}(rain_rate)
end
Adapt.@adapt_structure ConstantRain

compute_rain(rain::ConstantRain, t...) = rain.rain_rate/3600

struct TimeDependentRain{R, T} <: SourceTermRain
    rain_rates::R
    time::T
    function TimeDependentRain(rain_rates, time=[0.0])
        if size(rain_rates) != size(time)
            throw(DomainError("Dimensions of input parameters rain_rates and time does not match"))
        end
        return new{typeof(rain_rates), typeof(time)}(rain_rates, time)
    end
end
Adapt.@adapt_structure TimeDependentRain


function compute_rain(rain::TimeDependentRain, t, i...)
    index = 1
    while index < size(rain.time)[1]
        if t < rain.time[index + 1]
            break
        end  
        index += 1
    end
    return rain.rain_rates[index]/3600
end




function evaluate_source_term!(rain::SourceTermRain, output, current_state, cs::ConservedSystem, t)
    output_h = output.h
    @fvmloop for_each_cell(cs.backend, cs.grid) do index
        output_h[index] += compute_rain(rain, t, index)
    end
end

# function evaluate_directional_source_term!(::SourceTermBottom, output, current_state, cs::ConservedSystem, dir::Direction)

#     # {right, left}_buffer is (h, hu)
#     # output and current_state is (w, hu)
#     dx = compute_dx(cs.grid, dir)
#     output_momentum = (dir == XDIR) ? output.hu : output.hv
#     B = cs.equation.B 
#     g = cs.equation.g
#     h_right = cs.right_buffer.h
#     h_left  = cs.left_buffer.h
#     @fvmloop for_each_inner_cell(cs.backend, cs.grid, dir) do ileft, imiddle, iright
#         B_right = B_face_right( B, imiddle, dir)
#         B_left  = B_face_left(B, imiddle, dir)

#         output_momentum[imiddle] +=-g*((B_right - B_left)/dx)*((h_right[imiddle] + h_left[imiddle])/2.0)
#         nothing
#     end
# end

