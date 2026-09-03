# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




struct ConstantRain{R} <: SourceTermRain
    rain_rate::R
    ConstantRain(rain_rate) = new{typeof(rain_rate)}(rain_rate)
end
Adapt.@adapt_structure ConstantRain

compute_rain(rain::ConstantRain, t...) = rain.rain_rate/3600

struct TimeDependentRain{R, T} <: SourceTermRain
    rain_rates::R
    time::T
    # NOTE: `rain_rates` and `time` are indexed inside a kernel (see `compute_rain`
    # below), so they have to live on the device. Unlike `HortonInfiltration` and
    # `BottomTopography2D`, this constructor used to keep them as host arrays, which
    # silently produces invalid device pointers on any GPU backend. It went unnoticed
    # because the rain tests only ever construct a CPU backend.
    function TimeDependentRain(rain_rates, time=[0.0], backend=make_cpu_backend())
        if size(rain_rates) != size(time)
            throw(DomainError("Dimensions of input parameters rain_rates and time does not match"))
        end
        rain_rates = convert_to_backend(backend, rain_rates)
        time = convert_to_backend(backend, time)
        return new{typeof(rain_rates), typeof(time)}(rain_rates, time)
    end

    # Internal: the values are already on the right device. Keyword-*only* so it cannot
    # collide with the two-positional-argument form above -- keyword arguments do not take
    # part in dispatch, so `(rain_rates, time; kw)` would be the same method.
    TimeDependentRain(; rain_rates, time) =
        new{typeof(rain_rates), typeof(time)}(rain_rates, time)
end

function Adapt.adapt_structure(to, rain::TimeDependentRain)
    TimeDependentRain(; rain_rates=Adapt.adapt_structure(to, rain.rain_rates),
        time=Adapt.adapt_structure(to, rain.time))
end


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


struct FunctionalRain{T<:Function, G<:Grid}<: SourceTermRain
    rain_function::T
    grid::G
    FunctionalRain(rain_function, grid) = new{typeof(rain_function), typeof(grid)}(rain_function, grid)
end
Adapt.@adapt_structure FunctionalRain


function compute_rain(rain::FunctionalRain, t, index)
    x, y = cell_center(rain.grid, index)
    rain.rain_function(t, x, y)
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
