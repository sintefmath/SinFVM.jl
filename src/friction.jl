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

abstract type AbstractFriction end

struct ImplicitFriction{Real,FrictionType} <: AbstractFriction # TODO: Better name?
    Cz::Real
    friction_function::FrictionType
    ImplicitFriction(; Cz=0.03^2, friction_function=friction_bh2021) = new{typeof(Cz),typeof(friction_function)}(Cz, friction_function) # TODO: Correct default value?
end


function friction_bsa2012(c, h, speed)
    denom = cbrt(h) * h
    return -c * speed / denom
end

function friction_fcg2016(c, h, speed)
    denom = cbrt(h) * h * h
    return -c * speed / denom
end

function friction_bh2021(c, h, speed)
    denom = h * h
    return -c * speed / denom
end

function implicit_friction(friction::ImplicitFriction, equation::AllSWE1D, state, output, Bm, dt)
    h_star = desingularize(equation, state[1] - Bm)
    u = state[2] / h_star
    speed = sqrt(u^2)

    # This looks weird, but basically fixes the AD issue.
    # In essence, if the speed is zero, we don't want any friction,
    # so setting speed to zero will also make the derivative of speed zero.
    if speed == 0.0
        speed = 0.0
    end

    friction_factor = @SVector [0.0, friction.friction_function(friction.Cz, h_star, speed)]

    return output ./ (1 .- dt .* friction_factor)
end

function implicit_friction(friction::ImplicitFriction, equation::AllSWE2D, state, output, Bm, dt)
    h_star = desingularize(equation, state[1] - Bm)
    u = state[2] / h_star
    v = state[3] / h_star
    speed = sqrt(u^2 + v^2)

    # This looks weird, but basically fixes the AD issue.
    # In essence, if the speed is zero, we don't want any friction,
    # so setting speed to zero will also make the derivative of speed zero.
    if speed == 0.0
        speed = zero(speed)
    end
    friction_scalar = friction.friction_function(friction.Cz, h_star, speed)
    friction_factor = @SVector [0.0, friction_scalar, friction_scalar]

    return output ./ (1 .- dt .* friction_factor)
end

function implicit_substep!(output, previous_state, system, backend, friction::ImplicitFriction, equation::AllSWE, dt)
    dt_kernel = convert_realtype(paramtype(backend), dt)
    @fvmloop for_each_cell(backend, system.grid) do index
        output[index] = implicit_friction(friction, equation, previous_state[index], output[index], B_cell(equation, index), dt_kernel)
    end
end
