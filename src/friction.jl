abstract type AbstractFriction end

struct ImplicitFriction{Real, FrictionType} <: AbstractFriction # TODO: Better name?
    Cz::Real
    friction_function::FrictionType
    ImplicitFriction(;Cz= 0.03^2, friction_function=friction_bh2021) = new{typeof(Cz), typeof(friction_function)}(Cz, friction_function) # TODO: Correct default value?
end


function friction_bsa2012(c, h, speed)
    denom = cbrt(h)*h
    return -c*speed/denom
end

function friction_fcg2016(c, h, speed)
    denom = cbrt(h)*h*h
    return -c*speed/denom
end

function friction_bh2021(c, h, speed)
    denom = h*h
    return -c*speed/denom
end

function implicit_friction(friction::ImplicitFriction, equation::AllSWE1D, state, output, Bm, dt)
    h_star = desingularize(equation, state[1] - Bm)
    u = state[2] / h_star
    speed = sqrt(u^2)
    friction_factor = friction.friction_function(friction.Cz, h_star, speed)
    return output/(1 - dt * friction_factor)
end

function implicit_friction(friction::ImplicitFriction, equation::AllSWE2D, state, output, Bm, dt)
    h_star = desingularize(equation, state[1] - Bm)
    u = state[2] / h_star
    v = state[3] / h_star
    speed = sqrt(u^2 + v^2)
    friction_factor = friction.friction_function(friction.Cz, h_star, speed)
    return output/(1 - dt * friction_factor)
end

function implicit_substep!(output, previous_state, system, backend, friction::ImplicitFriction, equation::AllSWE, dt)
    @fvmloop for_each_cell(backend, system.grid) do index
        output[index] = implicit_friction(friction, equation, previous_state[index], output[index], B_cell(equation, index), dt)
    end
end