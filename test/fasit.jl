# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

using Logging
using Plots
using ProgressMeter
struct Burgers end


(::Burgers)(u) = 0.5 * u^2

deriv(::Burgers, u) = u

function godunov(left, right, f, dt, dx)
    f_left = f(max(left, 0.0))
    f_right = f(min(right, 0.0))

    F = max(f_left, f_right)

    return F
end

function lax_friedrich(left, right, f, dt, dx)
    0.5 * (f(left) + f(right) - dx / dt * (right - left))
end

function neumann(u)
    u[end] = u[end-1]
    u[1] = u[2]
end

function periodic(u)
    u[end] = u[2]
    u[1] = u[end-1]
end


mutable struct DummyPrinter
    endtime::Float64
    message::String
    last_time::Float64
    printevery::Float64
    DummyPrinter(endtime, message; printevery=2) = new(endtime, message, time(), printevery)
end

function update!(m::DummyPrinter, state)
    return
    currenttime = time()
    if currenttime - m.last_time >= m.printevery
        print(stdout, "$state")
        flush(stdout)
        m.last_time = currenttime
    end
end

function solve_fvm(u0, T::Float64, number_of_x_cells, flux;
    start_x=0.0, end_x=1.0, numerical_flux=godunov,
    cfl_constant=0.5, bc=periodic, progress_printer=DummyPrinter)
    cfl(u) = deriv(flux, u)

    x = range(start_x, end_x, length=number_of_x_cells)
    @assert size(x, 1) == number_of_x_cells
    dx = x[2] - x[1]
    u = zeros(size(x, 1) + 2)
    u[2:end-1] = u0.(x .+ dx / 2)
    bc(u)
    dt = cfl_constant * dx / maximum(abs.(cfl.(u)))
    t = 0.0

    u_new = zero(u)



    progress = progress_printer(T, "Current time: ")
    total_timesteps_done = 0
    while t < T
        update!(progress, t)
        for i = 2:size(u, 1)-1
            time_derivative = -1 / dx * (numerical_flux(u[i], u[i+1], flux, dt, dx) - numerical_flux(u[i-1], u[i], flux, dt, dx))
            u_new[i] = u[i] + dt * time_derivative
        end

        u[2:end-1] .= u_new[2:end-1]
        bc(u)
        t += dt

        dt = cfl_constant * dx / maximum(abs.(cfl.(u)))

        total_timesteps_done += 1
    end
    println()

    return x, u[2:end-1], total_timesteps_done
end

function runme()
    T = 0.1
    number_of_x_cells = 64
    number_of_saves = 100
    u0 = x -> 1.0 * (x .< 0.5)
    u0 = x -> sin(2π * x)
    x, u, _, _ = solve_fvm(x -> sin(2π * x), T, number_of_x_cells, number_of_saves, Burgers())
    #x, u, _, _ = solve_fvm(u0, T, number_of_x_cells, number_of_saves, Burgers(), bc=periodic)
    f = Burgers()
    plot(x, u)
    #s = (f(1.0)-f(0.0))/(1.0-0.0)
    plot!(x, u0.(x))
end
