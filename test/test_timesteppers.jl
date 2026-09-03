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

using VolumeFluxes
isdefined(Main, :test_backends) || include("testing_utils.jl")
using Test
import CUDA
using Polynomials
using ProgressMeter
struct SimpleSystem <: VolumeFluxes.System
    backend
    grid
    equation
end
VolumeFluxes.create_volume(backend, grid, cs::SimpleSystem) = VolumeFluxes.create_volume(backend, grid, cs.equation)

function VolumeFluxes.add_time_derivative!(output, system::SimpleSystem, state, t)
    CUDA.@allowscalar output[2] += state[2]

    return [1.0]
end

function run_without_simulator(dt, steppertype, backend)
    stepper = steppertype()
    grid = VolumeFluxes.CartesianGrid(1)
    system = SimpleSystem(backend, grid, nothing)
    buffers = [VolumeFluxes.convert_to_backend(backend, ones(3)) for _ in 1:(VolumeFluxes.number_of_substeps(stepper)+1)]

    compute_timestep(wavespeed) = dt

    T = 1.0
    t = 0.0

    while t < T
        for substep in 1:VolumeFluxes.number_of_substeps(stepper)
            VolumeFluxes.do_substep!(buffers[substep+1], stepper, system, buffers, dt, compute_timestep, substep, t)
        end
        buffers[end], buffers[1] = buffers[1], buffers[end]
        t += dt
    end

    return CUDA.@allowscalar buffers[1][2]
end

function run_with_simulator(dt, steppertype, backend)
    stepper = steppertype()
    grid = VolumeFluxes.CartesianGrid(1, extent=[0 dt])
    equation = VolumeFluxes.Burgers()
    system = SimpleSystem(backend, grid, equation)
    simulator = VolumeFluxes.Simulator(backend, system, stepper, grid)
    
    T = 1.0
    t = 0.0

    VolumeFluxes.set_current_state!(simulator, [1.0])
    VolumeFluxes.simulate_to_time(simulator, T, show_progress=false)

    return collect(VolumeFluxes.current_interior_state(simulator))[1]
end
function test_timestepper(steppertype, backend, order, runfunction)
    dts = 1.0 ./( 2.0 .^ (6:15))
    errors = Float64[]

    @showprogress for dt in dts
        approx = runfunction(dt, steppertype, backend)
        err = abs(approx - exp(1.0))
        push!(errors, err)
    end


    poly = fit(log.(dts), log.(errors), 1)
    @test poly[1] ≈ order atol=1e-2
end

# NOTE: Float64 backends only, deliberately.
#
# This measures an *observed order of convergence* by fitting a line through the errors
# at dt down to 2^-15, i.e. up to 32768 sequential steps on a one-cell grid. Two reasons
# it does not belong on a Float32 or GPU backend:
#
#  * In Float32 the round-off accumulated over 32768 steps (~2e-5) is the same size as
#    the discretization error being measured (dt ~ 3e-5), so the fitted slope is
#    meaningless -- loosening the tolerance would not make the test correct, just quiet.
#  * On a GPU it times kernel launch overhead on a single cell, tens of thousands of
#    times over, which takes minutes and tests nothing about the device.
timestepper_backends() = filter(is_float64_backend, test_backends())

@testset "$(backend_label(backend))" for backend in timestepper_backends()
    test_timestepper(VolumeFluxes.ForwardEulerStepper, backend, 1.0, run_without_simulator)
    test_timestepper(VolumeFluxes.RungeKutta2, backend, 2.0, run_without_simulator)
end

@testset "$(backend_label(backend))" for backend in timestepper_backends()
    test_timestepper(VolumeFluxes.ForwardEulerStepper, backend, 1.0, run_with_simulator)
    test_timestepper(VolumeFluxes.RungeKutta2, backend, 2.0, run_with_simulator)

end
