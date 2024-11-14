using SinFVM
using Test
import CUDA
using Polynomials
using ProgressMeter
struct SimpleSystem <: SinFVM.System
    backend
    grid
    equation
end
SinFVM.create_volume(backend, grid, cs::SimpleSystem) = SinFVM.create_volume(backend, grid, cs.equation)

function SinFVM.add_time_derivative!(output, system::SimpleSystem, state, t)
    CUDA.@allowscalar output[2] += state[2]

    return [1.0]
end

function run_without_simulator(dt, steppertype, backend)
    stepper = steppertype()
    grid = SinFVM.CartesianGrid(1)
    system = SimpleSystem(backend, grid, nothing)
    buffers = [SinFVM.convert_to_backend(backend, ones(3)) for _ in 1:(SinFVM.number_of_substeps(stepper)+1)]

    compute_timestep(wavespeed) = dt

    T = 1.0
    t = 0.0

    while t < T
        for substep in 1:SinFVM.number_of_substeps(stepper)
            SinFVM.do_substep!(buffers[substep+1], stepper, system, buffers, dt, compute_timestep, substep, t)
        end
        buffers[end], buffers[1] = buffers[1], buffers[end]
        t += dt
    end

    return CUDA.@allowscalar buffers[1][2]
end

function run_with_simulator(dt, steppertype, backend)
    stepper = steppertype()
    grid = SinFVM.CartesianGrid(1, extent=[0 dt])
    equation = SinFVM.Burgers()
    system = SimpleSystem(backend, grid, equation)
    simulator = SinFVM.Simulator(backend, system, stepper, grid)
    
    T = 1.0
    t = 0.0

    SinFVM.set_current_state!(simulator, [1.0])
    SinFVM.simulate_to_time(simulator, T, show_progress=false)

    return collect(SinFVM.current_interior_state(simulator))[1]
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

for backend in get_available_backends()
    test_timestepper(SinFVM.ForwardEulerStepper, backend, 1.0, run_without_simulator)
    test_timestepper(SinFVM.RungeKutta2, backend, 2.0, run_without_simulator)
end

for backend in get_available_backends()
    test_timestepper(SinFVM.ForwardEulerStepper, backend, 1.0, run_with_simulator)
    test_timestepper(SinFVM.RungeKutta2, backend, 2.0, run_with_simulator)

end

