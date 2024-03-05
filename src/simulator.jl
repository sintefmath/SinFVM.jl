import ProgressMeter

struct Simulator{BackendType, SystemType, TimeStepperType, GridType, StateType, FloatType}
    backend::BackendType
    system::SystemType
    timestepper::TimeStepperType
    grid::GridType

    substep_outputs::Vector{StateType}
    current_timestep::MVector{1,FloatType}
    cfl::FloatType
end

function Simulator(backend, system, timestepper, grid)
    return Simulator{
        typeof(backend),
        typeof(system),
        typeof(timestepper),
        typeof(grid),
        typeof(create_buffer(backend, grid, system)),
        Float64
    }(backend,
        system,
        timestepper,
        grid,
        [create_buffer(backend, grid, system) for _ in 1:number_of_substeps(timestepper)+1],
        MVector{1,Float64}([0]),
        0.5)
end

current_state(simulator::Simulator) = simulator.substep_outputs[1]

current_interior_state(simulator::Simulator) = current_state(simulator)[simulator.grid.ghostcells[1]+1:end-simulator.grid.ghostcells[1]]

function set_current_state!(simulator::Simulator, new_state)
    @assert length(simulator.grid.ghostcells) == 1

    gc = simulator.grid.ghostcells[1]
    current_state(simulator)[gc+1:end-gc] = new_state
    update_bc!(simulator.backend, simulator.grid, current_state(simulator))
end

current_timestep(simulator::Simulator) = simulator.current_timestep[1]
function compute_timestep(simulator::Simulator)
    # TODO: This can be done more efficiently by doing it in the computation of the flux

    wavespeed = compute_wavespeed(simulator.system, simulator.grid, current_state(simulator))
    return simulator.cfl * compute_dx(simulator.grid) / wavespeed
end

function perform_step!(simulator::Simulator)
    ##@info "Before step" simulator.substep_outputs
    simulator.current_timestep[1] = compute_timestep(simulator)
    for substep in 1:number_of_substeps(simulator.timestepper)
        @assert substep + 1 == 2
        do_substep!(simulator.substep_outputs[substep+1], simulator.timestepper, simulator.system, simulator.substep_outputs[substep], simulator.current_timestep[1])
        ##@info "before bc" simulator.substep_outputs[substep + 1]
        update_bc!(simulator.backend, simulator.grid, simulator.substep_outputs[substep+1])
        ##@info "after bc" simulator.substep_outputs[substep + 1]
    end
    ##@info "After step" simulator.substep_outputs[1] simulator.substep_outputs[2]
    simulator.substep_outputs[1], simulator.substep_outputs[end] = simulator.substep_outputs[end], simulator.substep_outputs[1]
end

function simulate_to_time(simulator::Simulator, endtime; t=0.0, callback=nothing)
    while t <= endtime
        perform_step!(simulator)
        t += current_timestep(simulator)

        if ! isnothing(callback)
            callback(t, simulator)
        end
    end
end