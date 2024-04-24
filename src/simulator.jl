import ProgressMeter
interior(state) = InteriorVolume(state)
struct Simulator{BackendType,SystemType,TimeStepperType,GridType,StateType,FloatType}
    backend::BackendType
    system::SystemType
    timestepper::TimeStepperType
    grid::GridType

    substep_outputs::Vector{StateType}
    current_timestep::MVector{1,FloatType}
    cfl::FloatType
end

function Simulator(backend, system, timestepper, grid; cfl=0.25)
    # TODO: Get cfl from reconstruction
    return Simulator{
        typeof(backend),
        typeof(system),
        typeof(timestepper),
        typeof(grid),
        typeof(create_volume(backend, grid, system)),
        Float64,
    }(
        backend,
        system,
        timestepper,
        grid,
        [create_volume(backend, grid, system) for _ = 1:number_of_substeps(timestepper)+1],
        MVector{1,Float64}([0]),
        cfl,
    )
end

current_state(simulator::Simulator) = simulator.substep_outputs[1]

current_interior_state(simulator::Simulator) =
    interior(current_state(simulator))



function set_current_state!(simulator::Simulator, new_state)
    # TODO: Implement validation:
    # validate_state(new_state, simulator.grid) # Throw expection if dimensions of state don't match dimension of grid
    # validate_state!(new_state, simulator.equation) # Ensure that we don't initialize negative water depth
    # TODO: By adding the : operator to a normal volume in 2d, this should work with one line...
    if dimension(simulator.grid) == 1
        # TODO: Get it to work without allowscalar
        CUDA.@allowscalar current_interior_state(simulator)[:] = new_state
    elseif dimension(simulator.grid) == 2
        # TODO: Get it to work without allowscalar
        #CUDA.@allowscalar current_interior_state(simulator)[:, :] = new_state
        CUDA.@allowscalar current_interior_state(simulator)[1:end, 1:end] = convert_to_backend(simulator.backend, new_state)
    else
        error("Unandled dimension")
    end
    update_bc!(simulator, current_state(simulator))
end

function set_current_state!(simulator::Simulator, new_state::Volume)
    set_current_state!(simulator, InteriorVolume(new_state))
end

current_timestep(simulator::Simulator) = simulator.current_timestep[1]


function perform_step!(simulator::Simulator)
    for substep = 1:number_of_substeps(simulator.timestepper)
        function timestep_computer(wavespeed) 
            directional_dt = [compute_dx(simulator.grid, direction) / wavespeed[direction] for direction in directions(simulator.grid)]
            return simulator.cfl * minimum(directional_dt)
        end
        simulator.current_timestep[1] = do_substep!(
            simulator.substep_outputs[substep+1],
            simulator.timestepper,
            simulator.system,
            simulator.substep_outputs,
            simulator.current_timestep[1],
            timestep_computer,
            substep
        )

        implicit_substep!(simulator.substep_outputs[substep+1],
            simulator.substep_outputs[substep],
            simulator.system,
            simulator.current_timestep[1],
            )

        post_proc_substep!(
            simulator.substep_outputs[substep+1], 
            simulator.system, 
            simulator.system.equation
        )
        update_bc!(simulator, simulator.substep_outputs[substep+1])
    end
    simulator.substep_outputs[1], simulator.substep_outputs[end] =
        simulator.substep_outputs[end], simulator.substep_outputs[1]
end

function simulate_to_time(
    simulator::Simulator,
    endtime;
    t=0.0,
    callback=nothing,
    show_progress=true,
)
    prog = ProgressMeter.Progress(100;
        enabled=show_progress,
        desc="Simulating",
        dt=2.0,
    )
    while t < endtime
        perform_step!(simulator)
        t += current_timestep(simulator)
        ProgressMeter.update!(prog, ceil(Int64, t / endtime * 100),
            showvalues=[(:t, t), (:dt, current_timestep(simulator))])
        if !isnothing(callback)
            callback(t, simulator)
        end
    end
end
