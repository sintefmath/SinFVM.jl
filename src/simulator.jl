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
    t::MVector{1,FloatType}
end

function Simulator(backend, system, timestepper, grid; cfl=0.25, t0=0.0)
    # TODO: Get cfl from reconstruction
    return Simulator{
        typeof(backend),
        typeof(system),
        typeof(timestepper),
        typeof(grid),
        typeof(create_volume(backend, grid, system)),
        backend.realtype,
    }(
        backend,
        system,
        timestepper,
        grid,
        [create_volume(backend, grid, system) for _ = 1:number_of_substeps(timestepper)+1],
        MVector{1,Float64}([0]),
        cfl,
        MVector{1,Float64}([t0]),
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
current_time(simulator::Simulator) = simulator.t[1]


function perform_step!(simulator::Simulator, max_dt)
    for substep = 1:number_of_substeps(simulator.timestepper)
        function timestep_computer(wavespeed)
            directional_dt = [compute_dx(simulator.grid, direction) / wavespeed[direction] for direction in directions(simulator.grid)]
            return min(simulator.cfl * minimum(directional_dt), max_dt)
        end
        simulator.current_timestep[1] = do_substep!(
            simulator.substep_outputs[substep+1],
            simulator.timestepper,
            simulator.system,
            simulator.substep_outputs,
            simulator.current_timestep[1],
            timestep_computer,
            substep,
            simulator.t[1]
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
    match_endtime=true,
    callback=nothing,
    show_progress=true,
    maximum_timestep=nothing,
)
    prog = ProgressMeter.Progress(100;
        enabled=show_progress,
        desc="Simulating",
        dt=2.0,
    )
    t = simulator.t
    while t[1] < endtime
        max_dt = match_endtime ? endtime - t[1] : Inf
        if !isnothing(maximum_timestep)
            max_dt = min(max_dt, maximum_timestep)
        end
        perform_step!(simulator, max_dt)
        t[1] += simulator.current_timestep[1]
        ProgressMeter.update!(prog, ceil(Int64, t[1] / endtime * 100),
            showvalues=[(:t, t[1]), (:dt, current_timestep(simulator))])
        if !isnothing(callback)
            callback(t[1], simulator)
        end
    end
end
