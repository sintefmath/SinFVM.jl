import Meshes
"""   
    run_swe(
        grid::Meshes.CartesianGrid,
        initialvalue::ConservedVariables,
        bathymetry::Bathymetry,
        final_time::MyType,
        rain_function::Function, 
        callback;
        friction_function = friction_fcg2016,
        infiltration_function = infiltration_horton_fcg,
        friction_constant = 0.03^2,
        theta::MyType = 1.3,
    )


...
# Arguments
- `grid::CartesianGrid{2, Float64}`: A `CartesianGrid` (from the `Meshes` package) describing the simulation domain
- `initialvalue::ConservedVariables`: Initial value of the conserved variables. Should have the same dimension as `grid`
- final_time: The simulation will run up and to `final_time`.
- rain_function: The rain function should be callable with parameters `(x, y, t)`
- callback: Use this to generate plots per timestep. A callable function taking as parameters `(h, hu, hv, infiltration_rates, B, dx, dy, dt, Nx, Ny, t, Q_infiltrated, runoff)` as parameters. This callback will be called for every timestep. It is up the user to filter out relevant timesteps.
- friction_function::Callable = friction_fcg2016: A callable function with parameters `(x, y, t)`
- infiltration_function::Callable = infiltration_horton_fcg: A callable function with parameters `(x, y, t)`
- friction_constant::Callable = 0.03^2: The friction constant to use together with the friction function
- theta::Float64 = 1.3: is the difference between the soil porosity and the initial volumetric water content
"""
function run_swe(
    grid::Meshes.CartesianGrid,
    initialvalue::ConservedVariables,
    bathymetry::Bathymetry,
    final_time::MyType,
    rain_function::Function, 
    callback;
    friction_function = friction_fcg2016,
    infiltration_function = infiltration_horton_fcg,
    friction_constant = 0.03^2,
    theta::MyType = 1.3,
) where {MyType <: Real}
    flattenarr(x) = collect(Iterators.flatten(x))
    (dx, dy) = Meshes.spacing(grid)

    (Nx, Ny) = size(grid)

    dt = 0.02
    g = 9.81

    data_shape = size(initialvalue)
    B = ones(MyType, data_shape)
    Bi = ones(MyType, data_shape .+ 1)
    w0 = zeros(MyType, data_shape)
    hu0 = zeros(MyType, data_shape)
    hv0 = zeros(MyType, data_shape)

    w1 = zeros(MyType, data_shape)
    hu1 = zeros(MyType, data_shape)
    hv1 = zeros(MyType, data_shape)

    infiltration_rates = zeros(MyType, data_shape)

    B .= bathymetry.B
    Bi .= bathymetry.Bi

    hu0 .= initialvalue.hu
    hv0 .= initialvalue.hv

    

    bathymetry_at_cell_centers!(B, Bi)

    w0[:, :] .= B[:, :] .+ initialvalue.h




    bc::Int32 = 1

    num_threads = num_blocks = nothing

    w0_dev = hu0_dev = hv0_dev = w1_dev = hu1_dev = hv1_dev = nothing
    B_dev = Bi_dev = nothing

    num_threads = (BLOCK_WIDTH, BLOCK_HEIGHT)
    num_blocks = Tuple(cld.([Nx, Ny], num_threads))
    w0_dev = CuArray(w0)

    hu0_dev = CuArray(hu0)
    hv0_dev = CuArray(hv0)

    w1_dev = CuArray(w1)
    hu1_dev = CuArray(hu1)
    hv1_dev = CuArray(hv1)

    Bi_dev = CuArray(Bi)
    B_dev = CuArray(B)
    infiltration_rates_dev = CuArray(infiltration_rates)


    t = 0.0f0
    Q_infiltrated = zeros(0)
  
    runoff = zeros(0)
    callback(
                    w0_dev,
                    hu0_dev,
                    hv0_dev,
                    infiltration_rates_dev,
                    B,
                    dx,
                    dy,
                    dt,
                    Nx,
                    Ny,
                    t,
                    Q_infiltrated,
                    runoff,
                )
    
    
    while t < final_time
        curr_w0_dev = nothing
        curr_hu0_dev = nothing
        curr_hv0_dev = nothing
        curr_w1_dev = nothing
        curr_hu1_dev = nothing
        curr_hv1_dev = nothing
        for step::Int32 in Int32(0):Int32(1)
            if step == 0
                curr_w0_dev = w0_dev
                curr_hu0_dev = hu0_dev
                curr_hv0_dev = hv0_dev
                curr_w1_dev = w1_dev
                curr_hu1_dev = hu1_dev
                curr_hv1_dev = hv1_dev
            else
                curr_w0_dev = w1_dev
                curr_hu0_dev = hu1_dev
                curr_hv0_dev = hv1_dev
                curr_w1_dev = w0_dev
                curr_hu1_dev = hu0_dev
                curr_hv1_dev = hv0_dev
            end


            call_kp07!(
                num_threads,
                num_blocks,
                Nx,
                Ny,
                dx,
                dy,
                dt,
                t,
                g,
                theta,
                step,
                curr_w0_dev,
                curr_hu0_dev,
                curr_hv0_dev,
                curr_w1_dev,
                curr_hu1_dev,
                curr_hv1_dev,
                Bi_dev,
                B_dev,
                bc;
                friction_handle = friction_function,
                friction_constant = friction_constant,
                rain_handle = rain_function,
                infiltration_handle = infiltration_function,
                infiltration_rates_dev = infiltration_rates_dev,
            )
        end

        t += dt

        callback(
                    curr_w1_dev,
                    curr_hu1_dev,
                    curr_hv1_dev,
                    infiltration_rates_dev,
                    B,
                    dx,
                    dy,
                    dt,
                    Nx,
                    Ny,
                    t,
                    Q_infiltrated,
                    runoff,
                )
        h = curr_w1_dev .- B_dev
        hp = h#clamp.(curr_w1_dev .- B_dev, 0.0, Inf)
        udev = curr_hu1_dev./h
        vdev = curr_hv1_dev./h
        v1 = maximum(abs.(udev))
        v2 = maximum(abs.(vdev))
        v3 = maximum(abs.(vdev .+ sqrt.(g.*hp)))
        v4 = maximum(abs.(udev .+ sqrt.(g.*hp)))
        v5 = maximum(abs.(vdev .- sqrt.(g.*hp)))
        v6 = maximum(abs.(udev .- sqrt.(g.*hp)))
        cfl = max(v1, v2, v3, v4, v5, v6)

        #@show cfl v1 v2 v3 v4 v5 v6
        if isinf(cfl)
            cfl = 1e-7
        end
        
        dt = 0.1 * min(dx, dy) / cfl

        # dt = min(dt, 0.002)
        dt = min(dt, min(dx, dy))
    end
    return nothing
end
