import Meshes

function run_swe(
    grid::Meshes.CartesianGrid,
    initialvalue::ConservedVariables,
    bathymetry::Bathymetry,
    final_time::MyType,
    rain_function::Function, 
    callback::Function;
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
        v1 = maximum(abs.(curr_hu1_dev./curr_w1_dev))
        v2 = maximum(abs.(curr_hv1_dev./curr_w1_dev))
        v3 = maximum(abs.(curr_hv1_dev./curr_w1_dev .+ sqrt.(g.*curr_hv1_dev)))
        v4 = maximum(abs.(curr_hu1_dev./curr_w1_dev .+ sqrt.(g.*curr_hu1_dev)))
        v5 = maximum(abs.(curr_hv1_dev./curr_w1_dev .- sqrt.(g.*curr_hv1_dev)))
        v6 = maximum(abs.(curr_hu1_dev./curr_w1_dev .- sqrt.(g.*curr_hu1_dev)))
        cfl = max(v1, v2, v3, v4, v5, v6)

        if isinf(cfl)
            cfl = 1e-7
        end
        dt = 0.5 * min(dx, dy) / cfl

        dt = min(dt, 0.002)
    end
    return nothing
end
