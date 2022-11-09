
"""

"""
function run_swe{MyType}(
    grid,
    final_time,
    rain_function, 
    callback;
    friction_function = friction_fcg2016,
    infiltration_function = infiltration_horton_fcg,
    friction_constant = 0.03^2,
    topography = 1,
    theta::Float32 = 1.3,

    x0 = nothing,
    init_dambreak = false,
) where {MyType<:Real}

    flattenarr(x) = collect(Iterators.flatten(x))
    (Lx, Ly) = maximum(grid) - minimum(grid)
    (dx, dy) = spacing(grid)

    (Nx, Ny) = size(grid)

    dt = 0.02
    g = 9.81
    ngc = 2



    data_shape = (Nx + 2 * ngc, Ny + 2 * ngc)
    B = ones(MyType, data_shape)
    Bi = ones(MyType, data_shape .+ 1)
    w0 = zeros(MyType, data_shape)
    hu0 = zeros(MyType, data_shape)
    hv0 = zeros(MyType, data_shape)

    w1 = zeros(MyType, data_shape)
    hu1 = zeros(MyType, data_shape)
    hv1 = zeros(MyType, data_shape)

    infiltration_rates = zeros(MyType, data_shape)

    if topography == 1
        make_case_1_bathymetry!(B, Bi, dx)
    elseif topography == 2
        @assert(!isnothing(x0))
        make_case_2_bathymetry!(B, Bi, dx, x0)
    elseif topography == 3
        make_case_3_bathymetry!(B, Bi, dx)
    elseif topography == 4
        make_case_widebump_bathymetry!(B, Bi, dx)
    else
        @assert(false)
    end

    bathymetry_at_cell_centers!(B, Bi)

    w0[:, :] = B[:, :]




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

    number_of_timesteps = 350 * 60 * Integer(1 / dt) * 2

    save_every = 60 * Integer(1 / dt) * 2
    plot_every = 10 * 60 * Integer(1 / dt) * 2
    fig = nothing

    #number_of_timesteps = number_of_timesteps*100/350


    t = 0.0f0
    Q_infiltrated = zeros(0)
    save_t = zeros(0)
    runoff = zeros(0)

    num_saves = 1

    
    while t < T
        step::Int32 = (i + 1) % 2
        if i % 2 == 1
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

        if (i % 2 == 0)
            t = dt * (i / 2.0f0)
        end

        callback(
                    curr_w1_dev,
                    curr_hu1_dev,
                    curr_hv1_dev,
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
    end
    return nothing
end
