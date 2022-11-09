using Test, Plots
using ProgressMeter
###
# This file contains functions that defines topographies, rain functions, infiltration functions, 
# and some specific analysis functions that helps us reproduce results from 
#  Fernandez-Pato, Caviedes-Voullieme, Garcia-Navarro (2016), "Rainfall/runoff simulation with 
# 2D full shallow water equations: Sensitivity analysis and calibration of infiltration parameters".
# Journal of Hydrology, 536, 496-513 (https://doi.org/10.1016/j.jhydrol.2016.03.021).

# This file contains the following sets of functions:
# - Topographies, functions that create the topographies used in the validations cases
# - Rain source terms, functions that define rain
# - Infiltration source terms, functions that define infiltration
# - Misc utilities specific to the validation cases, but which also might have value in other contexts.


#############################################
# Topographies
#############################################
function _B_case_1(x)
    if x < 2000
        return 10 - (10.0/2000.0)*abs(x)
    end
    if x > 4000
        x = 4000 - (x - 4000)
        #return -(20/2000.0)*(x - 2000)
        return -(10/2000.0)*(x - 2000)
    end
    #return -(20/2000.0)*(x-2000)
    return -(10/2000.0)*(x-2000)
end

function make_case_1_bathymetry!(B, Bi, dx)
    for i = 1:size(B,1)
        x = (i-0.5-2)*dx
        B[i, :] .= _B_case_1(x)
    end
    for i = 1:size(Bi,1)
        xi = (i-1-2)*dx
        Bi[i,:] .= _B_case_1(xi)
    end
end

function _B_case_2(x, x0)
    B = _B_case_1(x)
    if x >= x0 && x <= x0 + 5
        B = B - (x - x0)
    elseif x >= x0 + 5 && x <= x0 + 15
        B = B - 5
    elseif x >= x0 + 15 && x <= x0 + 20
        B = B - (x0 + 20 - x)
    end
    return B
end

function make_case_2_bathymetry!(B, Bi, dx, x0)
    for i = 1:size(B,1)
        x = (i-0.5-2)*dx
        B[i, :] .= _B_case_2(x, x0)
    end
    for i = 1:size(Bi,1)
        xi = (i-1-2)*dx
        Bi[i,:] .= _B_case_2(xi, x0)
    end
end

function _B_case_3(x)
    x0 = 200
    if x <= x0
        return 21.0 - 1.0*sin(Float32(π)*abs(x)/10.0) - 0.005*abs(x) - 20.0
    else
        b0 = 21.0 - 1.0*sin(Float32(π)*x0/10.0) - 0.005*x0
        if x > x0*2
            x = x0*2 - (x - x0*2)
        end
        return b0 - 0.02*(x - x0) - 20.0
    end
end

function _B_case_widebump(x)
    x0 = 200
    if x <= x0
        return 21.0 - 1.0*sin(Float32(π)*abs(x)/100.0) - 0.005*abs(x) - 20.0
    else
        b0 = 21.0 - 1.0*sin(Float32(π)*x0/100.0) - 0.005*x0
        if x > x0*2
            x = x0*2 - (x - x0*2)
        end
        return b0 - 0.02*(x - x0) - 20.0
    end
end

function make_case_3_bathymetry!(B, Bi, dx)
    for i = 1:size(B,1)
        x = (i-0.5-2)*dx
        B[i, :] .= _B_case_3(x)
    end
    for i = 1:size(Bi,1)
        xi = (i-1-2)*dx
        Bi[i,:] .= _B_case_3(xi)
    end
end

function make_case_widebump_bathymetry!(B, Bi, dx)
    for i = 1:size(B,1)
        x = (i-0.5-2)*dx
        B[i, :] .= _B_case_widebump(x)
    end
    for i = 1:size(Bi,1)
        xi = (i-1-2)*dx
        Bi[i,:] .= _B_case_widebump(xi)
    end
end


function make_init_w_dummy_case_1!(w, dx)
    for i = 3:size(w,1)-2
        x = (i-0.5-2)*dx
        if x < 1000
            w[i, :] .= _B_case_1(x) + 3.75
        else
            w[i, :] .= _B_case_1(x)
        end    
    end
end

#############################################
# Rain source terms
#############################################
@inline @make_numeric_literals_32bits function 
    zero_rain(x, y, t)
    return 0.0
end

lots_of_rain(x, y, t) = 100.0

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_1(x, y, t)
    
    if x < 2000 && t < 250.0*60.0
        return 0.000125
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_2(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.00025
        elseif t < 2*rain_step
            return 0.000125
        elseif t < 3*rain_step
            return 7.4755e-5
        elseif t < 4*rain_step
            return 4.9020e-5
        elseif t < 5*rain_step
            return 7.4755e-5
        elseif t < 6*rain_step
            return 0.0001765
        end
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_3(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.0001867
        elseif t < 2*rain_step
            return 9.3341e-5
        elseif t < 3*rain_step
            return 5.6496e-5
        elseif t < 4*rain_step
            return 3.6845e-5
        elseif t < 5*rain_step
            return 5.6496e-5
        elseif t < 6*rain_step
            return 0.0001326
        end
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_4(x, y, t)
    
    if x < 2000 && t < 50.0*60.0
        #return 0.000291667
        return 0.000125
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_5(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 2000 
        if t < rain_step
            return 0.0001873
        elseif t < 2*
            rain_step
            return 3.6963e-5
        elseif t < 3*rain_step
            return 1.7249e-5
        elseif t < 4*rain_step
            return 0.000000
        elseif t < 5*rain_step
            return 5.5444e-5
        elseif t < 6*rain_step
            return 0.0001331
        end
    end
    return 0.0
end


@inline @make_numeric_literals_32bits function 
    rain_fcg_3(x, y, t)
    
    if x < 200 && t < 125.0*60.0
        # 0.25 mm/s
        return 0.00025
    end
    return 0.0
end



# Utility functions that are used to compute parameters in the 
# source term functions
function _get_rates_1_1()
    rain_volume = 75000
    duration = 250*60.0
    area = 2000*20
    rain_per_meter_per_second = rain_volume/(area * duration)
    return rain_per_meter_per_second
end

function _get_rates_1_x(rain_volume, ratios; duration=nothing)
    if isnothing(duration)
        duration = 250.0*60.0/6.0
    end
    area = 2000*20
    ratios = ratios/sum(ratios)
    
    rain_per_meter_per_second = rain_volume*ratios/(area * duration)
    return rain_per_meter_per_second
end

function _get_rates_1_2()
    rain_volume = 75000
    ratios = [5.1, 2.55, 1.525, 1, 1.525, 3.6]
    return _get_rates_1_x(rain_volume, ratios)
end

function _get_rates_1_3()
    rain_volume = 56250
    ratios = [3.8, 1.9, 1.15, 0.75, 1.15, 2.7]
    return _get_rates_1_x(rain_volume, ratios)
end

function _get_rates_1_4()
    rain_volume = 35000
    ratios = [1]
    return _get_rates_1_x(rain_volume, ratios, duration=50.0*60.0)
end

function _get_rates_1_5()
    rain_volume = 43000
    ratios = [3.8, 0.75, 0.35, 0.0, 1.125, 2.7]
    return _get_rates_1_x(rain_volume, ratios)
end

#############################################
# Infiltration source term
#############################################

@inline @make_numeric_literals_32bits function 
    infiltration_horton_fcg(x, y, t)
    if x < 2000
        fc = 3.272e-5
        f0 = 1.977e-4
        k  = 2.43e-3 
        return fc + (f0 - fc)*exp(-k*t)
    end
    return 0.0
end


#@inline @make_numeric_literals_32bits 
@inline function 
    infiltration_horton_fcg_3(x, y, t)
    if x < 200
        fc = 3.272e-5
        f0 = 1.977e-4
        k  = 2.43e-3 
        return fc + (f0 - fc)*exp(-k*t)
    end
    return 0.0
end



#############################################
### Misc utilities
#############################################



function get_runoff_h(subfolder, wfilename)
    B = npzread("runoff_validation/data/$(subfolder)/B.npy")
    w = npzread("runoff_validation/data/$(subfolder)/$(wfilename).npy")
    shape = size(w)
    h = w[3:shape[1]-2, 3:shape[2]-2] - B[3:shape[1]-2, 3:shape[2]-2]
    return h
end

function compute_runoff(subfolder, wfilename, dx, dy; runoff_start=nothing)
    h = get_runoff_h(subfolder, wfilename)    
    if isnothing(runoff_start)
        # Assuming in the middle of domain
        runoff_start = Integer(floor(size(h,1)/2)+1)
    end
    return sum(h[runoff_start:end, :])*dx*dy
end

function compute_remaining_water(subfolder, wfilename, dx, dy; runoff_start=nothing)
    h = get_runoff_h(subfolder, wfilename)   
    if isnothing(runoff_start)
        # Assuming in the middle of domain
        runoff_start = Integer(floor(size(h,1)/2)+1)
    end
    return sum(h[1:runoff_start-1, :])*dx*dy
end

function get_Q_infiltrated(infiltration_rates, dx, dy; runoff_start=nothing)
    shape = size(infiltration_rates)
    inf_rates = infiltration_rates[3:shape[1]-2, 3:shape[2]-2]   
    if isnothing(runoff_start)
        # Assuming in the middle of domain
        runoff_start = Integer(floor(size(inf_rates,1)/2)+1)
    end
    return sum(inf_rates[1:runoff_start-1, :])*dx*dy
end


function plot_hydrographs_at_location(subpath, rain_function, dx, dy)

    folder = "runoff_validation/data/$(subpath)"
    B = npzread("$(folder)/B.npy")
    t = npzread("$(folder)/t.npy")
    shape = size(B)
    B = B[3:shape[1]-2, 3:shape[2]-2]

    x_index = Integer(ceil(size(B, 1)/4)) # Middle of the relevant first half of domain
    x = dx*(x_index - 0.5) 
    y_index = Integer(ceil(size(B, 2)/2))
    y = dy*(y_index - 0.5) 
    println("Looking up index ($(x_index), $(y_index)) at position ($(x), $(y))")

    num_w_files = size(filter(x->contains(x, "w_"), readdir(folder)), 1)

    @assert(size(t,1) == num_w_files)

    infiltration = zeros(num_w_files)
    water_height = zeros(num_w_files)
    rain = zeros(num_w_files)

    if !isnothing(rain_function)
        rain = rain_function.(x, y, t)
    end

    for i in 1:num_w_files
        w = npzread("$(folder)/w_$(lpad(i-1, 3, "0")).npy")
        h = w[3:shape[1]-2, 3:shape[2]-2] - B
        f = npzread("$(folder)/infiltration_$(lpad(i-1, 3, "0")).npy")
        f = f[3:shape[1]-2, 3:shape[2]-2] 

        infiltration[i] = f[x_index, y_index]
        water_height[i] = h[x_index, y_index]
    end

    fig = Plots.plot(t/60.0, water_height, label="h at x=$(x) m", 
                    legend=:topleft, ylims=[0, 0.25], title=subpath, 
                    ylabel="h [m]", xlabel="t [minutes]", color=:blue,
                    right_margin=20Plots.mm)
    Plots.plot!(twinx(), t/60.0, [rain, infiltration], label=["rainfall rate r" "f at x = $(x) m"],
                 ylims=[0,0.00025], ylabel="f [m/s], r [m/s]")
    display(fig)
    return fig
end

function check_friction_terms(friction_function, friction_constant, folder::String, timestep::Integer)

    w = npzread("$(folder)/w_$(lpad(timestep, 3, "0")).npy")[3:end-2, 3:end-2];
    hu = npzread("$(folder)/hu_$(lpad(timestep, 3, "0")).npy")[3:end-2, 3:end-2];
    hv = npzread("$(folder)/hv_$(lpad(timestep, 3, "0")).npy")[3:end-2, 3:end-2];
    B = npzread("$(folder)/B.npy")[3:end-2, 3:end-2];
    
    return check_friction_terms(friction_function, friction_constant, w, B, hu, hv)
end


function check_friction_terms(friction_function, friction_constant, w, B, hu, hv)

    nx, ny = size(w)
    sf = zeros((nx, ny))
    for i = 1:nx
        for j = 1:ny
            h = w[i,j] - B[i,j]
            h_star = h
            if (h < KP_DESINGULARIZE_DEPTH)
                h_star = desingularize_depth(h)
            end
            u = hu[i,j]/h_star
            v = hv[i,j]/h_star
            sf[i,j] = friction_function(friction_constant, h_star, u, v)
        end
    end
    
    return sf
end


function total_mass(folder; trailing_zeros=0)
    B = npzread("$(folder)/B.npy")
    num_w_files = size(filter(x->contains(x, "w_"), readdir(folder)), 1)
    mass = zeros(num_w_files)

    for i in 1:num_w_files
        w = npzread("$(folder)/w_$(lpad(i-1, trailing_zeros, "0")).npy")
        mass[i] = sum(w-B)
    end
    return mass
end

function acummulated_infiltration(folder; trailing_zeros=0)
    num_w_files = size(filter(x->contains(x, "infiltration_"), readdir(folder)), 1)
    t = npzread("$(folder)/t.npy")
    @assert(size(ttmp,1) == num_w_files)
    f_tmp = npzread("$(folder)/infiltration_$(lpad(0, trailing_zeros, "0")).npy")
    f = f_tmp[3:end-2, 10]*t[1]
    for i in 2:num_w_files
        f_tmp = npzread("$(folder)/infiltration_$(lpad(i-1, trailing_zeros, "0")).npy")
        f += f_tmp[3:end-2, 10]*(t[i] - t[i-1])
    end
    return f
end


function run_validation_cases(subpath, rain_function; topography=1, x0 = nothing, init_dambreak=false)
    
    mkpath("runoff_validation/plots/$(subpath)/")
    mkpath("runoff_validation/data/$(subpath)/")

    flattenarr(x) = collect(Iterators.flatten(x))
    MyType = Float64
    Lx = 2000*2
    if topography > 2
        Lx = 200*2
    end
    Ly = 20
    dx = dy = 2.0
    if topography > 1
        dx = dy = 1.0
    end

    Nx = Int32(Lx/dx)
    Ny = Int32(Ly/dy)
    println("Setting up grid ($(Nx), $(Ny)) with (dx, dy) = ($(dx), $(dy)) for $(subpath)")
    
    dt = 0.02
    g = 9.81
    ngc = 2

    friction_function = friction_fcg2016
    friction_constant = 0.03^2 # As used by Fernandez-Paro et al (2016)
    
    #friction_function = friction_bsa2012
    #friction_constant = g* 0.033f0^2 # As used by Brodtkorb et al (2012)
    #friction_constant = 0.033f0^2 # As Brodtkorb et al (2012) but without g
    #friction_constant = 0.0033f0^2 # As Brodtkorb et al (2012) but without g and /100
    
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

    if init_dambreak
        if topography > 2 
            w0[1:200, :] .+= 0.1
        else
            w0[1:500, :] .+= 1
        end
    end

    npzwrite("runoff_validation/data/$(subpath)/B.npy", B)
    npzwrite("runoff_validation/data/$(subpath)/Bi.npy", Bi)
    
    infiltration_function = infiltration_horton_fcg
    if topography > 2
        infiltration_function = infiltration_horton_fcg_3
    end

    theta::Float32 = 1.3

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
    B_dev  = CuArray(B)
    infiltration_rates_dev = CuArray(infiltration_rates)


    #npzwrite("runoff_validation/data/eta_init.npy", eta0)

    number_of_timesteps = 350*60*Integer(1/dt) *2
    
    save_every = 60*Integer(1/dt)*2
    plot_every = 10*60*Integer(1/dt)*2
    fig = nothing

    #number_of_timesteps = number_of_timesteps*100/350
    

    t = 0.0f0
    Q_infiltrated =  zeros(0)
    save_t =  zeros(0)
    runoff =  zeros(0)
    print_and_plot_swe!(subpath, 0, w0_dev, hu0_dev, hv0_dev,
                        infiltration_rates_dev, B, dx, dy, Nx, Ny, t, data_shape, 
                        Q_infiltrated, save_t, runoff)
    num_saves = 1
                   

    @showprogress for i in 1:number_of_timesteps
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


        call_kp07!(num_threads, num_blocks,
            Nx, Ny, dx, dy, dt, t,
            g, theta, step,
            curr_w0_dev, curr_hu0_dev, curr_hv0_dev,
            curr_w1_dev, curr_hu1_dev, curr_hv1_dev,
            Bi_dev, B_dev,
            bc;
            friction_handle=friction_function, friction_constant=friction_constant,
            rain_handle = rain_function, infiltration_handle = infiltration_function,
            infiltration_rates_dev = infiltration_rates_dev)
    
        if (i % 2 == 0)
            t = dt*(i/2.0f0)
        end

        if step == 1
            if (save_every > 0 && i % save_every == 0) || (i == number_of_timesteps)

                doPlot = (i % plot_every == 0) || (i == number_of_timesteps)
                fig = print_and_plot_swe!(subpath, num_saves, 
                                          curr_w1_dev, curr_hu1_dev, curr_hv1_dev,
                                          infiltration_rates_dev, B, dx, dy, Nx, Ny, t, data_shape, 
                                          Q_infiltrated, save_t, runoff, doPlot=doPlot)
                num_saves += 1
            end
        end

        if number_of_timesteps < 3
            hu1_copied = reshape(collect(curr_hu1_dev), data_shape)
            fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                plot_title="hu i=$i, t=$(t) s")
            w1_copied = reshape(collect(curr_w1_dev), data_shape)
            fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
                plot_title="w i=$i")
        end        

    end

    


    npzwrite("runoff_validation/data/$(subpath)/t.npy", save_t)
    npzwrite("runoff_validation/data/$(subpath)/Q_infiltration.npy", Q_infiltrated)
    npzwrite("runoff_validation/data/$(subpath)/runoff.npy", runoff)
    #println(save_t)
    #println(Q_infiltrated)
    #println(runoff)

    cons_mass_fig = plot_conservation_of_mass(subpath)
    swim_save("runoff_validation/plots/$(subpath)/conservation_of_mass_fig.png", cons_mass_fig)

    fcg_fig = make_fcg_plot(subpath, rain_function, topography=topography)
    swim_save("runoff_validation/plots/$(subpath)/fcg_hyd_outlet_infiltration_fig.png", fcg_fig) 

    fcg_fig = make_fcg_plot(subpath, rain_function, topography=topography, with_infiltration=false)
    swim_save("runoff_validation/plots/$(subpath)/fcg_hyd_outlet_fig.png", fcg_fig) 

    hyd_fig = plot_hydrographs_at_location(subpath, rain_function, dx, dy)
    swim_save("runoff_validation/plots/$(subpath)/fcg_hyd_1000_fig.png", hyd_fig) 

    display(fig)    
    return nothing
end

function make_fcg_plot(subpath, rain_function; topography=1, with_infiltration=true)
    t = npzread("runoff_validation/data/$(subpath)/t.npy")
    Q_infiltrated = npzread("runoff_validation/data/$(subpath)/Q_infiltration.npy")
    runoff = npzread("runoff_validation/data/$(subpath)/runoff.npy")
    rain_Q = zeros(size(runoff))
    if !isnothing(rain_function)
        rain_Q = rain_function.(10,10, t)*2000*20
    end
    outlet_Q = zeros(size(runoff))
    num_t = size(t,1)
    outlet_Q[2:num_t] = (runoff[2:num_t] - runoff[1:num_t-1])./(t[2:num_t] - t[1:num_t-1])
    #print(runoff)


    yaxis2lim = 75000
    if rain_function == rain_fcg_1_3
        yaxis2lim = 50000
    elseif rain_function == rain_fcg_1_4
        yaxis2lim = 15000
    elseif rain_function == rain_fcg_1_5
        yaxis2lim = 30000
    end
    
    yaxis1lim = 10
    if topography == 3
        yaxis2lim = 5000
        yaxis1lim = 1.5
    end

    fig = Plots.plot(t/60.0, [rain_Q outlet_Q], label=["rain Q" "runoff Q"], 
                     legend=:topleft, ylims=[0, yaxis1lim], title=subpath, ylabel="Discharge Q [m^3/s]", xlabel="minutes",
                     right_margin=20Plots.mm)
    if with_infiltration
        Plots.plot!(t/60, Q_infiltrated, label="infiltration Q")
    end                     
    Plots.plot!(twinx(), t/60.0, runoff, color=:red, label="runoff", ylims=[0,yaxis2lim], ylabel="Runoff volume [m^3]")
    display(fig)
    return fig
end
    
function plot_conservation_of_mass(subpath)
    t = npzread("runoff_validation/data/$(subpath)/t.npy")
    mass = total_mass("runoff_validation/data/$(subpath)", trailing_zeros=3)
    fig = Plots.plot(t/60.0, mass, title="Conservation of mass - $(subpath)", ylabel="total mass", xlabel="time [minutes]")
    display(fig)
    return fig
end


function print_and_plot_swe!(subpath, index, w_dev, hu_dev, hv_dev, infiltration_dev, B, dx, dy, Nx, Ny, t, data_shape, 
                             Q_infiltrated, save_t, runoff; doPlot = true)
    #println("saving $(index) at t=$(t)")
    if (index isa Number)
        index = lpad(index, 3, "0")
    end
    w1_copied = reshape(collect(w_dev), data_shape)
    npzwrite("runoff_validation/data/$(subpath)/w_$(index).npy", w1_copied)
    hu1_copied = reshape(collect(hu_dev), data_shape)
    npzwrite("runoff_validation/data/$(subpath)/hu_$(index).npy", hu1_copied)
    hv1_copied = reshape(collect(hv_dev), data_shape)
    npzwrite("runoff_validation/data/$(subpath)/hv_$(index).npy", hv1_copied)
    infiltration_copied = reshape(collect(infiltration_dev), data_shape)
    npzwrite("runoff_validation/data/$(subpath)/infiltration_$(index).npy", infiltration_copied)
    
    append!(save_t, t)
    append!(Q_infiltrated, get_Q_infiltrated(infiltration_copied, dx, dy))
    append!(runoff, compute_runoff(subpath, "w_$(index)", dx, dy))
    
    fig = nothing
    if doPlot
        fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                    plot_title="hu t=$(t/60) min")
        swim_save("runoff_validation/plots/$(subpath)/hu_$(index).png", fig) 
        #fig = plotSurf(hv1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
        #            plot_title="hv t=$(t/60) min")
        #swim_save("runoff_validation/plots/$(subpath)/hv_$(index).png", fig) 
        fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
                    plot_title="w  t=$(t/60) min")
        swim_save("runoff_validation/plots/$(subpath)/w_$(index).png", fig) 
    end
    return fig
end
