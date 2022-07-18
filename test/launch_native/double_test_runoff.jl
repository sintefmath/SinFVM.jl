using Test

using CUDA
using Plots
using ProgressMeter
using NPZ
using PyCall

#using .GPUOceanUtils
include("RunoffUtils.jl")
include("SWEPlottingNoMakie.jl")

include("SWEUtils.jl")
include("double_swe_kp07_pure.jl")

## This script aims to reproduce simulations from
# Fernandez-Pato, Caviedes-Voullieme, Garcia-Navarro (2016) 
# Rainfall/runoff simulation with 2D full shallow water equations: Sensitivity analysis and calibration of infiltration parameters.
# Journal of Hydrology, 536, 496-513. https://doi.org/10.1016/j.jhydrol.2016.03.021


function run_stuff(subpath, rain_function; topography=1, x0 = nothing, init_dambreak=false)
    flattenarr(x) = collect(Iterators.flatten(x))
    MyType = Float64
    Lx = 2000*2
    if topography == 3
        Lx = 200*2
    end
    Ly = 20
    dx = dy = 1.0

    Nx = Int32(Lx/dx)
    Ny = Int32(Ly/dy)
    println("Setting up grid ($(Nx), $(Ny)) with (dx, dy) = ($(dx), $(dy))")
    
    dt = 0.02
    g = 9.81
    ngc = 2

    friction_function = friction_fcg2016
    friction_constant = 0.03^2 # As used by Fernandez-Paro et al (2016)
    
    #friction_function = friction_bsa2012
    #friction_constant = g* 0.033f0^2 # As used by Brodtkorb et al (2012)
    #friction_constant = 0.033f0^2 # As Brodtkorb et al (2012) but without g
    #friction_constant = 0.0033f0^2 # As Brodtkorb et al (2012) but without g and /100
    
    #friction_constant = 0.01f0^2

    friction_constant = 0.0f0

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
        #make_init_w_dummy_case_1!(w0, dx)
    elseif topography == 2
        @assert(!isnothing(x0))
        make_case_2_bathymetry!(B, Bi, dx, x0)
    elseif topography == 3
        make_case_3_bathymetry!(B, Bi, dx)
    else
        @assert(false)
    end

    w0[:, :] = B[:, :]

    if init_dambreak
        w0[1:500, :] .+= 1
    end

    #println(size(H))
    #display(plotSurf(w0, B, dx, dy, Nx, Ny, show_ground=true, km=false, depth_cutoff=1e-4))
    #fig = plotSurf(eta0, H, dx, dy, Nx, Ny)
    #display(fig)    
    npzwrite("runoff/data/$(subpath)/B.npy", B)
    npzwrite("runoff/data/$(subpath)/Bi.npy", Bi)
    
    #return nothing

    #rain_function = rain_fcg_1_1
    infiltration_function = infiltration_horton_fcg
    if topography == 3
        infiltration_function = infiltration_horton_fcg_3
    end

    infiltration_function = nothing

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


    #npzwrite("runoff/data/eta_init.npy", eta0)

    #number_of_timesteps = 1
    #number_of_timesteps = Integer(100000*2)
    #number_of_timesteps = Integer(10000*2)
    number_of_timesteps = 350*60*Integer(1/dt)*2
    #umber_of_plots = 10
    #save_every = Integer(floor(number_of_timesteps/number_of_plots))
    save_every = 60*Integer(1/dt)*2
    plot_every = 10*60*Integer(1/dt)*2
    fig = nothing

    number_of_timesteps = number_of_timesteps*100/350
    

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
                #append!(save_t, t)

                #function print_and_plot_swe(subpath, index, w_dev, hu_dev, hv_dev, infiltration_dev, B, dx, dy, Nx, Ny)
                #println("saving $(div(i, save_every)) at t=$(t)")
                #w1_copied = reshape(collect(curr_w1_dev), data_shape)
                #npzwrite("runoff/data/$(subpath)/w_$(div(i, save_every)).npy", w1_copied)
                #hu1_copied = reshape(collect(curr_hu1_dev), data_shape)
                #npzwrite("runoff/data/$(subpath)/hu_$(div(i, save_every)).npy", hu1_copied)
                #hv1_copied = reshape(collect(curr_hv1_dev), data_shape)
                #npzwrite("runoff/data/$(subpath)/hv_$(div(i, save_every)).npy", hv1_copied)
                #infiltration_copied = reshape(collect(infiltration_rates_dev), data_shape)
                #npzwrite("runoff/data/$(subpath)/infiltration_$(div(i, save_every)).npy", infiltration_copied)
                #

                #fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                #             plot_title="hu i=$i, t=$(t/60) min")
                #swim_save("runoff/plots/$(subpath)/hu_$(div(i, save_every)).png", fig) 
                #fig = plotSurf(hv1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                #             plot_title="hv i=$i, t=$(t/60) min")
                #swim_save("runoff/plots/$(subpath)/hv_$(div(i, save_every)).png", fig) 
                #fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
                #             plot_title="w i=$i, t=$(t/60) min")
                #swim_save("runoff/plots/$(subpath)/w_$(div(i, save_every)).png", fig) 
                #lowest_w[(div(i, save_every))+1] = w1_copied[Nx+2, 3]
            end
        end

        if number_of_timesteps < 3
            hu1_copied = reshape(collect(curr_hu1_dev), data_shape)
            fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                plot_title="hu i=$i, t=$(t) s")
            #w1_copied = reshape(collect(curr_w1_dev), data_shape)
            #fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
            #    plot_title="w i=$i")
        end        

    end

    


    npzwrite("runoff/data/$(subpath)/t.npy", save_t)
    npzwrite("runoff/data/$(subpath)/Q_infiltration.npy", Q_infiltrated)
    npzwrite("runoff/data/$(subpath)/runoff.npy", runoff)
    #println(save_t)
    #println(Q_infiltrated)
    #println(runoff)

    cons_mass_fig = plot_conservation_of_mass(subpath)
    swim_save("runoff/plots/$(subpath)/conservation_of_mass_fig.png", cons_mass_fig)

    fcg_fig = make_fcg_plot(subpath, rain_function)
    swim_save("runoff/plots/$(subpath)/fcg_hyd_outlet_infiltration_fig.png", fcg_fig) 

    fcg_fig = make_fcg_plot(subpath, rain_function, with_infiltration=false)
    swim_save("runoff/plots/$(subpath)/fcg_hyd_outlet_fig.png", fcg_fig) 

    hyd_fig = plot_hydrographs_at_location(subpath, rain_function, dx, dy)
    swim_save("runoff/plots/$(subpath)/fcg_hyd_1000_fig.png", hyd_fig) 

    display(fig)    
    return nothing
end

function make_fcg_plot(subpath, rain_function; with_infiltration=true)
    t = npzread("runoff/data/$(subpath)/t.npy")
    Q_infiltrated = npzread("runoff/data/$(subpath)/Q_infiltration.npy")
    runoff = npzread("runoff/data/$(subpath)/runoff.npy")
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

    fig = Plots.plot(t/60.0, [rain_Q outlet_Q], label=["rain Q" "runoff Q"], 
                     legend=:topleft, ylims=[0,10], title=subpath, ylabel="Discharge Q [m^3/s]", xlabel="minutes",
                     right_margin=20Plots.mm)
    if with_infiltration
        Plots.plot!(t/60, Q_infiltrated, label="infiltration Q")
    end                     
    Plots.plot!(twinx(), t/60.0, runoff, color=:red, label="runoff", ylims=[0,yaxis2lim], ylabel="Runoff volume [m^3]")
    display(fig)
    return fig
end
    
function plot_conservation_of_mass(subpath)
    t = npzread("runoff/data/$(subpath)/t.npy")
    mass = total_mass("runoff/data/$(subpath)", trailing_zeros=3)
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
    npzwrite("runoff/data/$(subpath)/w_$(index).npy", w1_copied)
    hu1_copied = reshape(collect(hu_dev), data_shape)
    npzwrite("runoff/data/$(subpath)/hu_$(index).npy", hu1_copied)
    hv1_copied = reshape(collect(hv_dev), data_shape)
    npzwrite("runoff/data/$(subpath)/hv_$(index).npy", hv1_copied)
    infiltration_copied = reshape(collect(infiltration_dev), data_shape)
    npzwrite("runoff/data/$(subpath)/infiltration_$(index).npy", infiltration_copied)
    
    append!(save_t, t)
    append!(Q_infiltrated, get_Q_infiltrated(infiltration_copied, dx, dy))
    append!(runoff, compute_runoff(subpath, "w_$(index)", dx, dy))
    
    fig = nothing
    if doPlot
        fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                    plot_title="hu t=$(t/60) min")
        swim_save("runoff/plots/$(subpath)/hu_$(index).png", fig) 
        fig = plotSurf(hv1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                    plot_title="hv t=$(t/60) min")
        swim_save("runoff/plots/$(subpath)/hv_$(index).png", fig) 
        fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
                    plot_title="w  t=$(t/60) min")
        swim_save("runoff/plots/$(subpath)/w_$(index).png", fig) 
    end
    return fig
end

# Make output folders.
subpath = "dambreak_double"; rain_function = nothing
#subpath = "d_conservation_of_rain_1_1"; rain_function = rain_fcg_1_4;
#subpath = "d_fcg_case_1_1_bsafric"
#subpath = "d_fcg_case_1_1"; rain_function = rain_fcg_1_1;
#subpath = "d_fcg_case_1_2"; rain_function = rain_fcg_1_2;
#subpath = "d_fcg_case_1_3"; rain_function = rain_fcg_1_3;
#subpath = "d_fcg_case_1_4"; rain_function = rain_fcg_1_4;
#subpath = "d_fcg_case_1_5"; rain_function = rain_fcg_1_5;

#subpath = "d_fcg_case_2_100"; rain_function = rain_fcg_1_1; x0 = 100
#subpath = "d_fcg_case_2_1900"; rain_function = rain_fcg_1_1; x0 = 1900

#subpath = "d_fcg_case_3"; rain_function = rain_fcg_3; x0 = 200

mkpath("runoff/plots/$(subpath)/")
mkpath("runoff/data/$(subpath)/")


#@time run_stuff(subpath, rain_function)
#@time run_stuff(subpath, rain_function, topography=2, x0 = x0)
#@time run_stuff(subpath, rain_function, topography=3)
@time run_stuff(subpath, rain_function, init_dambreak=true)
