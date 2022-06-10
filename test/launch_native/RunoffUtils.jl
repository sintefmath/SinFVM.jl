using Test, Plots

include("../int32testing.jl")


# Bathymetries

function _B_case_1(x)
    if x < 2000
        return 10 - (10.0/2000.0)*x
    end
    return -(20/2000.0)*(x-2000)
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

# Infiltration source term

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


# Rain source terms
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
        return 0.000291667
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

### Misc utilities

function get_runoff_h(subfolder, wfilename)
    B = npzread("runoff/data/$(subfolder)/B.npy")
    w = npzread("runoff/data/$(subfolder)/$(wfilename).npy")
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

    folder = "runoff/data/$(subpath)"
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

    rain = rain_function.(x, y, t)

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