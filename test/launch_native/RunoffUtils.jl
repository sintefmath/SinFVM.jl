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
    infiltration_horton_fcg(t)
    fc = 3.272e-5
    f0 = 1.977e-4
    k  = 2.43e-3 
    return fc + (f0 - fc)*exp(-k*t)
end


# Rain source terms
@inline @make_numeric_literals_32bits function 
    rain_fcg_1_1(x, y, t)
    
    if x < 1000 && t < 250.0*60.0
        return 0.00025
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_2(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 1000 
        if t < rain_step
            return 0.0005
        elseif t < 2*rain_step
            return 0.00025
        elseif t < 3*rain_step
            return 0.0001495
        elseif t < 4*rain_step
            return 9.80392e-5
        elseif t < 5*rain_step
            return 0.0001495
        elseif t < 6*rain_step
            return 0.0003529
        end
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_3(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 1000 
        if t < rain_step
            return 0.0003734
        elseif t < 2*rain_step
            return 0.0001867
        elseif t < 3*rain_step
            return 0.0001130
        elseif t < 4*rain_step
            return 7.3690e-5
        elseif t < 5*rain_step
            return 0.0001130
        elseif t < 6*rain_step
            return 0.0002653
        end
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_4(x, y, t)
    
    if x < 1000 && t < 50.0*60.0
        return 0.0005833
    end
    return 0.0
end

@inline @make_numeric_literals_32bits function 
    rain_fcg_1_5(x, y, t)
    rain_step = 250.0*60.0/6.0
    if x < 1000 
        if t < rain_step
            return 0.00037456
        elseif t < 2*rain_step
            return 7.39255e-5
        elseif t < 3*rain_step
            return 3.44986e-5
        elseif t < 4*rain_step
            return 0.000000
        elseif t < 5*rain_step
            return 0.0001109
        elseif t < 6*rain_step
            return 0.00026613
        end
    end
    return 0.0
end






# Utility functions that are used to compute parameters in the 
# source term functions
function _get_rates_1_1()
    rain_volume = 75000
    duration = 250*60.0
    area = 1000*20
    rain_per_meter_per_second = rain_volume/(area * duration)
    return rain_per_meter_per_second
end

function _get_rates_1_x(rain_volume, ratios; duration=nothing)
    if isnothing(duration)
        duration = 250.0*60.0/6.0
    end
    area = 1000*20
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

