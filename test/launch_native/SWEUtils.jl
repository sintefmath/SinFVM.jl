include("../int32testing.jl")
using CUDA


# Intermediate CUDA-call function since the CUDA kernel did
# not love the idea of optional arguments...
function call_kp07!(num_threads, num_blocks,
    Nx, Ny, dx, dy, dt,
    g, theta, step,
    curr_w0_dev, curr_hu0_dev, curr_hv0_dev,
    curr_w1_dev, curr_hu1_dev, curr_hv1_dev,
    Bi_dev, B_dev,
    bc;
    friction_handle=nothing, friction_constant=0.0f0)

    if friction_constant > 0
        @assert(!isnothing(friction_handle))
    end

    @cuda threads=num_threads blocks=num_blocks julia_kp07!(
        Nx, Ny, dx, dy, dt,
        g, theta, step,
        curr_w0_dev, curr_hu0_dev, curr_hv0_dev,
        curr_w1_dev, curr_hu1_dev, curr_hv1_dev,
        Bi_dev, B_dev,
        bc,
        friction_handle, friction_constant)
    return nothing
end

## -------------------------------
## FRICTION FUNCTIONS
## -------------------------------

@inline @make_numeric_literals_32bits function 
    friction_bsa2012(c, h, u, v)

    velocity = sqrt(u*u + v*v)
    denom = cbrt(h)*h
    return -c*velocity/denom
end

@inline @make_numeric_literals_32bits function 
    friction_fcg2016(c, h, u, v)
    
    velocity = sqrt(u*u + v*v)
    denom = cbrt(h)*h*h
    return -c*velocity/denom
end

@inline @make_numeric_literals_32bits function 
    friction_bh2021(c, h, u, v)
    
    velocity = sqrt(u*u + v*v)
    denom = h*h
    return -c*velocity/denom
end

