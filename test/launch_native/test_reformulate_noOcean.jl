using Test

using CUDA
using Plots
using ProgressMeter
using NPZ
using PyCall

#using .GPUOceanUtils
include("GPUOceanUtils.jl")
include("SWEPlotting.jl")

include("swe_kp07_pure.jl")

function run_stuff(subpath)

  
    flattenarr(x) = collect(Iterators.flatten(x))
    MyType = Float32
    Nx = 503
    Ny = 487
    dt::Float32 = 1.0
    g::Float32 = 9.81
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
    dx::Float32 = dy::Float32 = 10.0
    #makeCentralBump!(eta0, Nx, Ny, dx, dy, bumpheight=2.2, offset=-1.2)
    #makeBathymetry!(H, Hi, Nx, Ny, dx, dy, amplitude=0.75, slope=[0.0001, -0.00005])
    makeCentralBump!(w0, Nx, Ny, dx, dy, bumpheight=1.1, offset=-1.2+3, centerX=0.2, centerY=0.8)
    makeBathymetry!(B, Bi, Nx, Ny, dx, dy, amplitude=0.2, slope=[0.0001, -0.00005], offset=2)
    ensureNonnegativeDepth!(w0, B)
    #println(size(H))
    #display(plotSurf(w0, B, dx, dy, Nx, Ny, show_ground=true))
    #fig = plotSurf(eta0, H, dx, dy, Nx, Ny)
    #display(fig)    

    #return nothing

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


    #npzwrite("data/eta_init.npy", eta0)

    number_of_timesteps = 10000
    number_of_plots = 10
    save_every = Integer(floor(number_of_timesteps/number_of_plots))
    fig = nothing
    w1_copied = nothing
    lowest_w = zeros(number_of_plots+1)

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


        #CUDA.@profile 
        @cuda threads=num_threads blocks=num_blocks julia_kp07!(
            Nx, Ny, dx, dy, dt,
            g, theta, step,
            curr_w0_dev, curr_hu0_dev, curr_hv0_dev,
            curr_w1_dev, curr_hu1_dev, curr_hv1_dev,
            Bi_dev, B_dev,
            bc)

        
        if step == 1
            if i % save_every == 2
                w1_copied = reshape(collect(curr_w1_dev), data_shape)
                npzwrite("data/$(subpath)/w_$(div(i, save_every)).npy", w1_copied)
                hu1_copied = reshape(collect(curr_hu1_dev), data_shape)
                npzwrite("data/$(subpath)/hu_$(div(i, save_every)).npy", hu1_copied)
                hv1_copied = reshape(collect(curr_hv1_dev), data_shape)
                npzwrite("data/$(subpath)/hv_$(div(i, save_every)).npy", hv1_copied)
                
                fig = plotSurf(hu1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                             plot_title="hu i=$i")
                save("plots/$(subpath)/hu_$(div(i, save_every)).png", fig) 
                fig = plotSurf(hv1_copied, B, dx, dy, Nx, Ny, show_ground=false, 
                             plot_title="hv i=$i")
                save("plots/$(subpath)/hv_$(div(i, save_every)).png", fig) 
                fig = plotSurf(w1_copied, B, dx, dy, Nx, Ny, show_ground=true, 
                             plot_title="w i=$i")
                save("plots/$(subpath)/w_$(div(i, save_every)).png", fig) 
                lowest_w[(div(i, save_every))+1] = w1_copied[Nx+2, 3]
            end
        end

    end

    display(fig)
    println("hei")
    println(typeof(w1_copied))
    #println(maximum(w1_copied))
    #println(maximum(-B))
    #println(maximum(w1_copied) - maximum(-B))
    #println(lowest_w)
    
end
# Make output folders.
subpath = "noOcean"
mkpath("plots/$(subpath)/")
mkpath("data/$(subpath)/")
@time run_stuff(subpath)
