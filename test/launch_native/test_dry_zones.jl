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

function run_stuff(subpath; use_julia::Bool = true)

  
    flattenarr(x) = collect(Iterators.flatten(x))
    MyType = Float32
    Nx = 503
    Ny = 487
    dt::Float32 = 1.0
    g::Float32 = 9.81
    f::Float32 = 0.0
    r::Float32 = 0.0
    wind_stress::Float32 = 0.0
    ngc = 2
    data_shape = (Nx + 2 * ngc, Ny + 2 * ngc)
    H = ones(MyType, data_shape) 
    Hi = ones(MyType, data_shape .+ 1)
    eta0 = zeros(MyType, data_shape)
    hu0 = zeros(MyType, data_shape)
    hv0 = zeros(MyType, data_shape)

    eta1 = zeros(MyType, data_shape)
    hu1 = zeros(MyType, data_shape)
    hv1 = zeros(MyType, data_shape)
    dx::Float32 = dy::Float32 = 10.0
    #makeCentralBump!(eta0, Nx, Ny, dx, dy, bumpheight=2.2, offset=-1.2)
    #makeBathymetry!(H, Hi, Nx, Ny, dx, dy, amplitude=0.75, slope=[0.0001, -0.00005])
    makeCentralBump!(eta0, Nx, Ny, dx, dy, bumpheight=1.1, offset=-1.2, centerX=0.2, centerY=0.8)
    makeBathymetry!(H, Hi, Nx, Ny, dx, dy, amplitude=0.2, slope=[0.0001, -0.00005])
    ensureNonnegativeDepth!(eta0, H)
    #println(size(H))
    #display(plotSurf(eta0, H, dx, dy, Nx, Ny, show_ground=true))
    #fig = plotSurf(eta0, H, dx, dy, Nx, Ny)
    #display(fig)    

    #return nothing

    theta::Float32 = 1.3
    beta::Float32 = 0.0
    y_zero_reference_cell::Float32 = 0.0

    bc::Int32 = 1

    md_rot_sw = swe_rot_2D = signature = nothing
    num_threads = num_blocks = nothing

    eta0_dev = hu0_dev = hv0_dev = eta1_dev = hu1_dev = hv1_dev = nothing
    H_dev = Hi_dev = nothing

    if use_julia
        num_threads = (BLOCK_WIDTH, BLOCK_HEIGHT)
        num_blocks = Tuple(cld.([Nx, Ny], num_threads))

        eta0_dev = CuArray(eta0)
        hu0_dev = CuArray(hu0)
        hv0_dev = CuArray(hv0)

        eta1_dev = CuArray(eta1)
        hu1_dev = CuArray(hu1)
        hv1_dev = CuArray(hv1)

        Hi_dev = CuArray(Hi)
        H_dev  = CuArray(H)

    else
        md_rot_sw = CuModuleFile(joinpath(@__DIR__, "KP07_rot_kernel.ptx"))
        swe_rot_2D = CuFunction(md_rot_sw, "swe_rot_2D")

        num_threads = (32, 16)
        num_blocks = Tuple(cld.([Nx, Ny], num_threads))


        signature = Tuple{
            Int32,Int32,
            Float32,Float32,Float32,Float32,
            Float32,
            Float32,
            Float32,
            Float32,Float32,Int32,CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            Int32,Int32,Int32,Int32,
            Float32
        }

        eta0_dev = CuArray(flattenarr(eta0))
        hu0_dev = CuArray(flattenarr(hu0))
        hv0_dev = CuArray(flattenarr(hv0))
        eta1_dev = CuArray(flattenarr(eta1))
        hu1_dev = CuArray(flattenarr(hu1))
        hv1_dev = CuArray(flattenarr(hv1))
        Hi_dev = CuArray(flattenarr(Hi))
        H_dev = CuArray(flattenarr(H))
    end


    



    #npzwrite("data/eta_init.npy", eta0)

    number_of_timesteps = 100000
    number_of_plots = 15
    save_every = Integer(floor(number_of_timesteps/number_of_plots))
    fig = nothing
    eta1_copied = nothing
    lowest_eta = zeros(number_of_plots+1)

    @showprogress for i in 1:number_of_timesteps
        step::Int32 = (i + 1) % 2
        if i % 2 == 1
            curr_eta0_dev = eta0_dev
            curr_hu0_dev = hu0_dev
            curr_hv0_dev = hv0_dev
            curr_eta1_dev = eta1_dev
            curr_hu1_dev = hu1_dev
            curr_hv1_dev = hv1_dev
        else
            curr_eta0_dev = eta1_dev
            curr_hu0_dev = hu1_dev
            curr_hv0_dev = hv1_dev
            curr_eta1_dev = eta0_dev
            curr_hu1_dev = hu0_dev
            curr_hv1_dev = hv0_dev
        end


        if use_julia 
            CUDA.@profile @cuda threads=num_threads blocks=num_blocks julia_kp07!(
                Nx, Ny, dx, dy, dt,
                g, theta, step,
                curr_eta0_dev, curr_hu0_dev, curr_hv0_dev,
                curr_eta1_dev, curr_hu1_dev, curr_hv1_dev,
                Hi_dev, H_dev,
                bc)

        else
            cudacall(swe_rot_2D, signature,
                Int32(Nx), Int32(Ny), dx, dy, dt,
                g, theta, f, beta, y_zero_reference_cell, r, step,
                curr_eta0_dev, Int32(data_shape[1] * sizeof(Float32)),
                curr_hu0_dev, Int32(data_shape[1] * sizeof(Float32)),
                curr_hv0_dev, Int32(data_shape[1] * sizeof(Float32)),
                curr_eta1_dev, Int32(data_shape[1] * sizeof(Float32)),
                curr_hu1_dev, Int32(data_shape[1] * sizeof(Float32)),
                curr_hv1_dev, Int32(data_shape[1] * sizeof(Float32)),
                Hi_dev, Int32((data_shape[1] + 1) * sizeof(Float32)),
                H_dev, Int32(data_shape[1] * sizeof(Float32)),
                bc, bc, bc, bc, wind_stress,
                threads = num_threads, blocks = num_blocks)
        end

        if step == 1
            if i % save_every == 2
                eta1_copied = reshape(collect(curr_eta1_dev), data_shape)
                npzwrite("data/$(subpath)/eta_$(div(i, save_every)).npy", eta1_copied)
                if (div(i, save_every) == 8)
                    hu1_copied = reshape(collect(curr_hu1_dev), data_shape)
                    hv1_copied = reshape(collect(curr_hv1_dev), data_shape)
                    npzwrite("data/$(subpath)/hu_$(div(i, save_every)).npy", hu1_copied)
                    npzwrite("data/$(subpath)/hv_$(div(i, save_every)).npy", hv1_copied)
                end
                fig = plotSurf(eta1_copied, H, dx, dy, Nx, Ny, show_ground=true, 
                             plot_title="i=$i")
                save("plots/$(subpath)/eta_$(div(i, save_every)).png", fig) 
                lowest_eta[(div(i, save_every))+1] = eta1_copied[Nx+2, 3]
                if false && (div(i, save_every)) == 14
                    display(fig)
                    println("hei")
                    println(maximum(eta1_copied))
                    println(maximum(-H))
                    println(maximum(eta1_copied) - maximum(-H))

                    return nothing
                end
            end
        end

    end

    display(fig)
    println("hei")
    println(maximum(eta1_copied))
    println(maximum(-H))
    println(maximum(eta1_copied) - maximum(-H))
    println(lowest_eta)
    # heatmap(eta1_copied)
    # swe_2D(
    #         int nx_, int ny_,
    #         float dx_, float dy_, float dt_,
    #         float g_,

    #         float theta_,

    #         float f_, //< Coriolis coefficient
    #         float beta_, //< Coriolis force f_ + beta_*(y-y0)
    #         float y_zero_reference_cell_, // the cell row representing y0 (y0 at southern face)

    #         float r_, //< Bottom friction coefficient

    #         int step_,

    #         //Input h^n
    #         float* eta0_ptr_, int eta0_pitch_,
    #         float* hu0_ptr_, int hu0_pitch_,
    #         float* hv0_ptr_, int hv0_pitch_,

    #         //Output h^{n+1}
    #         float* eta1_ptr_, int eta1_pitch_,
    #         float* hu1_ptr_, int hu1_pitch_,
    #         float* hv1_ptr_, int hv1_pitch_,

    #         // Depth at cell intersections (i) and mid-points (m)
    #         float* Hi_ptr_, int Hi_pitch_,
    #         float* Hm_ptr_, int Hm_pitch_,

    #         // Boundary conditions (1: wall, 2: periodic, 3: numerical sponge)
    #         int bc_north_, int bc_east_, int bc_south_, int bc_west_,

    #         float wind_stress_t_)
end
# Make output folders.

use_julia = false
subpath = use_julia ? "dry_jl" : "dry"
mkpath("plots/$(subpath)/")
mkpath("data/$(subpath)/")
@time run_stuff(subpath, use_julia = use_julia)
