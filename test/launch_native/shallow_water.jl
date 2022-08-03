using Test

using CUDA
using Plots
using ProgressMeter
using NPZ
using PyCall

#using .GPUOceanUtils
include("GPUOceanUtils.jl")

include("swe_kp07_pure.jl")


@make_numeric_literals_32bits function  compare_julia_and_cuda(; useJulia::Bool, number_of_timesteps=1)
    dataFolder = "data/plain"
    flattenarr(x) = collect(Iterators.flatten(x))

    MyType = Float32
    #N = Nx = Ny = 256
    Nx::Int32 = 270
    Ny::Int32 = 230
    #Nx::Int32 = 4096
    #Ny::Int32 = 4096
    #N_tot = Nx * Ny
    dt = 0.001
    g = 9.81
    f = 0.00
    r = 0.0
    dx = 0.5
    dy = 0.4

    ngc = 2
    data_shape = (Nx + 2 * ngc, Ny + 2 * ngc)
    H_h = ones(MyType, data_shape) .* 1
    Hi_h = ones(MyType, data_shape .+ 1) .* 1
    eta0 = zeros(MyType, data_shape)
    hu0 = zeros(MyType, data_shape)
    hv0 = zeros(MyType, data_shape)
    
    eta1 = zeros(MyType, data_shape)
    hu1 = zeros(MyType, data_shape)
    hv1 = zeros(MyType, data_shape)
    
    eta0 = npzread("data/plain/eta_final.npy")
    hu0 = npzread("data/plain/hu_final.npy")
    hv0 = npzread("data/plain/hv_final.npy")
    makeBathymetry!(H_h, Hi_h, Nx, Ny, dx, dy)


    eta0_dev = hu0_dev = hv0_dev = eta1_dev = hu1_dev = hv1_dev = nothing

    theta = 1.3
    
    bc = 1

    md_sw = swe_2D = signature = nothing
    num_threads = num_blocks = nothing
    
    if useJulia
        num_threads = (BLOCK_WIDTH, BLOCK_HEIGHT)
        num_blocks = Tuple(cld.([Nx, Ny], num_threads))

        eta0_dev = CuArray(eta0)
        hu0_dev = CuArray(hu0)
        hv0_dev = CuArray(hv0)

        eta1_dev = CuArray(eta1)
        hu1_dev = CuArray(hu1)
        hv1_dev = CuArray(hv1)

        Hi = CuArray(Hi_h)
        H  = CuArray(H_h)

        
        
        
    else
        num_threads = (32, 16)
        # num_threads = (width, height)
        num_blocks = Tuple(cld.([Nx, Ny], num_threads))

        md_sw = CuModuleFile(joinpath(@__DIR__, "KP07_kernel.ptx"))
        swe_2D = CuFunction(md_sw, "swe_2D")
        signature = Tuple{
            Int32,Int32,
            Float32,Float32,Float32,Float32,
            Float32,Float32,Int32,CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            CuPtr{Cfloat},Int32,
            Int32,Int32,Int32,Int32
        }



        eta0_dev = CuArray(flattenarr(eta0))
        hu0_dev = CuArray(flattenarr(hu0))
        hv0_dev = CuArray(flattenarr(hv0))
        eta1_dev = CuArray(flattenarr(eta1))
        hu1_dev = CuArray(flattenarr(hu1))
        hv1_dev = CuArray(flattenarr(hv1))
        Hi_dev = CuArray(flattenarr(Hi_h))
        H_dev = CuArray(flattenarr(H_h))
    end

    @showprogress for i in 1:number_of_timesteps
        step = (i + 1) % 2
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

        if useJulia

            CUDA.@profile @cuda threads=num_threads blocks=num_blocks julia_kp07!(
                Nx, Ny, dx, dy, dt,
                g, theta, step,
                curr_eta0_dev, curr_hu0_dev, curr_hv0_dev,
                curr_eta1_dev, curr_hu1_dev, curr_hv1_dev,
                Hi, H,
                bc)
        else
            CUDA.@profile cudacall(swe_2D, signature,
                    Int32(Nx), Int32(Ny), dx, dy, dt,
                    g, theta, r, step,
                    curr_eta0_dev, Int32(data_shape[1] * sizeof(Float32)),
                    curr_hu0_dev, Int32(data_shape[1] * sizeof(Float32)),
                    curr_hv0_dev, Int32(data_shape[1] * sizeof(Float32)),
                    curr_eta1_dev, Int32(data_shape[1] * sizeof(Float32)),
                    curr_hu1_dev, Int32(data_shape[1] * sizeof(Float32)),
                    curr_hv1_dev, Int32(data_shape[1] * sizeof(Float32)),
                    Hi_dev, Int32((data_shape[1] + 1) * sizeof(Float32)),
                    H_dev, Int32(data_shape[1] * sizeof(Float32)),
                    bc, bc, bc, bc,
                    threads = num_threads, blocks = num_blocks)

        end
    end

    return Array(eta1_dev), Array(hu1_dev), Array(hv1_dev), data_shape

end



function run_stuff(rotation::Bool,     
    number_of_timesteps = 30000
    )   

    md_sw = CuModuleFile(joinpath(@__DIR__, "KP07_kernel.ptx"))
    swe_2D = CuFunction(md_sw, "swe_2D")

    dataFolder = "data/plain"
    plotFolder = "plots/plain"

    signature = Tuple{
        Int32,Int32,
        Float32,Float32,Float32,Float32,
        Float32,Float32,Int32,CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        CuPtr{Cfloat},Int32,
        Int32,Int32,Int32,Int32
    }

    if rotation
        md_sw = CuModuleFile(joinpath(@__DIR__, "KP07_rot_kernel.ptx"))
        swe_2D = CuFunction(md_sw, "swe_rot_2D")

        dataFolder = "data/rot"
        plotFolder = "plots/rot"

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
    end
 
    mkpath(dataFolder)
    mkpath(plotFolder)


    #         float* eta0_ptr_, int eta0_pitch_,
    #         float* hu0_ptr_, int hu0_pitch_,
    #         float* hv0_ptr_, int hv0_pitch_,
    #         float* eta1_ptr_, int eta1_pitch_,
    #         float* hu1_ptr_, int hu1_pitch_,
    #         float* hv1_ptr_, int hv1_pitch_,
    #         float* Hi_ptr_, int Hi_pitch_,
    #         float* Hm_ptr_, int Hm_pitch_,
    flattenarr(x) = collect(Iterators.flatten(x))

    MyType = Float32
    #N = Nx = Ny = 256
    Nx::Int32 = 270
    Ny::Int32 = 230
    #N_tot = Nx * Ny
    dt::Float32 = 0.001
    g::Float32 = 9.81
    f::Float32 = 0.00
    r::Float32 = 0.0
    dx::Float32 = 0.5
    dy::Float32 = 0.4

    wind_stress::Float32 = 0.0
    ngc = 2
    data_shape = (Nx + 2 * ngc, Ny + 2 * ngc)
    H = ones(MyType, data_shape) .* 1
    Hi = ones(MyType, data_shape .+ 1) .* 1
    eta0 = zeros(MyType, data_shape)
    hu0 = zeros(MyType, data_shape)
    hv0 = zeros(MyType, data_shape)
    

    # H = python_simulator.H
    # eta0 = python_simulator.eta0
    # u0 = python_simulator.u0
    # v0 = python_simulator.v0

    eta1 = zeros(MyType, data_shape)
    hu1 = zeros(MyType, data_shape)
    hv1 = zeros(MyType, data_shape)
    makeCentralBump!(eta0, Nx, Ny, dx, dy; centerX=0.3, centerY=0.6)
    makeBathymetry!(H, Hi, Nx, Ny, dx, dy)
    
    #maxDt = 0.25*(dx/sqrt(61*g))
    #print("max Δt: $(maxDt)\n")

    theta::Float32 = 1.3
    beta::Float32 = 0.0
    y_zero_reference_cell::Float32 = 0.0

    bc::Int32 = 1

    num_threads = (32, 16)
    num_blocks = Tuple(cld.([Nx, Ny], num_threads))

    npzwrite("$(dataFolder)/eta_init.npy", eta0)

    
    eta0_dev = CuArray(flattenarr(eta0))
    hu0_dev = CuArray(flattenarr(hu0))
    hv0_dev = CuArray(flattenarr(hv0))
    eta1_dev = CuArray(flattenarr(eta1))
    hu1_dev = CuArray(flattenarr(hu1))
    hv1_dev = CuArray(flattenarr(hv1))
    Hi_dev = CuArray(flattenarr(Hi))
    H_dev = CuArray(flattenarr(H))




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

        pitch = Int32(data_shape[1] * sizeof(Float32))
        pitch_Hi = Int32((data_shape[1] + 1) * sizeof(Float32))

        if rotation
            cudacall(swe_2D, signature,
                Int32(Nx), Int32(Ny), dx, dy, dt,
                g, theta, f, beta, y_zero_reference_cell, r, step,
                curr_eta0_dev, pitch,
                curr_hu0_dev, pitch,
                curr_hv0_dev, pitch,
                curr_eta1_dev, pitch,
                curr_hu1_dev, pitch,
                curr_hv1_dev, pitch,
                Hi_dev, pitch_Hi,
                H_dev, pitch,
                bc, bc, bc, bc, wind_stress,
                threads = num_threads, blocks = num_blocks)
        else
            cudacall(swe_2D, signature,
                Int32(Nx), Int32(Ny), dx, dy, dt,
                g, theta, r, step,
                curr_eta0_dev, pitch,
                curr_hu0_dev, pitch,
                curr_hv0_dev, pitch,
                curr_eta1_dev, pitch,
                curr_hu1_dev, pitch,
                curr_hv1_dev, pitch,
                Hi_dev, pitch_Hi,
                H_dev, pitch,
                bc, bc, bc, bc,
                threads = num_threads, blocks = num_blocks)
        end

        save_every = number_of_timesteps ÷ 20
        if step == 1
            if i % save_every == 2
                
                #maxDt = 0.25*dx/reduce(max, broadcast(abs, curr_u1_dev) + sqrt.(g*(H_dev + curr_eta1_dev)))
                #print("max Δt: $(maxDt)\n")


                eta1_copied = reshape(collect(curr_eta1_dev), data_shape)
                npzwrite("$(dataFolder)/eta_$(div(i, save_every)).npy", eta1_copied)
                #plotField(eta1_copied, title ="i = $i")
               # savefig("$(plotFolder)/eta_$(div(i, save_every)).png") 
            end
        end
    end

    eta1_copied = reshape(collect(eta1_dev), data_shape)
    hu1_copied = reshape(collect(hu1_dev), data_shape)
    hv1_copied = reshape(collect(hv1_dev), data_shape)
    
    npzwrite("$(dataFolder)/eta_final.npy", eta1_copied)
    npzwrite("$(dataFolder)/hu_final.npy", hu1_copied)
    npzwrite("$(dataFolder)/hv_final.npy", hv1_copied)
    #plotField(eta1_copied, title="eta final")
#    savefig("$(plotFolder)/eta_final.png") 
   # plotField(hu1_copied, title="hu final")
    #savefig("$(plotFolder)/hu_final.png") 
   # plotField(hv1_copied, title="hv final")
   # savefig("$(plotFolder)/hv_final.png") 

    return Array(eta1_dev), Array(hu1_dev), Array(hv1_dev), data_shape
end

check_complete_kernel = false
if check_complete_kernel
    @time eta_rot, hu_rot, hv_rot, data_shape = run_stuff(true)
    @time eta_plain, hu_plain, hv_plain, data_shape = run_stuff(false)

    compareArrays(eta_rot, hu_rot, hv_rot, 
                  eta_plain, hu_plain, hv_plain, data_shape)
end

function runme()
    check_julia_kernel = true
    if check_julia_kernel
        num_iterations = 1
        @time eta_cuda, hu_cuda, hv_cuda, data_shape = compare_julia_and_cuda(useJulia=false, number_of_timesteps=num_iterations)
        @time eta_jl, hu_jl, hv_jl, data_shape = compare_julia_and_cuda(useJulia=true, number_of_timesteps=num_iterations)
        @time eta_cuda, hu_cuda, hv_cuda, data_shape = compare_julia_and_cuda(useJulia=false, number_of_timesteps=num_iterations)
        @time eta_jl, hu_jl, hv_jl, data_shape = compare_julia_and_cuda(useJulia=true, number_of_timesteps=num_iterations)

        compareArrays(eta_cuda, hu_cuda, hv_cuda, 
                    eta_jl, hu_jl, hv_jl, data_shape, doPlot=false,
                    forcePlot=false)
    end

end

runme()
runme()