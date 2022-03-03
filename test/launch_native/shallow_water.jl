using Test

using CUDA
using Plots
using ProgressMeter
using NPZ

function makeCentralBump!(eta, nx, ny, dx, dy)
    H0 = 60.0
    x_center = dx * nx / 2.0
    y_center = dy * ny / 2.0
    for j in range(-2, ny + 2 - 1)
        for i in range(-2, nx + 2 - 1)
            x = dx * i - x_center
            y = dy * j - y_center
            sizenx = (0.15 * min(nx, ny) * min(dx, dy))^2
            if (sqrt(x^2 + y^2) < sizenx)
                eta[j+2+1, i+2+1] = exp(-(x^2 / sizenx + y^2 / sizenx))
            end
        end
    end
    nothing
end


function run_stuff()


    md_sw = CuModuleFile(joinpath(@__DIR__, "KP07_kernel.ptx"))
    swe_2D = CuFunction(md_sw, "swe_2D")

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
    N = Nx = Ny = 256
    N_tot = Nx * Ny
    dt::Float32 = 1.0
    g::Float32 = 9.81
    f::Float32 = 0.0012
    r::Float32 = 0.0
    wind_stress::Float32 = 0.0
    ngc = 2
    data_shape = (Ny + 2 * ngc, Nx + 2 * ngc)
    H = ones(MyType, data_shape) .* 60.0
    Hi = ones(MyType, data_shape .+ 1) .* 60.0
    eta0 = zeros(MyType, data_shape)
    u0 = zeros(MyType, data_shape)
    v0 = zeros(MyType, data_shape)

    eta1 = zeros(MyType, data_shape)
    u1 = zeros(MyType, data_shape)
    v1 = zeros(MyType, data_shape)
    dx::Float32 = dy::Float32 = 200.0
    makeCentralBump!(eta0, Nx, Ny, dx, dy)

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

    theta::Float32 = 1.3
    beta::Float32 = 0.0
    y_zero_reference_cell::Float32 = 0.0

    bc::Int32 = 1

    num_threads = (32, 16)
    num_blocks = (Ny ÷ num_threads[1], Nx ÷ num_threads[2])

    eta0_dev = CuArray(flattenarr(eta0))
    u0_dev = CuArray(flattenarr(u0))
    v0_dev = CuArray(flattenarr(v0))
    eta1_dev = CuArray(flattenarr(eta1))
    u1_dev = CuArray(flattenarr(u1))
    v1_dev = CuArray(flattenarr(v1))
    Hi_dev = CuArray(flattenarr(Hi))
    H_dev = CuArray(flattenarr(H))

    number_of_timesteps = 100_000
    @showprogress for i in 1:number_of_timesteps
        step::Int32 = (i + 1) % 2
        if i % 2 == 1
            curr_eta0_dev = eta0_dev
            curr_u0_dev = u0_dev
            curr_v0_dev = v0_dev
            curr_eta1_dev = eta1_dev
            curr_u1_dev = u1_dev
            curr_v1_dev = v1_dev
        else
            curr_eta0_dev = eta1_dev
            curr_u0_dev = u1_dev
            curr_v0_dev = v1_dev
            curr_eta1_dev = eta0_dev
            curr_u1_dev = u0_dev
            curr_v1_dev = v0_dev
        end

        cudacall(swe_2D, signature,
            Int32(Nx), Int32(Ny), dx, dy, dt,
            g, theta, f, beta, y_zero_reference_cell, r, step,
            curr_eta0_dev, Int32(data_shape[1] * sizeof(Float32)),
            curr_u0_dev, Int32(data_shape[1] * sizeof(Float32)),
            curr_v0_dev, Int32(data_shape[1] * sizeof(Float32)),
            curr_eta1_dev, Int32(data_shape[1] * sizeof(Float32)),
            curr_u1_dev, Int32(data_shape[1] * sizeof(Float32)),
            curr_v1_dev, Int32(data_shape[1] * sizeof(Float32)),
            Hi_dev, Int32((data_shape[1] + 1) * sizeof(Float32)),
            H_dev, Int32(data_shape[1] * sizeof(Float32)),
            bc, bc, bc, bc, wind_stress,
            threads = num_threads, blocks = num_blocks)
        save_every = 200
        if step == 1
            if i % save_every == 2
                eta1_copied = reshape(collect(curr_eta1_dev), data_shape)
                npzwrite("data/eta_$(div(i, save_every)).npy", eta1_copied)
                heatmap(eta1_copied, clim=(0.0, 1.0), c=:viridis, aspect_ratio=1)
                title!("i = $i")
                savefig("plots/eta_$(div(i, save_every)).png") 
            end
        end

    end
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
mkpath("plots")
mkpath("data")
@time run_stuff()