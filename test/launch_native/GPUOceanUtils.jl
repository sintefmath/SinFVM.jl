#module GPUOceanUtils
#export makeCentralBump

using CUDA, Test, Plots
include("swe_kp07_pure.jl")


function makeCentralBump!(w, nx, ny, dx, dy; centerX=0.5, centerY=0.5, bumpheight=1.0, offset=0.0)
    x_center = dx * nx * centerX
    y_center = dy * ny * centerY
    for j in range(-2, ny + 2 - 1)
        for i in range(-2, nx + 2 - 1)
            x = dx * i - x_center
            y = dy * j - y_center
            sizenx = (0.15 * min(nx, ny) * min(dx, dy))^2
            if (sqrt(x^2 + y^2) < sizenx)
                w[i+2+1, j+2+1] = bumpheight * exp(-(x^2 / sizenx + y^2 / sizenx)) + offset
            end
        end
    end
    nothing
end

function makeBathymetry!(B, Bi, nx, ny, dx, dy; amplitude=0.25, slope=[0.0, 0.0], offset=0.0)
    length = dx*nx*1.0
    height = dy*ny*1.0
    x0 = length/2.0
    y0 = height/2.0
    println("heisann")
    depth(x, y) = offset + (amplitude*((sin(π*(x/length)*4)^2 + sin(π*(y/height)*4)^2)) - (x-x0)*slope[1] - (y-y0)*slope[2])
    for j in range(-2, ny + 2 - 1)
        for i in range(-2, nx + 2 -1)
            x = dx*i
            y = dy*j
            B[i+2+1, j+2+1] = depth(x, y)
        end
    end
    for j in range(-2, ny + 2)
        for i in range(-2, nx + 2)
            x = dx*(i-0.5)
            y = dy*(j-0.5)
            Bi[i+2+1, j+2+1] = depth(x, y)
        end
    end
    

end

function ensureNonnegativeDepth!(w, B)
    @assert(size(w) == size(B))
    
    w[:,:] = broadcast(max, w, B)
    return nothing
end

function plotField(field; kwargs...)
    Plots.heatmap(transpose(field), 
        c=:viridis,
        aspect_ratio=1;
        kwargs...)
end


function compareArrays(w1, hu1, hv1, w2, hu2, hv2, 
        data_shape; doPlot=true, forcePlot=false)
    w1_h = reshape(collect(w1), data_shape)
    hu1_h = reshape(collect(hu1), data_shape)
    hv1_h = reshape(collect(hv1), data_shape)
    w2_h = reshape(collect(w2), data_shape)
    hu2_h = reshape(collect(hu2), data_shape)
    hv2_h = reshape(collect(hv2), data_shape)
    
    w_diff = w_h .- w_h
    hu_diff  = hu1_h .- hu2_h
    hv_diff  = hv1_h .- hv2_h

    if !(all(w_diff .== 0) && all(hu_diff .== 0) && all(hv_diff .== 0))
        max_w_diff = maximum(broadcast(abs, w_diff))
        max_hu_diff = maximum(broadcast(abs, hu_diff))
        max_hv_diff = maximum(broadcast(abs, hv_diff))
        if forcePlot || (doPlot && max(max_w_diff, max(max_hu_diff, max_hv_diff)) > 1e-5)

            plot_array = Any[]

            field_array = [w1_h, w2_h, w_diff ,
                        hu1_h, hu2_h, hu_diff, 
                        hv1_h, hv2_h, hv_diff]
            titles = ["w cuda", "w julia", "w diff",
                    "hu cuda", "hu julia", "hu diff",
                    "hv cuda", "hv julia", "hv diff"]
            for i in 1:9
                push!(plot_array, 
                      plotField(field_array[i], title=titles[i], 
                      titlefontsize=10))
            end
            display(Plots.plot(plot_array..., layout=(3,3)))
            display(plot_array[3])
            display(plot_array[6])
            display(plot_array[9])
        end
        print("Results differ!\n")
        print("norms: w = $(max_w_diff), hu = $(max_hu_diff), hv = $(max_hv_diff): " )
        
    else
        print("Results are the same!\n")
    end

end    

function compareNpyFiles(filename1, filename2)
    data1 = npzread(filename1)
    data2 = npzread(filename2)
    diff = data1 .- data2
    if !all(diff .== 0) 
        println("Results differ!")
        println("Max difference: $(maximum(broadcast(abs, diff))) ")
        display(plotField(diff))
    else
        println("Results are the same")
    end
    return nothing
end

function showMemoryStructureJl!(a, b, nx, ny)
    # Kernel to illustrate memory layout (row-major vs column major)
    tx = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    ty = (blockIdx().y - 1)*blockDim().y + threadIdx().y

    if ((tx <= nx) && (ty <= ny))
        a[tx, ty] = tx
        b[tx, ty] = ty
    end
    return nothing
end

function testMemoryLayout(;doPlot=true)
    nx::Int32 = 19
    ny::Int32 = 31
    a_h = zeros(Float32, (nx, ny))
    b_h = zeros(Float32, (nx, ny))
    
 
    num_threads = (4, 8)
    num_blocks =  (cld(nx, num_threads[1]), cld(ny, num_threads[2]))
    
    @info "num threads = $(num_threads), num blocks = $(num_blocks)"
    @info "num threads = $(num_threads), num blocks = $(num_blocks)"
    threads_total = num_threads .* num_blocks
    @info "total num threads = $(threads_total), (nx, ny) = $((nx, ny))"
    @info "nx*ny = $(nx*ny)"

    # Run julia:
    a_djl = CuArray(a_h)
    b_djl = CuArray(b_h)

    @cuda threads=num_threads blocks=num_blocks showMemoryStructureJl!(a_djl, b_djl, nx, ny)
    a_hjl = Array(a_djl)
    b_hjl = Array(b_djl)

    # Run native CUDA:
    md_sw = CuModuleFile(joinpath(@__DIR__, "KP07_kernel.ptx"))
    showMemoryStructureCu = CuFunction(md_sw, "showMemoryStructureCu")

    flattenarr(x) = collect(Iterators.flatten(x))
    
    a_dcu = CuArray(flattenarr(a_h))
    b_dcu = CuArray(flattenarr(b_h))
    
   
    cudacall(showMemoryStructureCu, 
             Tuple{CuPtr{Cfloat}, CuPtr{Cfloat},
                   Int32, Int32},
            a_dcu, b_dcu, nx, ny,
            threads=num_threads, blocks = num_blocks)

    rebuildarr(x) = reshape(collect(x), (ny, nx))
    
    a_hcu = rebuildarr(a_dcu)
    b_hcu = rebuildarr(b_dcu)
    
    if doPlot
        plot_array = Any[]
        push!(plot_array, plot(1:(nx*ny), 
            [flattenarr(a_hcu) flattenarr(a_hjl)],
                label=["a_hcu" "a_hjl"], title="x values"))
        push!(plot_array, plot(1:(nx*ny), 
            [flattenarr(b_hcu) flattenarr(b_hjl)],
            label=["b_hcu" "b_hjl"], title="y values"))
        display(plot(plot_array..., layout=(2,1)))
    end

    @test all(flattenarr(a_hcu) .== flattenarr(a_hjl))
    @test all(flattenarr(b_hcu) .== flattenarr(b_hjl))
    
    return nothing
end

function plot_cross_section_file(folder, wfile; x=nothing, y=nothing)
    w = npzread("$(folder)/$(wfile).npy")
    B = npzread("$(folder)/B.npy")
    plot_cross_section(w, B, x=x, y=y)
end

function plot_cross_section(w, B; x=nothing, y=nothing)
    @assert (!isnothing(x) || !isnothing(y))
    xaxis = nothing
    title = nothing
    if x == nothing
        @assert(isa(y, Integer))
        x = range(3,size(B,1)-2)
        xaxis = x
        title = "Cross section at y = $(y)"
    else
        @assert(isa(x, Integer))
        y = range(3, size(B,2)-2)
        xaxis = y
        title = "Cross section at x = $(x)"
    end
    Plots.plot(xaxis, [w[x, y], B[x, y]], title=title, label=["w", "B"])
end

#testMemoryLayout(doPlot=false)

function compute_max_timestep_from_files(folder, timestep, dx, dy; g=9.81)
    w  = npzread("$(folder)/w_$(timestep).npy");
    hu = npzread("$(folder)/hu_$(timestep).npy");
    hv = npzread("$(folder)/hv_$(timestep).npy");
    B = npzread("$(folder)/B.npy");
    return compute_max_timestep(w, B, hu, hv, dx, dy, g=g)
end

function compute_max_timestep(w, B, hu, hv, dx, dy; g=9.81, eps=10e-5)
    h = w - B
    h = desingularize_depth.(h)
    u = hu./h
    v = hv./h
    #u[h .< eps] .= 0.0
    #v[h .< eps] .= 0.0

    dt = 0.25*min(dx/maximum(abs.(u)+sqrt.(g*h)), 
                  dy/maximum(abs.(v)+sqrt.(g*h)))
    return dt
end




