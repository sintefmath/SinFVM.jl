#module GPUOceanUtils
#export makeCentralBump

function makeCentralBump!(eta, nx, ny, dx, dy, bumpheight=1.0)
    H0 = 60.0
    x_center = dx * nx / 2.0
    y_center = dy * ny / 2.0
    for j in range(-2, ny + 2 - 1)
        for i in range(-2, nx + 2 - 1)
            x = dx * i - x_center
            y = dy * j - y_center
            sizenx = (0.15 * min(nx, ny) * min(dx, dy))^2
            if (sqrt(x^2 + y^2) < sizenx)
                eta[j+2+1, i+2+1] = bumpheight * exp(-(x^2 / sizenx + y^2 / sizenx))
            end
        end
    end
    nothing
end



function compareArrays(eta1, hu1, hv1, eta2, hu2, hv2, 
        data_shape; doPlot=true)
    eta1_h = reshape(collect(eta1), data_shape)
    hu1_h = reshape(collect(hu1), data_shape)
    hv1_h = reshape(collect(hv1), data_shape)
    eta2_h = reshape(collect(eta2), data_shape)
    hu2_h = reshape(collect(hu2), data_shape)
    hv2_h = reshape(collect(hv2), data_shape)
    
    eta_diff = eta1_h .- eta2_h
    hu_diff  = hu1_h .- hu2_h
    hv_diff  = hv1_h .- hv2_h

    if !(all(eta_diff .== 0) && all(hu_diff .== 0) && all(hv_diff .== 0))
        if doPlot
            plot_array = Any[]

            field_array = [eta1_h, eta2_h, eta_diff ,
                        hu1_h, hu2_h, hu_diff, 
                        hv1_h, hv2_h, hv_diff]
            titles = ["eta1", "eta2", "eta diff",
                    "hu1", "hu2", "hu diff",
                    "hv1", "hv2", "hv diff"]
            for i in 1:9
                push!(plot_array, heatmap(field_array[i], 
                                        c=:viridis, aspect_ratio=1, 
                                        title=titles[i], titlefontsize=10))
            end
            plot(plot_array..., layout=(3,3))
        else
            max_eta_diff = maximum(broadcast(abs, eta_diff))
            max_hu_diff = maximum(broadcast(abs, hu_diff))
            max_hv_diff = maximum(broadcast(abs, hv_diff))
            print("Results differ!\n")
            print("norms: eta = $(max_eta_diff), hu = $(max_hu_diff), hv = $(max_hv_diff): " )
                    
        end
    else
        print("Results are the same!\n")
    end

end    


#end