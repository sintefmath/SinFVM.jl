#module GPUOceanUtils
#export makeCentralBump

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



#end