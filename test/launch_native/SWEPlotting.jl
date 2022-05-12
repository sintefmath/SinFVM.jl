using CUDA, Test, GLMakie 
GLMakie.activate!()


function plotSurf(eta, H, dx, dy, nx, ny; show_ground=true, 
    depth_cutoff=1e-5, zlim_max=1.2, zlim_min=-1.0, plot_title="")
    @assert(size(eta) == size(H))
    #println(size(eta))
    gcx::Int = (size(eta, 1) - nx) / 2
    gcy::Int = (size(eta, 2) - ny) / 2
    gc = (gcx, gcy)
    #println(gc, (nx, ny))
    x = range(dx/2, (nx-0.5)*dx, nx)/1000
    y = range(dy/2, (ny-0.5)*dy, ny)/1000
    #println(size(-H[gc[1]+1:nx+gc[1], gc[2]+1:ny+gc[2]]))
    #println(size(eta[gc[1]+1:nx+gc[1], gc[2]+1:ny+gc[2]]))

    fig = Figure(resolution=(700, 700), fontsize=14)
    axs = [Axis3(fig[1, 1]; aspect=(1, 1, 0.2), title=plot_title)]

    GLMakie.surface!(x, y, eta[gc[1]+1:nx+gc[1], gc[2]+1:ny+gc[2]].-(100*depth_cutoff), 
                    #color=fill((:blue,0.4),100,100),
                    colormap=("Blues", 0.7),
                    shading=true, transparency=true)
    #         framestyle = false, minogrid = false, colorbar = false, 
    #         camera=(30, 20))
    if show_ground
        GLMakie.surface!(axs[1], x, y, -H[gc[1]+1:nx+gc[1], gc[2]+1:ny+gc[2]], 
                        color=fill((:brown,1),100,100),
                        #colorrange = (-2,-1), highclip=(:red, 0.3), 
                        shading=true, transparancy=true)
    end
    #GLMakie.zlims!(axs[1], (zlim_min, zlim_max))
    fig
end
