
const BLOCK_WIDTH = 32
const BLOCK_HEIGHT = 16

function clamp(i, low, high)
    if (i > high)
        @cuprintln("woops!")
    elseif (i < low)
        @cuprintln("whaaat?")
    end
    return max(low, min(i, high))
end


function julia_kp07!(
    Nx, Ny, dx, dy, dt,
    g, theta, step,
    eta0, hu0, hv0,
    eta1, hu1, hv1,
    Hi, H,
    bc)

    tx = threadIdx().x
    ty = threadIdx().y

    blockStart_i = (blockIdx().x - 1)*blockDim().x
    blockStart_j = (blockIdx().y - 1)*blockDim().y
    
    ti = blockStart_i + threadIdx().x + 2
    tj = blockStart_j + threadIdx().y + 2;

    Qshmem = CuStaticSharedArray(Float32, (3,(4+BLOCK_HEIGHT),(4+BLOCK_WIDTH)))

    
    # Read eta0, hu0 and hv0 into shmem:
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+4
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+4
            glob_j = clamp(blockStart_j + j, 1, Ny+4)
            glob_i = clamp(blockStart_i + i, 1, Nx+4)
            Qshmem[1, j, i] = eta0[glob_j, glob_i];
            Qshmem[2, j, i] = hu0[glob_j, glob_i];
            Qshmem[3, j, i] = hv0[glob_j, glob_i];
        end
    end
    
    sync_threads()    


    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + 2
        j = ty + 2
        eta1[tj, ti] = Qshmem[1, j, i];
        hu1[tj, ti]  = Qshmem[2, j, i];
        hv1[tj, ti]  = Qshmem[3, j, i];
    
        #eta1[tj, ti] = eta0[tj, ti]
        #hu1[tj, ti] = hu0[tj, ti]
        #hv1[tj, ti] = hv0[tj, ti]
    end

    return nothing
end