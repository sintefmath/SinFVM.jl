
const BLOCK_WIDTH = 16
const BLOCK_HEIGHT = 8

function clamp(i, low, high)
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

    Q = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4), (BLOCK_HEIGHT+4), 3))
    #Qx = CuStaticSharedArray(Float32, (3,(BLOCK_HEIGHT+2),(BLOCK_WIDTH+2)))
    Hi_shmem = CuStaticSharedArray(Float32, ((BLOCK_HEIGHT+4),(BLOCK_WIDTH+4)))

    # Read eta0, hu0, hv0 and Hi into shmem:
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+4
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+4
            glob_j = clamp(blockStart_j + j, 1, Ny+4)
            glob_i = clamp(blockStart_i + i, 1, Nx+4)
            Q[i, j, 1] = eta0[glob_i, glob_j];
            Q[i, j, 2] = hu0[glob_i, glob_j];
            Q[i, j, 3] = hv0[glob_i, glob_j];
            Hi_shmem[i, j] = i-1 #100*(glob_i-1) + glob_j-1 #Hi[j, i]
        end
    end



    sync_threads()    

    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + 2
        j = ty + 2
        eta1[ti, tj] = Q[i, j, 1];
        hu1[ti, tj]  = Q[i, j, 2];
        hv1[ti, tj]  = Q[i, j, 3];
        #hv1[ti, tj]  = Hi_shmem[i, j];
    
        #eta1[ti, tj] = eta0[ti, tj]
        #hu1[ti, tj] = hu0[ti, tj]
        #hv1[ti, tj] = hv0[ti, tj]
    end

    return nothing
end