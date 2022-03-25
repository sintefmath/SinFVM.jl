
const BLOCK_WIDTH = 16
const BLOCK_HEIGHT = 8

function clamp(i, low, high)
    return max(low, min(i, high))
end

function minmodSlope(left::Float32, center::Float32, right::Float32, theta::Float32) 
    backward = (center - left) * theta
    central = (right - left) * 0.5
    forward = (right - center) * theta
    
	return (0.25
		*copysign(1.0, backward)
		*(copysign(1.0, backward) + copysign(1.0, central))
		*(copysign(1.0, central) + copysign(1.0, forward))
		*min( min(abs(backward), abs(central)), abs(forward) ) )
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
    tj = blockStart_j + threadIdx().y + 2

    Q = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4), (BLOCK_HEIGHT+4), 3))
    Qx = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+2),(BLOCK_HEIGHT+2), 3))
    Hi_shmem = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4),(BLOCK_HEIGHT+4)))

    # Read eta0, hu0, hv0 and Hi into shmem:
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+4
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+4
            glob_j = clamp(blockStart_j + j, 1, Ny+4)
            glob_i = clamp(blockStart_i + i, 1, Nx+4)
            Q[i, j, 1] = eta0[glob_i, glob_j]
            Q[i, j, 2] = hu0[glob_i, glob_j]
            Q[i, j, 3] = hv0[glob_i, glob_j]
            Hi_shmem[i, j] = Hi[glob_i, glob_j]
        end
    end
    sync_threads()    

    if (bc == 1)
        #wall_bc_to_shmem!(Q, Nx, Ny)
        i = tx + 2
        j = ty + 2
        if (ti == 3)
            # First index within domain in x (west)
            Q[i-1, j, 1] =  Q[i, j, 1]
            Q[i-1, j, 2] = -Q[i, j, 2]
            Q[i-1, j, 3] =  Q[i, j, 3]
                
            Q[i-2, j, 1] =  Q[i+1, j, 1]
            Q[i-2, j, 2] = -Q[i+1, j, 2]
            Q[i-2, j, 3] =  Q[i+1, j, 3]
        end
        if (ti == Nx+2)
            # Last index within domain in x (east)
            Q[i+1, j, 1] =  Q[i, j, 1]
            Q[i+1, j, 2] = -Q[i, j, 2]
            Q[i+1, j, 3] =  Q[i, j, 3]
                
            Q[i+2, j, 1] =  Q[i-1, j, 1]
            Q[i+2, j, 2] = -Q[i-1, j, 2]
            Q[i+2, j, 3] =  Q[i-1, j, 3]
        end
        if (tj == 3) 
            # First index in domain in y (south)
            Q[i, j-1, 1] =  Q[i, j, 1]
            Q[i, j-1, 2] =  Q[i, j, 2]
            Q[i, j-1, 3] = -Q[i, j, 3]
                
            Q[i, j-2, 1] =  Q[i, j+1, 1]
            Q[i, j-2, 2] =  Q[i, j+1, 2]
            Q[i, j-2, 3] = -Q[i, j+1, 3]
        end
        if (tj == Ny+2)
            # Last index in domain in y (north)
            Q[i, j+1, 1] =  Q[i, j, 1]
            Q[i, j+1, 2] =  Q[i, j, 2]
            Q[i, j+1, 3] = -Q[i, j, 3]
                
            Q[i, j+2, 1] =  Q[i, j-1, 1]
            Q[i, j+2, 2] =  Q[i, j-1, 2]
            Q[i, j+2, 3] = -Q[i, j-1, 3]
           
        end 
        sync_threads()
    end

    # Reconstruct Q in x-direction into Qx
    # 
    # Reconstruct slopes along x axis
    # Qx is here dQ/dx*0.5*dx
    # and represents [eta_x, hu_x, hv_x]
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT
        l = j + 2
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+2
            k = i + 1
            for p=1:3
                Qx[i, j, p] = 0.5 * minmodSlope(Q[k-1, l, p], Q[k, l, p], Q[k+1, l, p], theta);
            end
        end
    end
    sync_threads()

    # TODO: Skipping adjustSlope_x





    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + 2
        j = ty + 2

        eta1[ti, tj] = Qx[i-1, j-2, 1];
        hu1[ti, tj]  = Qx[i-1, j-2, 2];
        hv1[ti, tj]  = Qx[i-1, j-2, 3];
        #hv1[ti, tj]  = Hi_shmem[i, j];

        #eta1[ti, tj] = eta0[ti, tj]
        #hu1[ti, tj] = hu0[ti, tj]
        #hv1[ti, tj] = hv0[ti, tj]
    end

    return nothing
end


function wall_bc_to_shmem!(Q::CuDeviceArray{Float32, 3, 3}, 
                           Nx::Int32, Ny::Int32)
    # Global and local indices:
    i = threadIdx().x + 2
    j = threadIdx().y + 2

    ti = (blockIdx().x - 1)*blockDim().x + i
    tj = (blockIdx().y - 1)*blockDim().y + j

    if (ti == 3)
        # First index within domain in x (west)
        Q[i-1, j, 1] =  Q[i, j, 1]
        Q[i-1, j, 2] = -Q[i, j, 2]
        Q[i-1, j, 3] =  Q[i, j, 3]
            
        Q[i-2, j, 1] =  Q[i+1, j, 1]
        Q[i-2, j, 2] = -Q[i+1, j, 2]
        Q[i-2, j, 3] =  Q[i+1, j, 3]
    end
    if (ti == Nx+2)
        # Last index within domain in x (east)
        Q[i+1, j, 1] =  Q[i, j, 1]
        Q[i+1, j, 2] = -Q[i, j, 2]
        Q[i+1, j, 3] =  Q[i, j, 3]
            
        Q[i+2, j, 1] =  Q[i-1, j, 1]
        Q[i+2, j, 2] = -Q[i-1, j, 2]
        Q[i+2, j, 3] =  Q[i-1, j, 3]
    end
    if (tj == 3) 
        # First index in domain in y (south)
        Q[i, j-1, 1] =  Q[i, j, 1]
        Q[i, j-1, 2] =  Q[i, j, 2]
        Q[i, j-1, 3] = -Q[i, j, 3]
            
        Q[i, j-2, 1] =  Q[i, j+1, 1]
        Q[i, j-2, 2] =  Q[i, j+1, 2]
        Q[i, j-2, 3] = -Q[i, j+1, 3]
    end
    if (tj == Ny+2)
        # Last index in domain in y (north)
        Q[i, j+1, 1] =  Q[i, j, 1]
        Q[i, j+1, 2] =  Q[i, j, 2]
        Q[i, j+1, 3] = -Q[i, j, 3]
            
        Q[i, j+2, 1] =  Q[i, j-1, 1]
        Q[i, j+2, 2] =  Q[i, j-1, 2]
        Q[i, j+2, 3] = -Q[i, j-1, 3]
       
    end
    return nothing

end