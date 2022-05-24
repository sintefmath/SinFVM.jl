include("../int32testing.jl")
const BLOCK_WIDTH = Int32(32)
const BLOCK_HEIGHT = Int32(16)

@inline @make_numeric_literals_32bits function clamp(i, low, high)
    return max(low, min(i, high))
end

@inline @make_numeric_literals_32bits function minmodSlope(left, center, right, theta) 
    backward = (center - left) * theta
    central = (right - left) * 0.5
    forward = (right - center) * theta
    
	return (0.25
		*copysign(1., backward)
		*(copysign(1., backward) + copysign(1., central))
		*(copysign(1., central) + copysign(1., forward))
		*min( min(abs(backward), abs(central)), abs(forward) ) )
end




@make_numeric_literals_32bits function julia_kp07!(
    Nx, Ny, dx, dy, dt,
    g, theta, step,
    eta0, hu0, hv0,
    eta1, hu1, hv1,
    Hi_glob, H,
    bc)

    tx = threadIdx().x
    ty = threadIdx().y


    blockStart_i = (blockIdx().x - 1)*blockDim().x
    blockStart_j = (blockIdx().y - 1)*blockDim().y
    
    ti = blockStart_i + threadIdx().x + 2
    tj = blockStart_j + threadIdx().y + 2

    # Q = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4), (BLOCK_HEIGHT+4), 3))
    # Qx = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+2),(BLOCK_HEIGHT+2), 3))
    # Hi = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4),(BLOCK_HEIGHT+4)))
    
    Q = CuStaticSharedArray(Float32, (36, 20, 3))
    Qx = CuStaticSharedArray(Float32, (34, 18, 3))
    Hi = CuStaticSharedArray(Float32, (36, 20))
    # Read eta0, hu0, hv0 and Hi into shmem:
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+4
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+4
            glob_j = clamp(blockStart_j + j, 1, Ny+4)
            glob_i = clamp(blockStart_i + i, 1, Nx+4)
             Q[i, j, 1] = eta0[glob_i, glob_j]
             Q[i, j, 2] = hu0[glob_i, glob_j]
             Q[i, j, 3] = hv0[glob_i, glob_j]
             Hi[i, j] = Hi_glob[glob_i, glob_j]
        end
    end
    sync_threads()    

    wall_bc_to_shmem!(Q, Nx, Ny, Int32(tx+2), Int32(ty+2), ti, tj)
    sync_threads()

    # Reconstruct Q in x-direction into Qx
    # 
    # Reconstruct slopes along x axis
    # Qx is here dQ/dx*0.5*dx
    # and represents [eta_x, hu_x, hv_x]
    reconstruct_slope_x!(Q, Qx, theta, tx, ty)

    sync_threads()

    # TODO: Skipping adjustSlope_x

    R1 = R2 = R3 = 0.
    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + 2
        j = ty + 2

        # Bottom topography source term along x 
        # TODO Desingularize and ensure h >= 0
        ST2 = bottom_source_term_x(Q, Qx, Hi, g, i, j)

        # TODO: Not written for dry cells
        F_flux_p_x, F_flux_p_y, F_flux_p_z = compute_single_flux_F(Q, Qx, Hi, g, tx+1, ty)
        F_flux_m_x, F_flux_m_y, F_flux_m_z = compute_single_flux_F(Q, Qx, Hi, g, tx         , ty)
        
        R1 = - (F_flux_p_x - F_flux_m_x) / dx
        R2 = - (F_flux_p_y - F_flux_m_y) / dx + ( - ST2/dx)
        R3 = - (F_flux_p_z - F_flux_m_z) / dx
 
    end
    sync_threads()

    # Reconstruct Q in y-direction into Qx
    # 
    # Reconstruct slopes along y axis
    # Qx is here dQ/dy*0.5*dy
    # and represents [eta_y, hu_y, hv_y]
    reconstruct_slope_y!(Q, Qx, theta, tx, ty)
    sync_threads()


    if (ti > 2 && tj > 2 && ti <= (Nx + 2) && (tj <= Ny + 2))
        i = tx + 2
        j = ty + 2
        
        # Bottom topography source term along y 
        # TODO Desingularize and ensure h >= 0
        ST3 = bottom_source_term_y(Q, Qx, Hi, g, i, j)

        # TODO: Not written for dry cells
        G_flux_p_x, G_flux_p_y, G_flux_p_z = compute_single_flux_G(Q, Qx, Hi, g, tx, ty+1)
        G_flux_m_x, G_flux_m_y, G_flux_m_z = compute_single_flux_G(Q, Qx, Hi, g, tx         , ty)
        
        R1 += - (G_flux_p_x - G_flux_m_x) / dy
        R2 += - (G_flux_p_y - G_flux_m_y) / dy
        R3 += - (G_flux_p_z - G_flux_m_z) / dy + ( - ST3/dy );

        if step == 0
             eta1[ti, tj] = Q[i, j, 1] + dt*R1
             hu1[ti, tj] = Q[i, j, 2] + dt*R2
              hv1[ti, tj] = Q[i, j, 3] + dt*R3
        elseif step == 1
            # RK2 ODE integrator
              eta1[ti, tj] = 0.5*( eta1[ti, tj] +  (Q[i, j, 1] + dt*R1))
              hu1[ti, tj] = 0.5*(  hu1[ti, tj] +  (Q[i, j, 2] + dt*R2))
              hv1[ti, tj] = 0.5*(  hv1[ti, tj] +  (Q[i, j, 3] + dt*R3))
        end
    end

    return nothing
end

@inline @make_numeric_literals_32bits function fillWithCrap!(Q, i, j)
    Q[i, j, 2] = i+j
    return nothing
end


@inline @make_numeric_literals_32bits function wall_bc_to_shmem!(Q, 
                           Nx, Ny, 
                           i, j,
                           ti, tj)
    # Global and local indices:
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

@inline @make_numeric_literals_32bits function reconstruct_Hx(Hi, i  , j)
    return 0.5*(Hi[i  , j] + Hi[i  , j+1])
end
@inline @make_numeric_literals_32bits function reconstruct_Hy(Hi, i  , j)
    return 0.5*(Hi[i  , j] + Hi[i+1, j  ])
end

@inline @make_numeric_literals_32bits function reconstruct_slope_x!(Q,
                               Qx, 
                               theta, tx, ty)
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT
        l = j + 2
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+2
            k = i + 1
            for p=1:3
                 Qx[i, j, p] = 0.5 * minmodSlope(Q[k-1, l, p], Q[k, l, p], Q[k+1, l, p], theta);
            end
        end
    end
    return nothing
end

@inline @make_numeric_literals_32bits function reconstruct_slope_y!(Q,
                              Qx, 
                              theta, tx, ty)
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+2
        l = j + 1
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH
            k = i + 2
            for p=1:3
                 Qx[i, j, p] = 0.5 * minmodSlope(Q[k, l-1, p], Q[k, l, p], Q[k, l+1, p], theta);
            end
        end
    end
    return nothing
end

@inline @make_numeric_literals_32bits function bottom_source_term_x(Q,
                              Qx,
                              Hi,
                              g, i, j)
                               eta_p = Q[i, j, 1] + Qx[i-1, j-2, 1]
                               eta_m = Q[i, j, 1] - Qx[i-1, j-2, 1]
    RHx_p = reconstruct_Hx(Hi, i+1, j)
    RHx_m = reconstruct_Hx(Hi, i         , j)
    
    #RHx_p =  0.5*(Hi[i+1, j] + Hi[i+1, j+1])
    H_x = RHx_p - RHx_m
    #h = Q[j,i,1] + (RHx_p + RHx_m)/2.0
    # TODO Desingularize and ensure h >= 0
    return -0.5*g*H_x *(eta_p + RHx_p + eta_m + RHx_m)
end

@inline @make_numeric_literals_32bits function bottom_source_term_y(Q,
                              Qx,
                              Hi,
                              g, i, j)
                               eta_p = Q[i, j, 1] + Qx[i-2, j-1, 1]
                               eta_m = Q[i, j, 1] - Qx[i-2, j-1, 1]
    RHy_p = reconstruct_Hy(Hi, i, j+1)
    RHy_m = reconstruct_Hy(Hi, i, j         )
    
    H_y = RHy_p - RHy_m
    # TODO Desingularize and ensure h >= 0
    return -0.5*g*H_y *(eta_p + RHy_p + eta_m + RHy_m)
end

@inline @make_numeric_literals_32bits function compute_single_flux_F(Q,
                               Qx,
                               Hi,
                               g, qxi, qxj)
    # Indices into Q with input being indices into Qx
    qj = qxj + 2
    qi = qxi + 1;

    # Q at interface from the right (p) and left (m)
     Qpx = Q[qi+1, qj, 1] - Qx[qxi+1, qxj, 1]
     Qpy = Q[qi+1, qj, 2] - Qx[qxi+1, qxj, 2]
     Qpz = Q[qi+1, qj, 3] - Qx[qxi+1, qxj, 3]
     Qmx = Q[qi  , qj, 1] + Qx[qxi  , qxj, 1]
     Qmy = Q[qi  , qj, 2] + Qx[qxi  , qxj, 2]
     Qmz = Q[qi  , qj, 3] + Qx[qxi  , qxj, 3]
    

    #float3 Qp = make_float3(Q[0][l][k+1] - Qx[0][j][i+1],
    #                        Q[1][l][k+1] - Qx[1][j][i+1],
    #                        Q[2][l][k+1] - Qx[2][j][i+1]);
    #float3 Qm = make_float3(Q[0][l][k  ] + Qx[0][j][i  ],
    #                        Q[1][l][k  ] + Qx[1][j][i  ],
    #                        Q[2][l][k  ] + Qx[2][j][i  ]);
                                
    # Computed flux with respect to the reconstructed bottom elevation at the interface
    RHx = reconstruct_Hx(Hi, qi+1, qj);
    F1, F2, F3 = central_upwind_flux_bottom(Qmx, Qmy, Qmz, Qpx, Qpy, Qpz, RHx, g);

    return F1, F2, F3 
end

@inline @make_numeric_literals_32bits function compute_single_flux_G(Q,
                               Qx,
                               Hi,
                               g, qxi, qxj)
    # Indices into Q with input being indices into Qx
    qj = qxj + 1
    qi = qxi + 2;

    # Q at interface from the north (p) and south (m)
    # Note that we swap hu and hv
     Qpx = Q[qi, qj+1, 1] - Qx[qxi, qxj+1, 1]
     Qpy = Q[qi, qj+1, 3] - Qx[qxi, qxj+1, 3]
     Qpz = Q[qi, qj+1, 2] - Qx[qxi, qxj+1, 2]
     Qmx = Q[qi, qj  , 1] + Qx[qxi, qxj  , 1]
     Qmy = Q[qi, qj  , 3] + Qx[qxi, qxj  , 3]
     Qmz = Q[qi, qj  , 2] + Qx[qxi, qxj  , 2]
    

    #float3 Qp = make_float3(Q[0][l][k+1] - Qx[0][j][i+1],
    #                        Q[1][l][k+1] - Qx[1][j][i+1],
    #                        Q[2][l][k+1] - Qx[2][j][i+1]);
    #float3 Qm = make_float3(Q[0][l][k  ] + Qx[0][j][i  ],
    #                        Q[1][l][k  ] + Qx[1][j][i  ],
    #                        Q[2][l][k  ] + Qx[2][j][i  ]);
                                
    # Computed flux with respect to the reconstructed bottom elevation at the interface
    RHy = reconstruct_Hy(Hi, qi, qj+1);
    G1, G2, G3 = central_upwind_flux_bottom(Qmx, Qmy, Qmz, Qpx, Qpy, Qpz, RHy, g);

    # Swap fluxes back
    return G1, G3, G2 
end

@inline @make_numeric_literals_32bits function central_upwind_flux_bottom(Qmx, Qmy, Qmz, 
                                    Qpx, Qpy, Qpz, 
                                    RH, g)
    # TODO: Not specialized for dry cells!
    hp = Qpx + RH
    up = Float32(Qpy / hp)
    Fpx, Fpy, Fpz = F_func_bottom(Qpx, Qpy, Qpz, hp, up, g)
    cp = sqrt(g*hp)

    hm = Qmx + RH
    um = Float32(Qmy / hm)
    Fmx, Fmy, Fmz = F_func_bottom(Qmx, Qmy, Qmz, hm, um, g)
    cm = sqrt(g*hm)
    
    am = min(min(um-cm, up-cp), 0.0)
    ap = max(max(um+cm, up+cp), 0.0)

    Fx = ((ap*Fmx - am*Fpx) + ap*am*(Qpx-Qmx))/(ap-am);
    Fy = ((ap*Fmy - am*Fpy) + ap*am*(Qpy-Qmy))/(ap-am);
    Fz = ((ap*Fmz - am*Fpz) + ap*am*(Qpz-Qmz))/(ap-am);

    return Fx, Fy, Fz
end

@inline @make_numeric_literals_32bits function F_func_bottom(qx, qy, qz, h, u, g) 
    Fx = qy;                       
    Fy = qy*u + 0.5*g*(h*h);      
    Fz = qz*u;                     
    return Fx, Fy, Fz;
end


@make_numeric_literals_32bits function compute_flux(numflux, reconstruct, u, cellindex)
    return numflux(reconstruct_right(u, cellindex)) - numflux(reconstruct_left(u, cellindex+1, +1))
end