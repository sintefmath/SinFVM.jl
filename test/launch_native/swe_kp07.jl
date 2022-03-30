
const BLOCK_WIDTH = Int32(16)
const BLOCK_HEIGHT = Int32(8)

function clamp(i, low, high)
    return max(low, min(i, high))
end

function minmodSlope(left::Float32, center::Float32, right::Float32, theta::Float32) 
    backward = (center - left) * theta
    central = (right - left) * 0.5f0
    forward = (right - center) * theta
    
	return (0.25f0
		*copysign(1.0f0, backward)
		*(copysign(1.0f0, backward) + copysign(1.0f0, central))
		*(copysign(1.0f0, central) + copysign(1.0f0, forward))
		*min( min(abs(backward), abs(central)), abs(forward) ) )
end




function julia_kp07!(
    Nx::Int32, Ny::Int32, dx::Float32, dy::Float32, dt::Float32,
    g::Float32, theta::Float32, step::Int32,
    eta0, hu0, hv0,
    eta1, hu1, hv1,
    Hi_glob, H,
    bc)

    tx::Int32 = threadIdx().x
    ty::Int32 = threadIdx().y


    blockStart_i::Int32 = (blockIdx().x - 1)*blockDim().x
    blockStart_j::Int32 = (blockIdx().y - 1)*blockDim().y
    
    ti::Int32 = blockStart_i + threadIdx().x + 2
    tj::Int32 = blockStart_j + threadIdx().y + 2

    Q = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4), (BLOCK_HEIGHT+4), 3))
    Qx = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+2),(BLOCK_HEIGHT+2), 3))
    Hi = CuStaticSharedArray(Float32, ((BLOCK_WIDTH+4),(BLOCK_HEIGHT+4)))

    # Read eta0, hu0, hv0 and Hi into shmem:
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+4
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+4
            glob_j = clamp(blockStart_j + j, 1, Ny+4)
            glob_i = clamp(blockStart_i + i, 1, Nx+4)
            @inbounds Q[i, j, 1] = eta0[glob_i, glob_j]
            @inbounds Q[i, j, 2] = hu0[glob_i, glob_j]
            @inbounds Q[i, j, 3] = hv0[glob_i, glob_j]
            @inbounds Hi[i, j] = Hi_glob[glob_i, glob_j]
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

    R1 = R2 = R3 = 0.0f0
    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + Int32(2)
        j = ty + Int32(2)

        # Bottom topography source term along x 
        # TODO Desingularize and ensure h >= 0
        ST2 = bottom_source_term_x(Q, Qx, Hi, g, i, j)

        # TODO: Not written for dry cells
        F_flux_p_x, F_flux_p_y, F_flux_p_z = compute_single_flux_F(Q, Qx, Hi, g, tx+Int32(1), ty)
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


    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        i = tx + Int32(2)
        j = ty + Int32(2)
        
        # Bottom topography source term along y 
        # TODO Desingularize and ensure h >= 0
        ST3 = bottom_source_term_y(Q, Qx, Hi, g, i, j)

        # TODO: Not written for dry cells
        G_flux_p_x, G_flux_p_y, G_flux_p_z = compute_single_flux_G(Q, Qx, Hi, g, tx, ty+Int32(1))
        G_flux_m_x, G_flux_m_y, G_flux_m_z = compute_single_flux_G(Q, Qx, Hi, g, tx         , ty)
        
        R1 += - (G_flux_p_x - G_flux_m_x) / dy
        R2 += - (G_flux_p_y - G_flux_m_y) / dy
        R3 += - (G_flux_p_z - G_flux_m_z) / dy + ( - ST3/dy );

        if step == 0
            @inbounds eta1[ti, tj] = Q[i, j, 1] + dt*R1
            @inbounds  hu1[ti, tj] = Q[i, j, 2] + dt*R2
            @inbounds  hv1[ti, tj] = Q[i, j, 3] + dt*R3
        elseif step == 1
            # RK2 ODE integrator
            @inbounds eta1[ti, tj] = 0.5f0*( eta1[ti, tj] +  (Q[i, j, 1] + dt*R1))
            @inbounds  hu1[ti, tj] = 0.5f0*(  hu1[ti, tj] +  (Q[i, j, 2] + dt*R2))
            @inbounds  hv1[ti, tj] = 0.5f0*(  hv1[ti, tj] +  (Q[i, j, 3] + dt*R3))
        end
    end

    return nothing
end

function fillWithCrap!(Q::CuDeviceArray{Float32, 3, 3}, i, j)
    Q[i, j, 2] = i+j
    return nothing
end


function wall_bc_to_shmem!(Q::CuDeviceArray{Float32, 3, 3}, 
                           Nx::Int32, Ny::Int32, 
                           i::Int32, j::Int32,
                           ti::Int32, tj::Int32)
    # Global and local indices:
    if (ti == 3)
        # First index within domain in x (west)
        @inbounds Q[i-1, j, 1] =  Q[i, j, 1]
        @inbounds Q[i-1, j, 2] = -Q[i, j, 2]
        @inbounds Q[i-1, j, 3] =  Q[i, j, 3]
            
        @inbounds Q[i-2, j, 1] =  Q[i+1, j, 1]
        @inbounds Q[i-2, j, 2] = -Q[i+1, j, 2]
        @inbounds Q[i-2, j, 3] =  Q[i+1, j, 3]
    end
    if (ti == Nx+2)
        # Last index within domain in x (east)
        @inbounds Q[i+1, j, 1] =  Q[i, j, 1]
        @inbounds Q[i+1, j, 2] = -Q[i, j, 2]
        @inbounds Q[i+1, j, 3] =  Q[i, j, 3]
            
        @inbounds Q[i+2, j, 1] =  Q[i-1, j, 1]
        @inbounds Q[i+2, j, 2] = -Q[i-1, j, 2]
        @inbounds Q[i+2, j, 3] =  Q[i-1, j, 3]
    end
    if (tj == 3) 
        # First index in domain in y (south)
        @inbounds Q[i, j-1, 1] =  Q[i, j, 1]
        @inbounds Q[i, j-1, 2] =  Q[i, j, 2]
        @inbounds Q[i, j-1, 3] = -Q[i, j, 3]
            
        @inbounds Q[i, j-2, 1] =  Q[i, j+1, 1]
        @inbounds Q[i, j-2, 2] =  Q[i, j+1, 2]
        @inbounds Q[i, j-2, 3] = -Q[i, j+1, 3]
    end
    if (tj == Ny+2)
        # Last index in domain in y (north)
        @inbounds Q[i, j+1, 1] =  Q[i, j, 1]
        @inbounds Q[i, j+1, 2] =  Q[i, j, 2]
        @inbounds Q[i, j+1, 3] = -Q[i, j, 3]
            
        @inbounds Q[i, j+2, 1] =  Q[i, j-1, 1]
        @inbounds Q[i, j+2, 2] =  Q[i, j-1, 2]
        @inbounds Q[i, j+2, 3] = -Q[i, j-1, 3]
        
    end 
    return nothing
end

#function reconstruct_Hx(Hi::CuDeviceMatrix{Float32, 3}, i , j)
function reconstruct_Hx(Hi, i , j)
        return 0.5f0*(Hi[i  , j] + Hi[i  , j+1])
end
function reconstruct_Hy(Hi::CuDeviceMatrix{Float32, 3}, i::Int32  , j::Int32)
    return 0.5f0*(Hi[i  , j] + Hi[i+1, j  ])
end

function reconstruct_slope_x!(Q::CuDeviceArray{Float32, 3, 3},
                               Qx::CuDeviceArray{Float32, 3, 3}, 
                               theta::Float32, tx::Int32, ty::Int32)
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT
        l = j + 2
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH+2
            k = i + 1
            for p=1:3
                @inbounds Qx[i, j, p] = 0.5f0 * minmodSlope(Q[k-1, l, p], Q[k, l, p], Q[k+1, l, p], theta);
            end
        end
    end
    return nothing
end

function reconstruct_slope_y!(Q::CuDeviceArray{Float32, 3, 3},
                              Qx::CuDeviceArray{Float32, 3, 3}, 
                              theta::Float32, tx::Int32, ty::Int32)
    for j = ty:BLOCK_HEIGHT:BLOCK_HEIGHT+2
        l = j + 1
        for i = tx:BLOCK_WIDTH:BLOCK_WIDTH
            k = i + 2
            for p=1:3
                @inbounds Qx[i, j, p] = 0.5f0 * minmodSlope(Q[k, l-1, p], Q[k, l, p], Q[k, l+1, p], theta);
            end
        end
    end
    return nothing
end

function bottom_source_term_x(Q::CuDeviceArray{Float32, 3, 3},
                              Qx::CuDeviceArray{Float32, 3, 3},
                              Hi::CuDeviceMatrix{Float32, 3},
                              g::Float32, i::Int32, j::Int32)
    @inbounds eta_p = Q[i, j, 1] + Qx[i-1, j-2, 1]
    @inbounds eta_m = Q[i, j, 1] - Qx[i-1, j-2, 1]
    RHx_p = reconstruct_Hx(Hi, i+1, j)
    RHx_m = reconstruct_Hx(Hi, i         , j)
    
    #RHx_p =  0.5*(Hi[i+1, j] + Hi[i+1, j+1])
    H_x = RHx_p - RHx_m
    #h = Q[j,i,1] + (RHx_p + RHx_m)/2.0
    # TODO Desingularize and ensure h >= 0
    return -0.5f0*g*H_x *(eta_p + RHx_p + eta_m + RHx_m)
end

function bottom_source_term_y(Q::CuDeviceArray{Float32, 3, 3},
                              Qx::CuDeviceArray{Float32, 3, 3},
                              Hi::CuDeviceMatrix{Float32, 3},
                              g::Float32, i::Int32, j::Int32)
    @inbounds eta_p = Q[i, j, 1] + Qx[i-2, j-1, 1]
    @inbounds eta_m = Q[i, j, 1] - Qx[i-2, j-1, 1]
    RHy_p = reconstruct_Hy(Hi, i, j+Int32(1))
    RHy_m = reconstruct_Hy(Hi, i, j         )
    
    H_y = RHy_p - RHy_m
    # TODO Desingularize and ensure h >= 0
    return -0.5f0*g*H_y *(eta_p + RHy_p + eta_m + RHy_m)
end

function compute_single_flux_F(Q::CuDeviceArray{Float32, 3, 3},
                               Qx::CuDeviceArray{Float32, 3, 3},
                               Hi::CuDeviceMatrix{Float32, 3},
                               g::Float32, qxi::Int32, qxj::Int32)
    # Indices into Q with input being indices into Qx
    qj = qxj + Int32(2)
    qi = qxi + Int32(1);

    # Q at interface from the right (p) and left (m)
    @inbounds Qpx = Q[qi+1, qj, 1] - Qx[qxi+1, qxj, 1]
    @inbounds Qpy = Q[qi+1, qj, 2] - Qx[qxi+1, qxj, 2]
    @inbounds Qpz = Q[qi+1, qj, 3] - Qx[qxi+1, qxj, 3]
    @inbounds Qmx = Q[qi  , qj, 1] + Qx[qxi  , qxj, 1]
    @inbounds Qmy = Q[qi  , qj, 2] + Qx[qxi  , qxj, 2]
    @inbounds Qmz = Q[qi  , qj, 3] + Qx[qxi  , qxj, 3]
    

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

function compute_single_flux_G(Q::CuDeviceArray{Float32, 3, 3},
                               Qx::CuDeviceArray{Float32, 3, 3},
                               Hi::CuDeviceMatrix{Float32, 3},
                               g::Float32, qxi::Int32, qxj::Int32)
    # Indices into Q with input being indices into Qx
    qj = qxj + Int32(1)
    qi = qxi + Int32(2);

    # Q at interface from the north (p) and south (m)
    # Note that we swap hu and hv
    @inbounds Qpx = Q[qi, qj+1, 1] - Qx[qxi, qxj+1, 1]
    @inbounds Qpy = Q[qi, qj+1, 3] - Qx[qxi, qxj+1, 3]
    @inbounds Qpz = Q[qi, qj+1, 2] - Qx[qxi, qxj+1, 2]
    @inbounds Qmx = Q[qi, qj  , 1] + Qx[qxi, qxj  , 1]
    @inbounds Qmy = Q[qi, qj  , 3] + Qx[qxi, qxj  , 3]
    @inbounds Qmz = Q[qi, qj  , 2] + Qx[qxi, qxj  , 2]
    

    #float3 Qp = make_float3(Q[0][l][k+1] - Qx[0][j][i+1],
    #                        Q[1][l][k+1] - Qx[1][j][i+1],
    #                        Q[2][l][k+1] - Qx[2][j][i+1]);
    #float3 Qm = make_float3(Q[0][l][k  ] + Qx[0][j][i  ],
    #                        Q[1][l][k  ] + Qx[1][j][i  ],
    #                        Q[2][l][k  ] + Qx[2][j][i  ]);
                                
    # Computed flux with respect to the reconstructed bottom elevation at the interface
    RHy = reconstruct_Hy(Hi, qi, qj+Int32(1));
    G1, G2, G3 = central_upwind_flux_bottom(Qmx, Qmy, Qmz, Qpx, Qpy, Qpz, RHy, g);

    # Swap fluxes back
    return G1, G3, G2 
end

function central_upwind_flux_bottom(Qmx::Float32, Qmy::Float32, Qmz::Float32, 
                                    Qpx::Float32, Qpy::Float32, Qpz::Float32, 
                                    RH::Float32, g::Float32)
    # TODO: Not specialized for dry cells!
    hp = Qpx + RH
    up = Qpy / hp
    Fpx, Fpy, Fpz = F_func_bottom(Qpx, Qpy, Qpz, hp, up, g)
    # https://github.com/JuliaArrays/StaticArrays.jl
    cp = sqrt(g*hp)

    hm = Qmx + RH
    um = Qmy / hm
    Fmx, Fmy, Fmz = F_func_bottom(Qmx, Qmy, Qmz, hm, um, g)
    cm = sqrt(g*hm)
    
    am = min(min(um-cm, up-cp), 0.0f0)
    ap = max(max(um+cm, up+cp), 0.0f0)

    Fx = ((ap*Fmx - am*Fpx) + ap*am*(Qpx-Qmx))/(ap-am);
    Fy = ((ap*Fmy - am*Fpy) + ap*am*(Qpy-Qmy))/(ap-am);
    Fz = ((ap*Fmz - am*Fpz) + ap*am*(Qpz-Qmz))/(ap-am);

    return Fx, Fy, Fz
end

function F_func_bottom(qx::Float32, qy::Float32, qz::Float32, h::Float32, u::Float32, g::Float32) 
    Fx = qy;                       
    Fy = qy*u + 0.5f0*g*(h*h);      
    Fz = qz*u;                     
    return Fx, Fy, Fz;
end