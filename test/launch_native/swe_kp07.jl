

function julia_kp07!(
    Nx, Ny, dx, dy, dt,
    g, theta, step,
    eta0, hu0, hv0,
    eta1, hu1, hv1,
    Hi, H,
    bc)

    tx = threadIdx().x
    ty = threadIdx().y

    ti = (blockIdx().x - 1)*blockDim().x + threadIdx().x + 2
    tj = (blockIdx().y - 1)*blockDim().y + threadIdx().y + 2;

    if (ti > 2 && tj > 2 && ti <= Nx + 2 && tj <= Ny + 2)
        eta1[tj, ti] = eta0[tj, ti]
        hu1[tj, ti] = hu0[tj, ti]
        hv1[tj, ti] = hv0[tj, ti]
    end

    return nothing
end