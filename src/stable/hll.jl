struct HLL{T} <: NumericalFlux
    eq::T
end

function (hll::HLL)(faceminus, faceplus, direction::Direction)




    h_r = faceminus[1]
    h_l = faceplus[1]

    if h_r < hll.eq.depth_cutoff && h_l < hll.eq.depth_cutoff
        return zero(faceminus), zero(eltype(faceminus))
    end

    hu_r = faceminus[2]
    hu_l = faceplus[2]

    u_r = 0.0
    u_l = 0.0

    flux_r = zero(faceminus)
    flux_l = zero(faceplus)

    if h_r > hll.eq.depth_cutoff
        u_r = hu_r / h_r
        flux_r = hll.eq(direction, faceminus...)
    end

    if h_l > hll.eq.depth_cutoff
        u_l = hu_l / h_l
        flux_l = hll.eq(direction, faceplus...)
    end
    grav = hll.eq.g

    # Roe averages
    h_hat = (h_r + h_l) / 2.0
    u_hat = (sqrt(h_r) * u_r + sqrt(h_l) * u_l) / (sqrt(h_r) + sqrt(h_l))
    c_hat = sqrt(grav * h_hat)

    lambda_1_l = u_l - sqrt(grav * h_l)
    lambda_2_r = u_r + sqrt(grav * h_r)

    s1 = min(lambda_1_l, u_hat - c_hat)
    s2 = max(lambda_2_r, u_hat + c_hat)

    if abs(s1 - s2) < hll.eq.desingularizing_kappa
        return zero(faceminus), zero(eltype(faceminus))
    end
    q_m = @. (flux_r - flux_l + s2 * faceminus - s1 * faceplus) / (s1 - s2)


    return q_m, max(abs(s1), abs(s2))
end

