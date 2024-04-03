struct Godunov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (god::Godunov)(faceminus, faceplus, direction)
    f(u) = god.eq(XDIR, u...)
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))

    F = max.(fluxminus, fluxplus)
    return F, max(compute_max_abs_eigenvalue(god.eq, XDIR, faceminus...),
        compute_max_abs_eigenvalue(god.eq, XDIR, faceplus...))
end