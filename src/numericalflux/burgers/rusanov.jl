struct Rusanov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (rus::Rusanov)(faceminus, faceplus, direction)
    fluxminus = rus.eq(XDIR, faceminus...)
    fluxplus = rus.eq(XDIR, faceplus...)

    eigenvalue_minus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceminus...)
    eigenvalue_plus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceplus...)

    eigenvalue_max = max(eigenvalue_minus, eigenvalue_plus)

    F = 0.5 .* (fluxminus .+ fluxplus) .- 0.5 * eigenvalue_max .* (faceplus .- faceminus)

    return F, eigenvalue_max
end
