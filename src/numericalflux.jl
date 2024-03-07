abstract type NumericalFlux end

struct Rusanov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (rus::Rusanov)(faceminus, faceplus)
    fluxminus = rus.eq(XDIR, faceminus...)
    fluxplus = rus.eq(XDIR, faceplus...)

    eigenvalue_minus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceminus...)
    eigenvalue_plus = compute_max_abs_eigenvalue(rus.eq, XDIR, faceplus...)

    eigenvalue_max = max(eigenvalue_minus, eigenvalue_plus)

    F = 0.5 .* (fluxminus .+ fluxplus) .- 0.5 * eigenvalue_max .* (faceplus .- faceminus)

    return F
end


struct Godunov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (god::Godunov)(faceminus, faceplus)
    f(u) = god.eq(XDIR, u...)
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))

    F = max.(fluxminus, fluxplus)
    return F
end

struct CentralUpwind{T} <: NumericalFlux
    eq::ShallowWaterEquations1D{T}
end

function (centralupwind::CentralUpwind)(faceminus, faceplus)
    fluxminus = centralupwind.eq(XDIR, faceminus...)
    fluxplus = centralupwind.eq(XDIR, faceplus...)

    eigenvalues_minus = compute_eigenvalues(centralupwind.eq, XDIR, faceminus...) # compute_max_eigenvalue(centralupwind.eq, XDIR, faceminus...)
    eigenvalues_plus = compute_eigenvalues(centralupwind.eq, XDIR, faceplus...)  # compute_max_eigenvalue(centralupwind.eq, XDIR, faceplus...)

    aplus = max.(eigenvalues_plus[1], eigenvalues_minus[1], 0.0)
    aminus = min.(eigenvalues_plus[2], eigenvalues_minus[2], 0.0)

    F = (aplus .* fluxminus - aminus .* fluxplus) ./ (aplus - aminus) + ((aplus .* aminus) ./ (aplus - aminus)) .* (faceplus - faceminus)
    return F
end


function compute_flux!(backend, F::NumericalFlux, output, left, right, grid, equation::Equation, direction::XDIRT)
    Δx = compute_dx(grid)
    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        output[imiddle] -= 1 / Δx * (F(right[imiddle], left[iright]) - F(right[ileft], left[imiddle]))
        nothing
    end
end
