
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

    return F, eigenvalue_max
end


struct Godunov{EquationType<:Equation} <: NumericalFlux
    eq::EquationType
end

function (god::Godunov)(faceminus, faceplus)
    f(u) = god.eq(XDIR, u...)
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))

    F = max.(fluxminus, fluxplus)
    return F, max(compute_max_abs_eigenvalue(god.eq, XDIR, faceminus...), 
        compute_max_abs_eigenvalue(god.eq, XDIR, faceplus...))
end

struct CentralUpwind{T, S} <: NumericalFlux
    eq::ShallowWaterEquations1D{T, S}
end

Adapt.@adapt_structure CentralUpwind


function (centralupwind::CentralUpwind)(faceminus, faceplus)
    centralupwind(centralupwind.eq, faceminus, faceplus)
end

function (centralupwind::CentralUpwind)(::Equation, faceminus, faceplus)

    fluxminus = centralupwind.eq(XDIR, faceminus...)
    fluxplus = centralupwind.eq(XDIR, faceplus...)
    
    eigenvalues_minus = compute_eigenvalues(centralupwind.eq, XDIR, faceminus...) # compute_max_eigenvalue(centralupwind.eq, XDIR, faceminus...)
    eigenvalues_plus = compute_eigenvalues(centralupwind.eq, XDIR, faceplus...)  # compute_max_eigenvalue(centralupwind.eq, XDIR, faceplus...)

    aplus = max.(eigenvalues_plus[1], eigenvalues_minus[1], 0.0)
    aminus = min.(eigenvalues_plus[2], eigenvalues_minus[2], 0.0)

    F = (aplus .* fluxminus - aminus .* fluxplus) ./ (aplus - aminus) + ((aplus .* aminus) ./ (aplus - aminus)) .* (faceplus - faceminus)
    return F, max(abs(aplus), abs(aminus))
end


function (centralupwind::CentralUpwind)(::ShallowWaterEquations1D, faceminus, faceplus)

    if faceminus[1] > centralupwind.eq.depth_cutoff
        fluxminus = centralupwind.eq(XDIR, faceminus...)
        eigenvalues_minus = compute_eigenvalues(centralupwind.eq, XDIR, faceminus...) # compute_max_eigenvalue(centralupwind.eq, XDIR, faceminus...)
    else
        fluxminus = zero(faceminus)
        eigenvalues_minus = zero(faceminus)
    end

    if faceplus[1] > centralupwind.eq.depth_cutoff
        fluxplus = centralupwind.eq(XDIR, faceplus...)
        eigenvalues_plus = compute_eigenvalues(centralupwind.eq, XDIR, faceplus...)  # compute_max_eigenvalue(centralupwind.eq, XDIR, faceplus...)
    else
        fluxplus = zero(faceplus)
        eigenvalues_plus = zero(faceplus)
    end

    aplus = max.(eigenvalues_plus[1], eigenvalues_minus[1], 0.0)
    aminus = min.(eigenvalues_plus[2], eigenvalues_minus[2], 0.0)

    # Check for dry states
    if abs(aplus - aminus) < centralupwind.eq.desingularizing_kappa
        return zero(faceminus), 0.0
    end

    F = (aplus .* fluxminus - aminus .* fluxplus) ./ (aplus - aminus) + ((aplus .* aminus) ./ (aplus - aminus)) .* (faceplus - faceminus)
    return F, max(abs(aplus), abs(aminus))
end


function compute_flux!(backend, F::NumericalFlux, output, left, right, wavespeeds, grid, equation::Equation, direction::XDIRT)
    Δx = compute_dx(grid, direction)

    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        F_right, speed_right = F(right[imiddle], left[iright])
        F_left, speed_left = F(right[ileft], left[imiddle])
        output[imiddle] -= 1 / Δx * (F_right - F_left)
        wavespeeds[imiddle] = max(speed_right, speed_left)
        nothing
    end

    return maximum(wavespeeds)
end
