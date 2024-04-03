struct CentralUpwind{E<:AllSWE} <: NumericalFlux
    eq::E #ShallowWaterEquations1D{T, S}
end

Adapt.@adapt_structure CentralUpwind


function (centralupwind::CentralUpwind)(faceminus, faceplus, direction::Direction)
    centralupwind(centralupwind.eq, faceminus, faceplus, direction)
end

function (centralupwind::CentralUpwind)(::Equation, faceminus, faceplus, direction::Direction)

    fluxminus = centralupwind.eq(direction, faceminus...)
    fluxplus = centralupwind.eq(direction, faceplus...)

    eigenvalues_minus = compute_eigenvalues(centralupwind.eq, direction, faceminus...)
    eigenvalues_plus = compute_eigenvalues(centralupwind.eq, direction, faceplus...)

    aplus = max.(eigenvalues_plus[1], eigenvalues_minus[1], 0.0)
    aminus = min.(eigenvalues_plus[2], eigenvalues_minus[2], 0.0)

    F = (aplus .* fluxminus - aminus .* fluxplus) ./ (aplus - aminus) + ((aplus .* aminus) ./ (aplus - aminus)) .* (faceplus - faceminus)
    return F, max(abs(aplus), abs(aminus))
end


function (centralupwind::CentralUpwind)(::AllPracticalSWE, faceminus, faceplus, direction::Direction)
    fluxminus = zero(faceminus)
    eigenvalues_minus = zero(faceminus)
    if faceminus[1] > centralupwind.eq.depth_cutoff
        fluxminus = centralupwind.eq(direction, faceminus...)
        eigenvalues_minus = compute_eigenvalues(centralupwind.eq, direction, faceminus...)
    end

    fluxplus = zero(faceplus)
    eigenvalues_plus = zero(faceplus)
    if faceplus[1] > centralupwind.eq.depth_cutoff
        fluxplus = centralupwind.eq(direction, faceplus...)
        eigenvalues_plus = compute_eigenvalues(centralupwind.eq, direction, faceplus...)
    end

    aplus = max.(eigenvalues_plus[1], eigenvalues_minus[1], zero(eigenvalues_plus[1]))
    aminus = min.(eigenvalues_plus[2], eigenvalues_minus[2], zero(eigenvalues_plus[2]))

    # Check for dry states
    if abs(aplus - aminus) < centralupwind.eq.desingularizing_kappa
        return zero(faceminus), zero(aminus)
    end

    F = (aplus .* fluxminus .- aminus .* fluxplus) ./ (aplus .- aminus) + ((aplus .* aminus) ./ (aplus .- aminus)) .* (faceplus .- faceminus)
    return F, max(abs(aplus), abs(aminus))
end

