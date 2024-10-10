function (centralupwind::CentralUpwind)(::AllPracticalSWEStable, faceminus, faceplus, direction::Direction)
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
    
    if faceminus[1] < centralupwind.eq.depth_cutoff && faceplus[1] < centralupwind.eq.depth_cutoff
        return F, zero(aplus)
    end    
    return F, max(abs(aplus), abs(aminus))
end
