# Copyright (c) 2024 SINTEF AS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    
    if faceminus[1] < centralupwind.eq.depth_cutoff && faceplus[1] < centralupwind.eq.depth_cutoff
        return F, zero(aplus)
    end    
    return F, max(abs(aplus), abs(aminus))
end


function (centralupwind::CentralUpwind)(::TwoLayerShallowWaterEquations1D, faceminus, faceplus,direction::Direction)
    eq = centralupwind.eq
    h2m = faceminus[3]
    h2p = faceplus[3]

    fluxminus = zero(faceminus)
    λmax_m = 0.0; λmin_m = 0.0 #Eigenvalues
    u1m = 0.0; u2m = 0.0       #Velocities

    if h2m > eq.depth_cutoff
        fluxminus = eq(direction, faceminus...)
        λm = compute_eigenvalues(eq, direction, faceminus...)
        λmax_m = maximum(λm); λmin_m = minimum(λm)

        #Desingularized velocities for propagation speeds
        u1m = desingularize(eq, faceminus[1], faceminus[2])
        u2m = desingularize(eq, faceminus[3], faceminus[4])
    end

    fluxplus = zero(faceplus)
    λmax_p = 0.0; λmin_p = 0.0 #Eigenvalues
    u1p = 0.0; u2p = 0.0       #Velocities

    if h2p > eq.depth_cutoff
        fluxplus = eq(direction, faceplus...)
        λp = compute_eigenvalues(eq, direction, faceplus...)
        λmax_p = maximum(λp); λmin_p = minimum(λp)

        #Desingularized velocities for propagation speeds
        u1p = desingularize(eq, faceplus[1], faceplus[2])
        u2p = desingularize(eq, faceplus[3], faceplus[4])
    end

    #Compute aplus and aminus
    aplus  = max(0.0, λmax_m, λmax_p, u1m, u2m, u1p, u2p)
    aminus = min(0.0, λmin_m, λmin_p, u1m, u2m, u1p, u2p)
    if abs(aplus - aminus) < eq.desingularizing_kappa
        return zero(faceminus), zero(aminus)
    end
    F = (aplus .* fluxminus .- aminus .* fluxplus)./( aplus - aminus) .+ ((aplus .* aminus) ./( aplus - aminus)) .* (faceplus .- faceminus)
    if h2m < eq.depth_cutoff && h2p < eq.depth_cutoff
        return F, zero(aplus)
    end
    
    return F, max(abs(aplus), abs(aminus))
end
