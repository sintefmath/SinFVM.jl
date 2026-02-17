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



function (centralupwind::CentralUpwind)(faceminus, faceplus, direction::Direction, Bface)
    centralupwind(centralupwind.eq, faceminus, faceplus, direction, Bface)
end

function (centralupwind::CentralUpwind)(eq::AllTwoLayerSWE, faceminus, faceplus, direction::Direction, Bface)
    nvars = length(faceminus)
    @assert nvars == length(faceplus)

    if nvars == 4 # 1D: (h1, q1, w, q2)
    h1idx = 1; m1idx = 2; widx  = 3; m2idx = 4 

    elseif nvars == 6 # 2D: (h1, q1, p1, w, q2, p2)
        h1idx = 1; widx  = 4
        if direction == XDIR # Update x momentum, in both layers
            m1idx = 2; m2idx = 5 
        elseif direction == YDIR # Update y momentum, in both layers
            m1idx = 3; m2idx = 6  
        else
            throw(ArgumentError("Unsupported direction $direction"))
        end
    else
        throw(ArgumentError("Unsupported state size $nvars for two-layer CentralUpwind"))
    end

    # physical depths at this interface
    h1m = faceminus[h1idx]; h1p = faceplus[h1idx] #h1 at face
    h2m = faceminus[widx] - Bface; h2p = faceplus[widx]  - Bface #h2 at face, using w-B

    #Checking wet/dry states using phisical depths
    wet_m = (h1m > eq.depth_cutoff) && (h2m > eq.depth_cutoff)
    wet_p = (h1p > eq.depth_cutoff) && (h2p > eq.depth_cutoff)

    #Minus state at face
    fluxminus = zero(faceminus)
    λmin_m = 0.0; λmax_m = 0.0; u1m = 0.0; u2m = 0.0
    if wet_m
        fluxminus = eq(direction, faceminus..., Bface) #Compute flux using face values and equation
        λm = compute_eigenvalues(eq, direction, h1m, faceminus[m1idx], h2m, faceminus[m2idx]) #Compute eigenvalues using correct momentum
        λmin_m = minimum(λm); λmax_m = maximum(λm) #Find min and max eigenvalues for minus state

        #Desingularize directional velocities for minus state
        u1m = desingularize(eq, h1m, faceminus[m1idx])
        u2m = desingularize(eq, h2m, faceminus[m2idx])
    end

    #Plus state at face
    fluxplus = zero(faceplus)
    λmin_p = 0.0; λmax_p = 0.0; u1p = 0.0; u2p = 0.0
    if wet_p
        fluxplus = eq(direction, faceplus..., Bface) #Compute flux using face values and equation
        λp = compute_eigenvalues(eq, direction, h1p, faceplus[m1idx], h2p, faceplus[m2idx]) #Compute eigenvalues using correct momentum
        λmin_p = minimum(λp); λmax_p = maximum(λp) #Find min and max eigenvalues for plus state

        #Desingularize directional velocities for plus state
        u1p = desingularize(eq, h1p, faceplus[m1idx])
        u2p = desingularize(eq, h2p, faceplus[m2idx])
    end

    #Bounds using eigenvalues and velocities for given direction
    aplus  = max(0.0, λmax_m, λmax_p, u1m, u2m, u1p, u2p)
    aminus = min(0.0, λmin_m, λmin_p, u1m, u2m, u1p, u2p)

    denom = aplus - aminus
    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), zero(aminus)
    end
    
    #Calculate the flux over the interface
    F = ((aplus*fluxminus - aminus*fluxplus) / denom) + ((aplus*aminus)/denom) * (faceplus - faceminus)

    if !wet_m && !wet_p
        return F, zero(aplus)
    end

    return F, max(abs(aplus), abs(aminus))
end
