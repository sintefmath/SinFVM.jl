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

################################# Two-layer Central Upwind #################################
@inline _m1_idx(::XDIRT) = 2  # q1
@inline _m1_idx(::YDIRT) = 3  # p1
@inline _m2_idx(::XDIRT) = 5  # q2
@inline _m2_idx(::YDIRT) = 6  # p2

function (centralupwind::CentralUpwind)(faceminus, faceplus, direction::Direction, Bface)
    return centralupwind(centralupwind.eq, faceminus, faceplus, direction, Bface)
end

# Guard: prevent accidentally calling 3-arg version for two-layer
function (centralupwind::CentralUpwind)(::AllTwoLayerSWE, faceminus, faceplus, direction::Direction)
    throw(ArgumentError("Two-layer CentralUpwind requires Bface. Call as centralupwind(faceminus, faceplus, direction, Bface)"))
end

function (centralupwind::CentralUpwind)(eq::AllTwoLayerSWE, faceminus, faceplus, direction::Direction, Bface)
    nvars = length(faceminus)
    @assert nvars == length(faceplus)
    if nvars == 4
        # 1D: (h1,q1,w,q2)
        h1idx = 1; m1idx = 2; widx = 3; m2idx = 4
    elseif nvars == 6
        # 2D: (h1,q1,p1,w,q2,p2)
        h1idx = 1; widx = 4
        m1idx = _m1_idx(direction)
        m2idx = _m2_idx(direction)
    else
        throw(ArgumentError("Unsupported state size $nvars for two-layer CentralUpwind"))
    end

    # Convert equilibrium w -> physical h2 at this face
    w_m = faceminus[widx];  w_p = faceplus[widx]
    h2m = w_m - Bface
    h2p = w_p - Bface

    # --- minus state
    fluxminus = zero(faceminus)
    λmax_m = 0.0; λmin_m = 0.0
    u1m = 0.0; u2m = 0.0
    if h2m > eq.depth_cutoff
        # eq(...) expects w and Bface (eq converts to h2 internally)
        fluxminus = eq(direction, faceminus..., Bface)

        # eigenvalues expect physical h2 (not w)
        λm = compute_eigenvalues(eq, direction,
            faceminus[h1idx], faceminus[m1idx], h2m, faceminus[m2idx]
        )
        λmax_m = maximum(λm); λmin_m = minimum(λm)

        u1m = desingularize(eq, faceminus[h1idx], faceminus[m1idx])
        u2m = desingularize(eq, h2m,               faceminus[m2idx])
    end

    # --- plus state
    fluxplus = zero(faceplus)
    λmax_p = 0.0; λmin_p = 0.0
    u1p = 0.0; u2p = 0.0
    if h2p > eq.depth_cutoff
        fluxplus = eq(direction, faceplus..., Bface)

        λp = compute_eigenvalues(eq, direction,
            faceplus[h1idx], faceplus[m1idx], h2p, faceplus[m2idx]
        )
        λmax_p = maximum(λp); λmin_p = minimum(λp)

        u1p = desingularize(eq, faceplus[h1idx], faceplus[m1idx])
        u2p = desingularize(eq, h2p,              faceplus[m2idx])
    end

    aplus  = max(0.0, λmax_m, λmax_p, u1m, u2m, u1p, u2p)
    aminus = min(0.0, λmin_m, λmin_p, u1m, u2m, u1p, u2p)
    denom = aplus - aminus
    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), 0.0
    end

    F = (aplus .* fluxminus .- aminus .* fluxplus) ./ denom .+
        ((aplus * aminus) / denom) .* (faceplus .- faceminus)

    if h2m < eq.depth_cutoff && h2p < eq.depth_cutoff
        return F, 0.0
    end

    return F, max(abs(aplus), abs(aminus))
end


