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

struct PathConservativeCentralUpwind{E<:AllSWE} <: NumericalFlux
    eq::E
end

Adapt.@adapt_structure PathConservativeCentralUpwind

#Index helper for 1D and 2D generalization
@inline function _twolayer_indices(nvars::Int, direction::Direction)
    if nvars == 4 # 1D: (h1, q1, w, q2)
        h1idx = 1; widx = 3; m1idx = 2; m2idx = 4
        return h1idx, widx, m1idx, m2idx
    elseif nvars == 6 # 2D: (h1, q1, p1, w, q2, p2)
        h1idx = 1; widx = 4
        if direction == XDIR
            m1idx = 2; m2idx = 5
        elseif direction == YDIR
            m1idx = 3; m2idx = 6
        else
            throw(ArgumentError("Unsupported direction $direction"))
        end
        return h1idx, widx, m1idx, m2idx
    else
        throw(ArgumentError("Unsupported state size $nvars (expected 4 or 6)"))
    end
end

function compute_path_integral(eq, faceminus, faceplus, Bface_minus, Bface_plus, direction::Direction)
    g = eq.g
    r = eq.ρ1 / eq.ρ2
    nvars = length(faceminus)
    @assert nvars == length(faceplus)
    h1idx, widx, m1idx, m2idx = _twolayer_indices(nvars, direction)
    h1m = faceminus[h1idx]; wm  = faceminus[widx]; h1p = faceplus[h1idx]; wp  = faceplus[widx]

    d_h1   = h1p - h1m
    sum_hw = (h1p + wp + h1m + wm)
    dB     = (Bface_plus - Bface_minus)

    if nvars == 4 # 1D layout: (h1, q1, w, q2)
        Bpsi = @SVector[0.0, 0.5*g*sum_hw*d_h1, 0.0, -0.5*g*r*sum_hw*d_h1]
        Spsi = @SVector[0.0, 0.0, 0.0, -0.5*g*(r*h1p + wp + r*h1m + wm)*dB]

        return Bpsi + Spsi

    else # 2D layout: (h1, q1, p1, w, q2, p2)
        out = zeros(eltype(faceminus), 6)
        out[m1idx] = 0.5*g*sum_hw*d_h1
        out[m2idx] = -0.5*g*r*sum_hw*d_h1 - 0.5*g*(r*h1p + wp + r*h1m + wm)*dB

        return SVector{6, eltype(faceminus)}(out)
    end
end

function (pccu::PathConservativeCentralUpwind)(faceminus, faceplus, direction::Direction, Bface_minus, Bface_plus)
    eq = pccu.eq
    nvars = length(faceminus)
    @assert nvars == length(faceplus)
    h1idx, widx, m1idx, m2idx = _twolayer_indices(nvars, direction)

    # Fluxes evaluated with the corresponding face-bottom
    fluxminus = eq(direction, faceminus..., Bface_minus)
    fluxplus  = eq(direction, faceplus...,  Bface_plus)

    # Physical depths at each side
    h1m = faceminus[h1idx]; h1p = faceplus[h1idx]
    h2m = faceminus[widx] - Bface_minus; h2p = faceplus[widx]  - Bface_plus

    # Eigenvalues: pass the direction-selected momenta
    λm = compute_eigenvalues(eq, direction, h1m, faceminus[m1idx], h2m, faceminus[m2idx])
    λp = compute_eigenvalues(eq, direction, h1p, faceplus[m1idx],  h2p, faceplus[m2idx])

    aplus  = max(maximum(λm), maximum(λp), 0.0)
    aminus = min(minimum(λm), minimum(λp), 0.0)
    denom = aplus - aminus

    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), aplus, aminus
    end

    F = ((aplus * fluxminus - aminus * fluxplus) / denom) + ((aplus * aminus) / denom) * (faceplus - faceminus)

    return F, aplus, aminus
end