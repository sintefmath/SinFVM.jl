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


"""
Compute path integral correction term BΨ + SΨ
Consistent with Castro et al. (2019), eq. (4.28)–(4.29)
"""
function compute_path_integral(eq, faceminus, faceplus, Bface_minus, Bface_plus)
    g = eq.g
    r = eq.ρ1 / eq.ρ2

    h1m = faceminus[1]; wm = faceminus[3]
    h1p = faceplus[1];  wp = faceplus[3]

    # BΨ term
    Bpsi = @SVector[
        0.0,
        0.5*g*(h1p + wp + h1m + wm)*(h1p - h1m),
        0.0,
        -0.5*g*r*(h1p + wp + h1m + wm)*(h1p - h1m)
    ]

    # SΨ term (bottom jump contribution)
    Spsi = @SVector[
        0.0,
        0.0,
        0.0,
        -0.5*g*(r*h1p + wp + r*h1m + wm)*(Bface_plus - Bface_minus)
    ]

    return Bpsi + Spsi
end


function (pccu::PathConservativeCentralUpwind)(faceminus, faceplus, direction::Direction, Bface_minus, Bface_plus)
    eq = pccu.eq

    # Physical fluxes using correct face bottoms
    fluxminus = eq(direction, faceminus..., Bface_minus)
    fluxplus  = eq(direction, faceplus...,  Bface_plus)

    # Physical depths
    h1m = faceminus[1]; h2m = faceminus[3] - Bface_minus
    h1p = faceplus[1];  h2p = faceplus[3]  - Bface_plus

    # Eigenvalues
    λm = compute_eigenvalues(eq, direction, h1m, faceminus[2], h2m, faceminus[4])
    λp = compute_eigenvalues(eq, direction, h1p, faceplus[2],  h2p, faceplus[4])

    # Wave speed bounds
    aplus  = max(maximum(λm), maximum(λp), 0.0)
    aminus = min(minimum(λm), minimum(λp), 0.0)
    denom  = aplus - aminus

    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), aplus, aminus
    end
    F = ((aplus * fluxminus - aminus * fluxplus) / denom) + ((aplus * aminus) / denom) * (faceplus - faceminus)
    return F, aplus, aminus
end