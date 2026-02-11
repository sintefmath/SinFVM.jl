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



function (centralupwind::CentralUpwind)(
    ::TwoLayerShallowWaterEquations1D,
    faceminus,
    faceplus,
    direction::Direction,
    Bface,
)
    eq = centralupwind.eq
    nvars = length(faceminus)
    @assert nvars == length(faceplus)

    # Indices depending on dimension
    if nvars == 5 # 1D: (h1,q1,h2,q2,B) at faces 
        h1idx = 1; m1idx = 2; h2idx = 3; m2idx = 4
    elseif nvars == 7 # 2D (h1,q1,p1,h2,q2,p2,B) at faces
        h1idx = 1; h2idx = 4; m1idx = _m1_idx(direction); m2idx = _m2_idx(direction)
    else
        throw(ArgumentError("Unsupported state size $nvars for two-layer CentralUpwind"))
    end

    h2m = faceminus[h2idx]
    h2p = faceplus[h2idx]

    fluxminus = zero(faceminus)
    λmax_m = 0.0; λmin_m = 0.0
    u1m = 0.0; u2m = 0.0

    if h2m > eq.depth_cutoff
        fluxminus = eq(direction, faceminus..., Bface)      # <-- pass Bface
        λm = compute_eigenvalues(eq, direction, faceminus...)  # eigenvalues don't need B
        λmax_m = maximum(λm)
        λmin_m = minimum(λm)

        u1m = desingularize(eq, faceminus[h1idx], faceminus[m1idx])
        u2m = desingularize(eq, faceminus[h2idx], faceminus[m2idx])
    end

    fluxplus = zero(faceplus)
    λmax_p = 0.0; λmin_p = 0.0
    u1p = 0.0; u2p = 0.0

    if h2p > eq.depth_cutoff
        fluxplus = eq(direction, faceplus..., Bface)        # <-- pass Bface
        λp = compute_eigenvalues(eq, direction, faceplus...)
        λmax_p = maximum(λp); λmin_p = minimum(λp)

        u1p = desingularize(eq, faceplus[h1idx], faceplus[m1idx])
        u2p = desingularize(eq, faceplus[h2idx], faceplus[m2idx])
    end

    aplus  = max(0.0, λmax_m, λmax_p, u1m, u2m, u1p, u2p)
    aminus = min(0.0, λmin_m, λmin_p, u1m, u2m, u1p, u2p)

    denom = aplus - aminus
    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), 0.0
    end

    F = (aplus .* fluxminus .- aminus .* fluxplus) ./ denom .+
        ((aplus .* aminus) ./ denom) .* (faceplus .- faceminus)

    if h2m < eq.depth_cutoff && h2p < eq.depth_cutoff
        return F, 0.0
    end

    return F, max(abs(aplus), abs(aminus))
end


#Need to make another one for 2D since we need to pass in the bottom in equation and hence need to pass in an index to find facevalues in the flux
# Direction to pick momentum components in 2D
@inline _m1_idx(::XDIRT) = 2  # q1
@inline _m1_idx(::YDIRT) = 3  # p1
@inline _m2_idx(::XDIRT) = 5  # q2
@inline _m2_idx(::YDIRT) = 6  # p2

function (centralupwind::CentralUpwind)(faceminus, faceplus, direction::Direction, I::CartesianIndex)
    centralupwind(centralupwind.eq, faceminus, faceplus, direction, I)
end

function (centralupwind::CentralUpwind)(eq::SinFVM.TwoLayerShallowWaterEquations2D,
                                        faceminus, faceplus, direction::Direction,
                                        I::CartesianIndex)
    @assert length(faceminus) == 6
    @assert length(faceplus)  == 6

    # scalar bottom at this face (works for ConstantBottomTopography and BottomTopography2D)
    bL = SinFVM.B_face_left(eq.B, I, direction)
    bR = SinFVM.B_face_right(eq.B, I, direction)

    # indices
    h1idx = 1
    h2idx = 4
    m1idx = _m1_idx(direction)
    m2idx = _m2_idx(direction)

    h2m = faceminus[h2idx]
    h2p = faceplus[h2idx]

    fluxminus = zero(faceminus)
    λmax_m = 0.0; λmin_m = 0.0
    u1m = 0.0; u2m = 0.0

    if h2m > eq.depth_cutoff
        fluxminus = eq(direction, faceminus..., bL)           # <-- 2D uses b
        λm = compute_eigenvalues(eq, direction, faceminus...) # keep eigenvalues signature unchanged
        λmax_m = maximum(λm)
        λmin_m = minimum(λm)

        u1m = desingularize(eq, faceminus[h1idx], faceminus[m1idx])
        u2m = desingularize(eq, faceminus[h2idx], faceminus[m2idx])
    end

    fluxplus = zero(faceplus)
    λmax_p = 0.0; λmin_p = 0.0
    u1p = 0.0; u2p = 0.0

    if h2p > eq.depth_cutoff
        fluxplus = eq(direction, faceplus..., bR)             # <-- 2D uses b
        λp = compute_eigenvalues(eq, direction, faceplus...)
        λmax_p = maximum(λp)
        λmin_p = minimum(λp)

        u1p = desingularize(eq, faceplus[h1idx], faceplus[m1idx])
        u2p = desingularize(eq, faceplus[h2idx], faceplus[m2idx])
    end

    aplus  = max(0.0, λmax_m, λmax_p, u1m, u2m, u1p, u2p)
    aminus = min(0.0, λmin_m, λmin_p, u1m, u2m, u1p, u2p)

    denom = aplus - aminus
    if abs(denom) < eq.desingularizing_kappa
        return zero(faceminus), 0.0
    end

    F = (aplus .* fluxminus .- aminus .* fluxplus) ./ denom .+
        ((aplus .* aminus) ./ denom) .* (faceplus .- faceminus)

    if h2m < eq.depth_cutoff && h2p < eq.depth_cutoff
        return F, 0.0
    end

    return F, max(abs(aplus), abs(aminus))
end