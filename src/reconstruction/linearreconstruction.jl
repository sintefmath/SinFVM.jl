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


struct LinearReconstruction <: Reconstruction
    theta::Float64
    LinearReconstruction(theta=1.2) = new(theta)
end

#Adding a new reconstruction for any limiter in limiters.jl
struct LinearLimiterReconstruction{L<:Limiter} <: Reconstruction
    limiter::L
end
LinearLimiterReconstruction(lim::L) where {L<:Limiter} = LinearLimiterReconstruction{L}(lim)


# 3-argument minmod
function minmod(a, b, c)
    if (a > 0) && (b > 0) && (c > 0)
        return min(a, b, c)
    elseif (a < 0) && (b < 0) && (c < 0)
        return max(a, b, c)
    end
    return zero(a)
end

function minmod_slope(left, center, right, theta)
    forward_diff  = right .- center
    backward_diff = center .- left
    central_diff  = (forward_diff .+ backward_diff) ./ 2.0
    return minmod.(theta .* forward_diff, central_diff, theta .* backward_diff)
end


function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, direction::Direction)
    @assert grid.ghostcells[1] > 1
    # NOTE: dx cancel, as the slope depends on 1/dx and face values depend on dx*slope
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        slope = minmod_slope.(input_conserved[ileft], input_conserved[imiddle], input_conserved[iright], linRec.theta)
        output_left[imiddle] = input_conserved[imiddle] .- 0.5 .* slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* slope
    end
end
function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, ::Equation, direction::Direction)
    reconstruct!(backend, linRec, output_left, output_right, input_conserved, grid, direction)
end

#Adds one generic reconstruction for arbitrary limiter in limiters.jl
function reconstruct!(backend, linRec::LinearLimiterReconstruction, output_left, output_right, input_conserved, grid::Grid, direction::Direction)
    @assert grid.ghostcells[1] > 1
    lim = linRec.limiter
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        s = slope(lim, input_conserved[ileft], input_conserved[imiddle], input_conserved[iright])
        output_left[imiddle]  = input_conserved[imiddle] .- 0.5 .* s
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* s
    end
end

function reconstruct!(backend, linRec::LinearLimiterReconstruction, output_left, output_right,
                      input_conserved, grid::Grid, ::Equation, direction::Direction)
    reconstruct!(backend, linRec, output_left, output_right, input_conserved, grid, direction)
end


function reconstruct!(backend, linRec::LinearReconstruction, output_left, output_right, input_conserved, grid::Grid, eq::AllPracticalSWE, direction::Direction)
    @assert grid.ghostcells[1] > 1

    w_input = input_conserved.h
    h_left = output_left.h
    h_right = output_right.h

    function fix_slope(slope, fix_val, ::ShallowWaterEquations1D)
        return typeof(slope)(fix_val, slope[2])
    end
    function fix_slope(slope, fix_val, ::ShallowWaterEquations)
        return typeof(slope)(fix_val, slope[2], slope[3])
    end

    # input_conserved is (w, hu)
    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        # 1) Obtain slope of (w, hu)
        slope = minmod_slope.(input_conserved[ileft], input_conserved[imiddle], input_conserved[iright], linRec.theta)
        B_left = B_face_left(eq.B, imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)
        w = w_input[imiddle]

        # 2) Adjust slope of water
        if (w - 0.5 * slope[1] < B_left)
            # Negative h on left face
            #TODO: uncomment and fix
            slope = fix_slope(slope, 2.0 * (w - B_left), eq)
            #slope[1] = 2.0*(w_input[imiddle] - eq.B[imiddle])
        elseif (w + 0.5 * slope[1] < B_right)
            # Negative h on right face
            #TODO:uncomment and fix
            slope = fix_slope(slope, 2.0 * (B_right - w), eq)
            #slope[1] = 2.0*(eq.B[imiddle] - w_input[imiddle])
        end

        # 3) Reconstruct face values (w, hu)
        output_left[imiddle] = input_conserved[imiddle] .- 0.5 .* slope
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* slope

        # 4) Return face values (h, hu)
        h_left[imiddle] -= B_left
        h_right[imiddle] -= B_right
    end
    nothing
end

# Two-layer SWE 1D reconstruction (PHYSICAL storage with well-balanced lift):
# input  (cell values):  U = (h1, q1, h2, q2)
# reconstruct in:        V = (h1, q1, w,  q2) with w = h2 + B_cell
# output (face values):  U_face = (h1, q1, h2, q2) with h2_face = w_face - B_face

@inline function fix_slope_w(slope, fix_val)
    # adjust only the w-slope (component 3)
    return typeof(slope)(slope[1], slope[2], fix_val, slope[4])
end

function reconstruct!(backend, linRec::LinearLimiterReconstruction,
    output_left, output_right, input_conserved, grid::Grid,
    eq::TwoLayerShallowWaterEquations1D, direction::Direction)

    @assert grid.ghostcells[1] > 1
    lim = linRec.limiter

    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        #Collect cell-centered conserved variables and bathymetry for the 3 cells needed for reconstruction
        Ul = input_conserved[ileft]; Um = input_conserved[imiddle]; Ur = input_conserved[iright]
        Blc = B_cell(eq.B, ileft,   direction); Bmc = B_cell(eq.B, imiddle, direction); Brc = B_cell(eq.B, iright,  direction)

        # Create equilibrium variables for the 3 cells (h1, q1, w, q2) with w = h2 + B_cell
        Vl = typeof(Ul)(Ul[1], Ul[2], Ul[3] + Blc, Ul[4])
        Vm = typeof(Um)(Um[1], Um[2], Um[3] + Bmc, Um[4])
        Vr = typeof(Ur)(Ur[1], Ur[2], Ur[3] + Brc, Ur[4])
        
        # Compute slope in equilibrium variables
        s = slope(lim, Vl, Vm, Vr)

        #Collect face bathymetry (needed for positivity in h2_face = w_face - B_face)
        B_left  = B_face_left(eq.B,  imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)

        # Enforce h2_face >= 0  <=>  w_face >= B_face
        wm = Vm[3]
        sw = s[3]
        if (wm - 0.5*sw < B_left)
            s = fix_slope_w(s, 2.0*(wm - B_left))
        elseif (wm + 0.5*sw < B_right)
            s = fix_slope_w(s, 2.0*(B_right - wm))
        end

        #Reconstruct equilibrium variables at faces
        VL = Vm .- 0.5 .* s ; VR = Vm .+ 0.5 .* s
        h1L, q1L, wL, q2L = VL; h1R, q1R, wR, q2R = VR

        #Convert w -> h2 at faces (physical output)
        h2L = wL - B_left
        h2R = wR - B_right

        #Desingularize if needed (keep momenta consistent)
        if h1L < eq.depth_cutoff
            q1L = h1L * desingularize(eq, h1L, q1L)
        end
        if h1R < eq.depth_cutoff
            q1R = h1R * desingularize(eq, h1R, q1R)
        end
        if h2L < eq.depth_cutoff
            q2L = h2L * desingularize(eq, h2L, q2L)
        end
        if h2R < eq.depth_cutoff
            q2R = h2R * desingularize(eq, h2R, q2R)
        end

        # OUTPUT: physical conserved variables (h1,q1,h2,q2)
        output_left[imiddle]  = typeof(Um)(h1L, q1L, h2L, q2L)
        output_right[imiddle] = typeof(Um)(h1R, q1R, h2R, q2R)
    end

    return nothing
end



###########################################################################################

# ------------------------------------------------------------
# Two-layer SWE 2D:
# Input (cell values):  (h1, q1, p1, w, q2, p2)
# Reconstruction vars:  (h1, q1, p1, w,  q2, p2) with w = h2 + B
# Output (face values): (h1, q1, p1, h2, q2, p2)
# ------------------------------------------------------------

@inline function fix_slope_w(slope, fix_val, ::TwoLayerShallowWaterEquations2D)
    # Only adjust w slope (component 4)
    return typeof(slope)(slope[1], slope[2], slope[3], fix_val, slope[5], slope[6])
end

function reconstruct!(backend, linRec::LinearLimiterReconstruction, output_left, output_right, input_conserved, grid::Grid, eq::TwoLayerShallowWaterEquations2D, direction::Direction,
)
    @assert grid.ghostcells[1] > 1
    lim = linRec.limiter

    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        # physical: U = (h1, q1, p1, h2, q2, p2)
        Ul = input_conserved[ileft]; Um = input_conserved[imiddle]; Ur = input_conserved[iright]

        # cell-centered bottom values
        Blc = B_cell(eq.B, ileft, direction); Bmc = B_cell(eq.B, imiddle, direction); Brc = B_cell(eq.B, iright, direction)

        # equilibrium vars V = (h1, q1, p1, ω, q2, p2), ω = h2 + B
        Vl = typeof(Ul)(Ul[1], Ul[2], Ul[3], Ul[4] + Blc, Ul[5], Ul[6])
        Vm = typeof(Um)(Um[1], Um[2], Um[3], Um[4] + Bmc, Um[5], Um[6])
        Vr = typeof(Ur)(Ur[1], Ur[2], Ur[3], Ur[4] + Brc, Ur[5], Ur[6])

        # slope in equilibrium variables
        s = slope(lim, Vl, Vm, Vr)

        # face bathymetry (needed for positivity in h2_face = ω_face - B_face)
        B_left  = B_face_left(eq.B, imiddle, direction); B_right = B_face_right(eq.B, imiddle, direction)
        ωm = Vm[4]; sω = s[4]

        # enforce ω_face >= B_face  <=> h2_face >= 0
        if (ωm - 0.5*sω < B_left)
            s = fix_slope_ω(s, 2.0 * (ωm - B_left), eq)
        elseif (ωm + 0.5*sω < B_right)
            s = fix_slope_ω(s, 2.0 * (B_right - ωm), eq)
        end

        # reconstruct equilibrium variables at faces
        VL = Vm .- 0.5 .* s
        VR = Vm .+ 0.5 .* s

        # convert ω -> h2 at faces (physical output)
        h2L = VL[4] - B_left
        h2R = VR[4] - B_right

        h1L, q1L, p1L, q2L, p2L = VL[1], VL[2], VL[3], VL[5], VL[6]
        h1R, q1R, p1R, q2R, p2R = VR[1], VR[2], VR[3], VR[5], VR[6]

        # desingularize if needed (keep momenta consistent)
        if h1L < eq.depth_cutoff
            q1L = h1L * desingularize(eq, h1L, q1L)
            p1L = h1L * desingularize(eq, h1L, p1L)
        end
        if h1R < eq.depth_cutoff
            q1R = h1R * desingularize(eq, h1R, q1R)
            p1R = h1R * desingularize(eq, h1R, p1R)
        end
        if h2L < eq.depth_cutoff
            q2L = h2L * desingularize(eq, h2L, q2L)
            p2L = h2L * desingularize(eq, h2L, p2L)
        end
        if h2R < eq.depth_cutoff
            q2R = h2R * desingularize(eq, h2R, q2R)
            p2R = h2R * desingularize(eq, h2R, p2R)
        end

        # OUTPUT: physical conserved variables (h1,q1,p1,h2,q2,p2)
        output_left[imiddle]  = typeof(VL)(h1L, q1L, p1L, h2L, q2L, p2L)
        output_right[imiddle] = typeof(VR)(h1R, q1R, p1R, h2R, q2R, p2R)
    end

    return nothing
end
