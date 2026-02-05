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
                      input_conserved, grid::Grid, eq::Equation, direction::Direction)
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

#Two-layer SWE analogue of the above, where we reconstruct (h1, q1, ω, q2) with ω = h2 + B.
@inline function fix_slope_ω(slope, fix_val, ::TwoLayerShallowWaterEquations1D)
    return typeof(slope)(slope[1], slope[2], fix_val, slope[4])
end

@inline function fix_slope_ω(slope, fix_val, ::TwoLayerShallowWaterEquations1D)
    # Only adjust ω slope (component 3)
    return typeof(slope)(slope[1], slope[2], fix_val, slope[4])
end


function reconstruct!(
    backend,
    linRec::LinearLimiterReconstruction,
    output_left,
    output_right,
    input_conserved,
    grid::Grid,
    eq::TwoLayerShallowWaterEquations1D,
    direction::Direction
)
    @assert grid.ghostcells[1] > 1
    lim = linRec.limiter

    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        # input_conserved = (h1, q1, ω, q2)
        s = slope(lim,input_conserved[ileft], input_conserved[imiddle], input_conserved[iright])
        B_left  = B_face_left(eq.B, imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)
        ω  = input_conserved[imiddle][3]
        sω = s[3]

        # --- well-balanced positivity of h2 = ω - B ---
        if (ω - 0.5*sω < B_left)
            s = fix_slope_ω(s, 2.0*(ω - B_left), eq)
        elseif (ω + 0.5*sω < B_right)
            s = fix_slope_ω(s, 2.0*(B_right - ω), eq)
        end

        # Reconstruct equilibrium variables
        UL = input_conserved[imiddle] .- 0.5 .* s
        UR = input_conserved[imiddle] .+ 0.5 .* s

        # Convert ω → h2 at faces
        h2L = UL[3] - B_left
        h2R = UR[3] - B_right

        h1L, q1L, q2L = UL[1], UL[2], UL[4]
        h1R, q1R, q2R = UR[1], UR[2], UR[4]

        # Desingularize and recalculate momenta
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

        # Output physical face states
        output_left[imiddle]  = typeof(UL)(h1L, q1L, h2L, q2L)
        output_right[imiddle] = typeof(UR)(h1R, q1R, h2R, q2R)
    end
    nothing
end
