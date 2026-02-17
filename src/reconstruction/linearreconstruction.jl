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



# Two-Layer reconstruction (EQUILIBRIUM STORAGE):
# Reconstruct in equilibrium variables, where w = h2 + B, and KEEP w at faces.
# Physical h2 is computed later in the flux as h2_face = w_face - B_face.
function reconstruct!(backend, linRec::LinearLimiterReconstruction, output_left, output_right, input_conserved, grid::Grid, eq::AllTwoLayerSWE, direction::Direction)
    @assert grid.ghostcells[1] > 1
    lim = linRec.limiter
    function fix_slope_w(slope, fix_val, ::TwoLayerShallowWaterEquations1D)
        return typeof(slope)(slope[1], slope[2], fix_val, slope[4])  # (h1,q1,w,q2)
    end
    function fix_slope_w(slope, fix_val, ::TwoLayerShallowWaterEquations2D)
        return typeof(slope)(slope[1], slope[2], slope[3], fix_val, slope[5], slope[6])  # (h1,q1,p1,w,q2,p2)
    end

    w_of(V, ::TwoLayerShallowWaterEquations1D) = V[3]; w_of(V, ::TwoLayerShallowWaterEquations2D) = V[4]

    @fvmloop for_each_inner_cell(backend, grid, direction; ghostcells=1) do ileft, imiddle, iright
        s = slope(lim, input_conserved[ileft], input_conserved[imiddle], input_conserved[iright])

        B_left  = B_face_left(eq.B,  imiddle, direction)
        B_right = B_face_right(eq.B, imiddle, direction)

        w  = w_of(input_conserved[imiddle], eq)
        sw = w_of(s, eq)

        if (w - 0.5 * sw < B_left)
            sw = 2*(w - B_left)
            s = fix_slope_w(s, sw, eq)
        elseif (w + 0.5 * sw < B_right)
            sw = 2*(B_right - w)
            s = fix_slope_w(s, sw, eq)
        end


        output_left[imiddle]  = input_conserved[imiddle] .- 0.5 .* s
        output_right[imiddle] = input_conserved[imiddle] .+ 0.5 .* s
    end

    return nothing
end

