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

function compute_flux!(backend, F::NumericalFlux, output, left, right, wavespeeds, grid, equation::Equation, direction)
    Δx = compute_dx(grid, direction)

    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        F_right, speed_right = F(right[imiddle], left[iright], direction)
        F_left, speed_left = F(right[ileft], left[imiddle], direction)
        output[imiddle] -= 1 / Δx * (F_right - F_left)
        wavespeeds[imiddle] = max(speed_right, speed_left)
        nothing
    end

    return maximum(wavespeeds)
end

function compute_flux!(backend, F::NumericalFlux, output, left, right, wavespeeds,
                       grid, equation::AllTwoLayerSWE, direction)
    Δx = compute_dx(grid, direction)
    B  = equation.B

    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        Bface_right = B_face_right(B, imiddle, direction)
        Bface_left  = B_face_left( B, imiddle, direction) 

        F_right, speed_right = F(equation, right[imiddle], left[iright],   direction, Bface_right)
        F_left,  speed_left  = F(equation, right[ileft],   left[imiddle], direction, Bface_left)

        output[imiddle] -= (F_right - F_left) / Δx
        wavespeeds[imiddle] = max(speed_right, speed_left)
        nothing
    end

    return maximum(wavespeeds)
end


function compute_flux!(backend, F::PathConservativeCentralUpwind, output, left, right, wavespeeds, 
                        grid, equation::AllTwoLayerSWE, direction)
    Δx = compute_dx(grid, direction)
    B  = equation.B

    @fvmloop for_each_inner_cell(backend, grid, direction) do ileft, imiddle, iright
        Zm_r = B_face_right(B, imiddle, direction)  # Z^-_{j+1/2}
        Zp_r = B_face_left( B, iright,  direction)  # Z^+_{j+1/2}
        
        Zm_l = B_face_right(B, ileft,   direction)  # Z^-_{j-1/2}
        Zp_l = B_face_left( B, imiddle, direction)  # Z^+_{j-1/2}

        # Reconstructed states at interfaces
        Um_r = right[imiddle]; Up_r = left[iright]; Um_l = right[ileft]; Up_l = left[imiddle]

        # H and speeds at interfaces
        H_r, aplus_r, aminus_r = F(Um_r, Up_r, direction, Zm_r, Zp_r)
        H_l, aplus_l, aminus_l = F(Um_l, Up_l, direction, Zm_l, Zp_l)

        denom_r = aplus_r - aminus_r; denom_l = aplus_l - aminus_l

        # Path integrals at interfaces (BΨ + SΨ)
        Dpsi_r = compute_path_integral(equation, Um_r, Up_r, Zm_r, Zp_r)
        Dpsi_l = compute_path_integral(equation, Um_l, Up_l, Zm_l, Zp_l)

        # PCCU interface corrections from (4.13)
        corr_r = (abs(denom_r) < equation.desingularizing_kappa) ? zero(Dpsi_r) : (aminus_r/denom_r)*Dpsi_r
        corr_l = (abs(denom_l) < equation.desingularizing_kappa) ? zero(Dpsi_l) : (aplus_l /denom_l)*Dpsi_l

        #Update cell average with flux difference and PCCU correction
        output[imiddle] -= (H_r - H_l + corr_r - corr_l) / Δx

        wavespeeds[imiddle] = max(max(abs(aplus_r), abs(aminus_r)), max(abs(aplus_l), abs(aminus_l)))
        nothing
    end
    return maximum(wavespeeds)
end



include("swe/centralupwind.jl")
include("swe/pathconservative_CU.jl")
include("advection/godunov.jl")
include("advection/rusanov.jl")
include("burgers/godunov.jl")
include("burgers/rusanov.jl")

