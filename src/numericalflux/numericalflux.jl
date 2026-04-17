# Copyright (c) 2024 SINTEF AS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

"""
    compute_flux!(backend, F, output, left, right, wavespeeds, grid::TriangularGrid, equation, direction)

Specialised `compute_flux!` for triangular grids.  The `direction` argument is
ignored because all edges are processed in a single pass.  Reconstruction
gradients are read from the cache populated by the preceding `reconstruct!`
call, and the central-upwind numerical flux is evaluated on every edge with
boundary conditions applied directly (no ghost cells).

Cell-averaged values are read from `left` (populated by `reconstruct!`).
"""
function compute_flux!(backend, F::NumericalFlux, output, left, right,
                       wavespeeds, grid::TriangularGrid,
                       equation::ShallowWaterEquationsPure, direction)
    ncells = number_of_cells(grid)

    # Extract cell-averaged values from the left buffer (set by reconstruct!)
    # using the triangular looping abstraction (no plain for-loop over cells).
    cell_values = Vector{SVector{3, Float64}}(undef, ncells)
    @fvmloop for_each_cell(backend, grid) do i
        cell_values[i] = left[i]
        nothing
    end

    # Retrieve gradients from cache (set by LinearReconstruction's reconstruct!).
    # When NoReconstruction is used, no cache entry exists and zero gradients
    # are used, giving piecewise-constant reconstruction at face values.
    grid_id = objectid(grid)
    if haskey(_TRI_GRADIENT_CACHE, grid_id)
        gradients = _TRI_GRADIENT_CACHE[grid_id]::Vector{SVector{3, SVector{2, Float64}}}
        delete!(_TRI_GRADIENT_CACHE, grid_id)
    else
        zero_grad = SVector{2,Float64}(0.0, 0.0)
        gradients = [SVector{3, SVector{2, Float64}}(zero_grad, zero_grad, zero_grad) for _ in 1:ncells]
    end

    # Accumulate flux contributions through the triangular edge loop;
    # per-cell wavespeeds are filled in the same pass.
    rhs = Vector{SVector{3, Float64}}(undef, ncells)
    max_speed = compute_triangular_fluxes!(backend, rhs, grid, equation,
                                           cell_values, gradients;
                                           wavespeeds=wavespeeds)

    # Add the accumulated RHS into the output volume via the looping abstraction.
    @fvmloop for_each_cell(backend, grid) do i
        output[i] += rhs[i]
        nothing
    end

    return max_speed
end


include("swe/centralupwind.jl")
include("burgers/godunov.jl")
include("burgers/rusanov.jl")
include("triangular_flux.jl")
