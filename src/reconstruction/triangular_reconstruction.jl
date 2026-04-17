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

"""
    reconstruct_triangular(grid, cell_values)

Compute limited piecewise-linear gradients on a `TriangularGrid`.

For every cell *i* with centroid ``\\mathbf{c}_i`` and value ``\\bar U_i``,
the three candidate gradients are obtained from the planes through
``\\mathbf{c}_i`` and two of its neighbour centroids:

```math
\\vec{g}_k = \\text{(finite difference through two centroids)}.
```

The final gradient is assembled component-wise with the three-argument
minmod limiter

```math
g^{(d)} = \\operatorname{minmod}(g_1^{(d)}, g_2^{(d)}, g_3^{(d)}).
```

Returns `gradients::Vector{SVector{N,SVector{2,Float64}}}` where `N` is
the number of conserved variables, giving the limited gradient for each
variable in each cell.
"""
"""
    reconstruct_triangular(backend, grid, cell_values)

Compute limited piecewise-linear gradients on a `TriangularGrid`.

All loops over triangles are expressed via [`for_each_cell`](@ref)
(wrapped in [`@fvmloop`](@ref)), so this function is backend-aware.

For every cell *i* with centroid ``\\mathbf{c}_i`` and value ``\\bar U_i``,
the three candidate gradients are obtained from the planes through
``\\mathbf{c}_i`` and two of its neighbour centroids:

```math
\\vec{g}_k = \\text{(finite difference through two centroids)}.
```

The final gradient is assembled component-wise with the three-argument
minmod limiter

```math
g^{(d)} = \\operatorname{minmod}(g_1^{(d)}, g_2^{(d)}, g_3^{(d)}).
```

Returns `gradients::Vector{SVector{N,SVector{2,Float64}}}` where `N` is
the number of conserved variables, giving the limited gradient for each
variable in each cell.
"""
function reconstruct_triangular(backend, grid::TriangularGrid,
                                cell_values::AbstractVector{SVector{N,T}}) where {N,T}
    ncells = number_of_cells(grid)
    # Each element stores per-variable 2D gradient
    gradients = Vector{SVector{N,SVector{2,T}}}(undef, ncells)

    zero_grad = SVector{2,T}(zero(T), zero(T))

    @fvmloop for_each_cell(backend, grid) do i
        ci = grid.centroids[i]

        # For each conserved variable, compute the limited gradient using
        # all three candidate gradients independently.
        var_grads = MVector{N, SVector{2,T}}(ntuple(_ -> zero_grad, Val(N)))

        for v in 1:N
            cand_gx = MVector{3,T}(zero(T), zero(T), zero(T))
            cand_gy = MVector{3,T}(zero(T), zero(T), zero(T))
            n_cands = 0
            for k in 1:3
                nb = grid.neighbors[i][k]
                if nb == 0
                    continue
                end
                n_cands += 1
                cj = grid.centroids[nb]
                dc = cj - ci
                dist2 = dc[1]^2 + dc[2]^2
                dU_v = cell_values[nb][v] - cell_values[i][v]
                cand_gx[n_cands] = dU_v * dc[1] / dist2
                cand_gy[n_cands] = dU_v * dc[2] / dist2
            end

            if n_cands >= 3
                gx = minmod(cand_gx[1], cand_gx[2], cand_gx[3])
                gy = minmod(cand_gy[1], cand_gy[2], cand_gy[3])
            elseif n_cands == 2
                gx = minmod(cand_gx[1], cand_gx[2], zero(T))
                gy = minmod(cand_gy[1], cand_gy[2], zero(T))
            elseif n_cands == 1
                gx = cand_gx[1]
                gy = cand_gy[1]
            else
                gx = zero(T)
                gy = zero(T)
            end
            var_grads[v] = SVector{2,T}(gx, gy)
        end

        gradients[i] = SVector{N,SVector{2,T}}(var_grads)
        nothing
    end

    return gradients
end

# Backwards-compatible wrapper: uses a default CPU backend.
function reconstruct_triangular(grid::TriangularGrid,
                                cell_values::AbstractVector{SVector{N,T}}) where {N,T}
    return reconstruct_triangular(make_cpu_backend(), grid, cell_values)
end

"""
    evaluate_reconstruction(cell_value, gradient, centroid, point)

Evaluate the linearly reconstructed value at `point` given the cell
average `cell_value`, its limited `gradient`, and the cell `centroid`.
"""
function evaluate_reconstruction(cell_value::SVector{N,T}, gradient::SVector{N,SVector{2,T}},
                                 centroid::SVector{2,Float64}, point::SVector{2,Float64}) where {N,T}
    dx = point - centroid
    return SVector{N,T}(ntuple(v -> cell_value[v] + gradient[v][1] * dx[1] + gradient[v][2] * dx[2], Val(N)))
end

# Module-level gradient cache for passing data from reconstruct! to compute_flux!
# on triangular grids. Keyed by objectid of the grid.
# NOTE: Not thread-safe.  The current codebase is single-threaded; if
# multi-threaded usage is needed, wrap accesses with a lock.
const _TRI_GRADIENT_CACHE = Dict{UInt, Any}()

"""
    reconstruct!(backend, ::LinearReconstruction, output_left, output_right,
                 input_conserved, grid::TriangularGrid, equation, direction)

Linear reconstruction specialisation for triangular grids.  Computes limited
piecewise-linear gradients via [`reconstruct_triangular`](@ref) and caches
them for the subsequent [`compute_flux!`](@ref) call.  Cell-averaged values
are copied into `output_left`.

Since triangular grids process all edges in a single pass (direction is
irrelevant), the reconstruction is only performed for the first direction;
subsequent directions reuse the cached result.
"""
function reconstruct!(backend, ::LinearReconstruction, output_left, output_right,
                      input_conserved, grid::TriangularGrid, equation::Equation, direction::Direction)
    ncells = number_of_cells(grid)

    # Copy cell values into output_left
    @fvmloop for_each_cell(backend, grid) do i
        output_left[i] = input_conserved[i]
    end

    # Extract cell values for gradient computation via the triangular
    # looping abstraction (no plain for-loop over cells).
    cell_values = Vector{SVector{3, Float64}}(undef, ncells)
    @fvmloop for_each_cell(backend, grid) do i
        cell_values[i] = input_conserved[i]
        nothing
    end

    # Compute gradients and cache them
    gradients = reconstruct_triangular(backend, grid, cell_values)
    _TRI_GRADIENT_CACHE[objectid(grid)] = gradients
end
