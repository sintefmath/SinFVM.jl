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
    physical_flux(eq, U, normal)

Compute the physical flux of the shallow-water equations projected onto the
outward unit `normal`.  `U = (h, hu, hv)`.

```math
\\vec F(U)\\cdot\\hat n = \\begin{pmatrix}
  h u_n \\\\
  h u u_n + \\tfrac12 g h^2 n_x \\\\
  h v u_n + \\tfrac12 g h^2 n_y
\\end{pmatrix},
\\quad u_n = (hu\\, n_x + hv\\, n_y)/h.
```
"""
function physical_flux(eq::ShallowWaterEquationsPure, U::SVector{3,T},
                       normal::SVector{2,Float64}) where {T}
    h  = U[1]
    hu = U[2]
    hv = U[3]
    g  = eq.g
    un = (hu * normal[1] + hv * normal[2]) / h          # normal velocity
    return @SVector [
        h * un,
        hu * un + 0.5 * g * h^2 * normal[1],
        hv * un + 0.5 * g * h^2 * normal[2],
    ]
end

"""
    normal_eigenvalues(eq, U, normal)

Return `(λ_max, λ_min)` — the fastest right- and left-going wave speeds
in the direction of `normal`.
"""
function normal_eigenvalues(eq::ShallowWaterEquationsPure, U::SVector{3,T},
                            normal::SVector{2,Float64}) where {T}
    h  = U[1]
    hu = U[2]
    hv = U[3]
    g  = eq.g
    un = (hu * normal[1] + hv * normal[2]) / h
    c  = sqrt(g * h)
    return (un + c, un - c)
end

"""
    central_upwind_flux(eq, U_minus, U_plus, normal)

Central-upwind numerical flux along the edge with outward unit `normal`.

```math
\\widehat{\\vec F}(U^-,U^+) =
  \\frac{a^+ F(U^-) - a^- F(U^+)}{a^+ - a^-}
  + \\frac{a^+ a^-}{a^+ - a^-}(U^+ - U^-),
```
where ``a^+ = \\max(\\lambda^+_{-}, \\lambda^+_{+}, 0)`` and
``a^- = \\min(\\lambda^-_{-}, \\lambda^-_{+}, 0)``.

Returns `(F_hat, max_speed)`.
"""
function central_upwind_flux(eq::ShallowWaterEquationsPure,
                             U_minus::SVector{3,T},
                             U_plus::SVector{3,T},
                             normal::SVector{2,Float64}) where {T}
    F_minus = physical_flux(eq, U_minus, normal)
    F_plus  = physical_flux(eq, U_plus,  normal)

    λ_max_m, λ_min_m = normal_eigenvalues(eq, U_minus, normal)
    λ_max_p, λ_min_p = normal_eigenvalues(eq, U_plus,  normal)

    a_plus  = max(λ_max_m, λ_max_p, zero(T))
    a_minus = min(λ_min_m, λ_min_p, zero(T))

    denom = a_plus - a_minus
    if abs(denom) < eps(T)
        return zero(U_minus), zero(T)
    end

    F_hat = (a_plus .* F_minus .- a_minus .* F_plus) ./ denom .+
            (a_plus * a_minus / denom) .* (U_plus .- U_minus)
    max_speed = max(abs(a_plus), abs(a_minus))
    return F_hat, max_speed
end

"""
    compute_triangular_fluxes!(backend, output, grid, eq, cell_values, gradients;
                               wavespeeds=nothing)

Accumulate the finite-volume right-hand side for every cell of a
`TriangularGrid`, using the central-upwind numerical flux and
piecewise-linear reconstruction.

At interior edges the reconstructed left and right states are used.
At boundary edges the boundary condition stored in `grid.boundary` is
applied directly — no ghost cells are needed.

All loops over triangles are expressed via the triangular-grid looping
functions [`for_each_cell`](@ref) and [`for_each_cell_neighbor`](@ref)
(wrapped in [`@fvmloop`](@ref)) to keep the implementation backend-aware.

`output` is modified **in-place** and is zero-initialised at the start of
this call.  Returns the maximum wave speed across all edges.
If `wavespeeds` is supplied it is filled with the per-cell maximum wave
speed; otherwise a temporary array is allocated.
"""
function compute_triangular_fluxes!(backend, output::AbstractVector{SVector{3,T}},
                                    grid::TriangularGrid,
                                    eq::ShallowWaterEquationsPure,
                                    cell_values::AbstractVector{SVector{3,T}},
                                    gradients::AbstractVector{SVector{3,SVector{2,T}}};
                                    wavespeeds::Union{Nothing,AbstractVector{T}}=nothing) where {T}
    ncells = number_of_cells(grid)
    ws = wavespeeds === nothing ? zeros(T, ncells) : wavespeeds

    # Zero-initialise outputs
    @fvmloop for_each_cell(backend, grid) do i
        output[i] = zero(SVector{3,T})
        ws[i] = zero(T)
        nothing
    end

    # Accumulate flux contributions edge by edge.  The per-cell inner
    # k-loop is performed inside `for_each_cell_neighbor_kernel`, so the
    # same triangle is handled by a single thread and updates to
    # `output[i]` / `ws[i]` are race-free.
    @fvmloop for_each_cell_neighbor(backend, grid) do i, k, nb
        Ai = grid.areas[i]
        ci = grid.centroids[i]
        normal = grid.edge_normals[i][k]
        len    = grid.edge_lengths[i][k]

        vi1 = grid.triangles[i][k]
        vi2 = grid.triangles[i][k % 3 + 1]
        edge_mid = (grid.nodes[vi1] + grid.nodes[vi2]) / 2.0

        U_minus = evaluate_reconstruction(cell_values[i], gradients[i], ci, edge_mid)

        if nb != 0
            cj = grid.centroids[nb]
            U_plus = evaluate_reconstruction(cell_values[nb], gradients[nb], cj, edge_mid)
        else
            U_plus = _boundary_state(grid.boundary, U_minus, normal)
        end

        F_hat, speed = central_upwind_flux(eq, U_minus, U_plus, normal)

        output[i] -= (len / Ai) .* F_hat
        ws[i] = max(ws[i], speed)
        nothing
    end

    return maximum(ws)
end

# Backwards-compatible wrapper: uses a default CPU backend.
function compute_triangular_fluxes!(output::AbstractVector{SVector{3,T}},
                                    grid::TriangularGrid,
                                    eq::ShallowWaterEquationsPure,
                                    cell_values::AbstractVector{SVector{3,T}},
                                    gradients::AbstractVector{SVector{3,SVector{2,T}}}) where {T}
    return compute_triangular_fluxes!(make_cpu_backend(), output, grid, eq, cell_values, gradients)
end

# Dispatch helpers for the two supported boundary types
_boundary_state(::OutflowBC, U, _normal) = boundary(OutflowBC(), U)
_boundary_state(bc::TriangularWallBC, U, normal) = boundary(bc, U, normal)
