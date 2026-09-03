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

import ForwardDiff

# Threading the floating point type through the parameter layer.
#
# The *state* layer of the package has always been generic: `create_buffer` uses
# `backend.realtype` and `Volume` picks its `RealType` up from `eltype` of the buffer.
# The *parameter* layer was not: grids, equation constants, reconstruction limiters and
# source term parameters all defaulted to `Float64`, and those values are `Adapt`ed
# straight into kernels. That is harmless on the CPU (arithmetic just promotes) and on
# CUDA (which supports `Float64`), but Metal rejects `Float64` anywhere in a kernel --
# a `Float64` struct field fails to compile with "unsupported use of double value".
#
# `convert_realtype(R, x)` rewrites the parameter layer of `x` to use `R`. It is applied
# once, at simulation setup, in the `ConservedSystem` and `Simulator` constructors, so
# no caller has to thread the type through by hand.

"""
    base_float(T)

The plain floating point type underlying `T`.

Used to separate the *state* element type (which may be a `ForwardDiff.Dual`, see
`test/test_swe_ad.jl`) from the type used for geometry and physical parameters, which
must stay a plain float: differentiating a simulation with respect to a wall height
should not turn `grid.Δ` into a dual number.
"""
base_float(::Type{T}) where {T<:AbstractFloat} = T
base_float(::Type{ForwardDiff.Dual{Tag,V,N}}) where {Tag,V,N} = base_float(V)
base_float(::Type{T}) where {T<:Real} = Float64

"""
    convert_realtype(R, x)

Return `x` with every plain floating point value it carries converted to `R`.

Anything not recognised as a float, an array of floats, or one of the package's
parameter structs is returned untouched. In particular `ForwardDiff.Dual` values pass
through unchanged, because `Dual <: Real` but not `Dual <: AbstractFloat`.
"""
convert_realtype(::Type{R}, x) where {R<:AbstractFloat} = x
convert_realtype(::Type{R}, ::Nothing) where {R<:AbstractFloat} = nothing
convert_realtype(::Type{R}, x::AbstractFloat) where {R<:AbstractFloat} = convert(R, x)

# `SVector`/`SMatrix`: keep the static size, change only the element type.
convert_realtype(::Type{R}, x::StaticArray) where {R<:AbstractFloat} =
    (eltype(x) <: AbstractFloat && eltype(x) !== R) ? similar_type(typeof(x), R)(x) : x

# Plain arrays, including arrays of `SVector`s (the initial condition layout).
function convert_realtype(::Type{R}, x::AbstractArray) where {R<:AbstractFloat}
    converted_eltype(R, eltype(x)) === eltype(x) && return x
    return map(y -> convert_realtype(R, y), x)
end

"""
    converted_eltype(R, T)

What `convert_realtype(R, ::T)` would produce, as a type. Lets callers skip the copy
when there is nothing to convert.
"""
converted_eltype(::Type{R}, ::Type{<:AbstractFloat}) where {R<:AbstractFloat} = R
converted_eltype(::Type{R}, ::Type{S}) where {R<:AbstractFloat,S<:StaticArray} =
    eltype(S) <: AbstractFloat ? similar_type(S, R) : S
converted_eltype(::Type{R}, ::Type{S}) where {R<:AbstractFloat,S} = S

# Identity when the grid already has the requested type, so the defensive conversions in
# the kernel launchers and in `Volume` are free rather than rebuilding the struct.
convert_realtype(::Type{R}, g::CartesianGrid{d,B,d2,R}) where {R<:AbstractFloat,d,B,d2} = g
convert_realtype(::Type{R}, g::CartesianGrid) where {R<:AbstractFloat} =
    CartesianGrid(g.ghostcells, g.totalcells, g.boundary,
        convert_realtype(R, g.extent), convert_realtype(R, g.Δ))

convert_realtype(::Type{R}, r::LinearReconstruction) where {R<:AbstractFloat} =
    LinearReconstruction(convert_realtype(R, r.theta))

convert_realtype(::Type{R}, f::CentralUpwind) where {R<:AbstractFloat} =
    CentralUpwind(convert_realtype(R, f.eq))
convert_realtype(::Type{R}, f::Godunov) where {R<:AbstractFloat} =
    Godunov(convert_realtype(R, f.eq))
convert_realtype(::Type{R}, f::Rusanov) where {R<:AbstractFloat} =
    Rusanov(convert_realtype(R, f.eq))

convert_realtype(::Type{R}, eq::ShallowWaterEquations) where {R<:AbstractFloat} =
    ShallowWaterEquations(convert_realtype(R, eq.B);
        ρ=convert_realtype(R, eq.ρ), g=convert_realtype(R, eq.g),
        depth_cutoff=convert_realtype(R, eq.depth_cutoff),
        desingularizing_kappa=convert_realtype(R, eq.desingularizing_kappa))

convert_realtype(::Type{R}, eq::ShallowWaterEquations1D) where {R<:AbstractFloat} =
    ShallowWaterEquations1D(convert_realtype(R, eq.B);
        ρ=convert_realtype(R, eq.ρ), g=convert_realtype(R, eq.g),
        depth_cutoff=convert_realtype(R, eq.depth_cutoff),
        desingularizing_kappa=convert_realtype(R, eq.desingularizing_kappa))

convert_realtype(::Type{R}, eq::ShallowWaterEquationsPure) where {R<:AbstractFloat} =
    ShallowWaterEquationsPure(convert_realtype(R, eq.ρ), convert_realtype(R, eq.g))
convert_realtype(::Type{R}, eq::ShallowWaterEquations1DPure) where {R<:AbstractFloat} =
    ShallowWaterEquations1DPure(convert_realtype(R, eq.ρ), convert_realtype(R, eq.g))

convert_realtype(::Type{R}, B::ConstantBottomTopography) where {R<:AbstractFloat} =
    ConstantBottomTopography(convert_realtype(R, B.B))
convert_realtype(::Type{R}, B::BottomTopography1D) where {R<:AbstractFloat} =
    BottomTopography1D(convert_realtype(R, B.B); should_never_be_called=nothing)
convert_realtype(::Type{R}, B::BottomTopography2D) where {R<:AbstractFloat} =
    BottomTopography2D(convert_realtype(R, B.B); should_never_be_called=nothing)

convert_realtype(::Type{R}, f::ImplicitFriction) where {R<:AbstractFloat} =
    ImplicitFriction(; Cz=convert_realtype(R, f.Cz), friction_function=f.friction_function)

convert_realtype(::Type{R}, r::ConstantRain) where {R<:AbstractFloat} =
    ConstantRain(convert_realtype(R, r.rain_rate))
convert_realtype(::Type{R}, r::TimeDependentRain) where {R<:AbstractFloat} =
    TimeDependentRain(; rain_rates=convert_realtype(R, r.rain_rates),
        time=convert_realtype(R, r.time))
# The user supplied `rain_function` is arbitrary code and is left alone; it is the
# caller's job to keep it generic in `t`. The grid it closes over is converted.
convert_realtype(::Type{R}, r::FunctionalRain) where {R<:AbstractFloat} =
    FunctionalRain(r.rain_function, convert_realtype(R, r.grid))

convert_realtype(::Type{R}, f::HortonInfiltration) where {R<:AbstractFloat} =
    HortonInfiltration(convert_realtype(R, f.f0), convert_realtype(R, f.fc),
        convert_realtype(R, f.k), convert_realtype(R, f.factor);
        should_never_be_called=nothing)
convert_realtype(::Type{R}, f::ConstantInfiltration) where {R<:AbstractFloat} =
    ConstantInfiltration(convert_realtype(R, f.infiltration_rate))
