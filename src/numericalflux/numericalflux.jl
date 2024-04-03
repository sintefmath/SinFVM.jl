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


include("swe/centralupwind.jl")
include("burgers/godunov.jl")
include("burgers/rusanov.jl")