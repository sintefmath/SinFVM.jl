abstract type NumericalFlux end

struct Rusanov{EquationType <: Equation} <: NumericalFlux 
    eq::EquationType
end

function (rus::Rusanov)(faceminus, faceplus)
    fluxminus = rus.eq(XDIR, faceminus...)
    fluxplus = rus.eq(XDIR, faceplus...)

    eigenvalue_minus = compute_max_eigenvalue(rus.eq, XDIR, faceminus...)
    eigenvalue_plus = compute_max_eigenvalue(rus.eq, XDIR, faceplus...)

    eigenvalue_max = max(eigenvalue_minus, eigenvalue_plus)

    F = 0.5 .* (fluxminus .+ fluxplus) .- 0.5 * eigenvalue_max .* (faceplus .- faceminus);

    return F
end


struct Godunov{EquationType <: Equation} <: NumericalFlux
    eq::EquationType
end

function (god::Godunov)(faceminus, faceplus)
    f(u) = god.eq(XDIR, u...)
    fluxminus = f(max.(faceminus, zero(faceminus)))
    fluxplus = f(min.(faceplus, zero(faceplus)))
    
    F = max.(fluxminus, fluxplus)
    return F
end



function compute_flux!(F::NumericalFlux, output, left, right, grid, equation::Equation, ::XDIRT)
    Δx = compute_dx(grid)
    for_each_inner_cell(grid) do ileft, imiddle, iright
        output[imiddle] -= 1/Δx *( F(right[imiddle], left[iright]) - F(right[ileft], left[imiddle]))

        nothing
    end
end
