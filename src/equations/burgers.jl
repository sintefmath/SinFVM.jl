
struct Burgers <: Equation end

(::Burgers)(::XDIRT, u) = @SVector [0.5 * u .^ 2]

compute_max_abs_eigenvalue(::Burgers, ::XDIRT, u) = abs(first(u))
conserved_variable_names(::Type{Burgers}) = (:u,)