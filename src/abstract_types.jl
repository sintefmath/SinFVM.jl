abstract type Equation end

abstract type BoundaryCondition end
abstract type Grid{dimension} end

abstract type NumericalFlux end

abstract type Reconstruction end

abstract type System end

abstract type TimeStepper end

abstract type SourceTerm end

struct SourceTermBottom <: SourceTerm end
struct SourceTermRain <: SourceTerm 
    R
end
struct SourceTermInfiltration <: SourceTerm 
    I
end

# TODO: Define AbstractSimulator as well?

# TODO: Define all none-abstract types here as well?



