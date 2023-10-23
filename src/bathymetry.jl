import CUDA
import Adapt
import Meshes


struct Bathymetry{RealVectorType}
    B::RealVectorType
    Bi::RealVectorType

    Bathymetry(B, Bi) = new{typeof(B)}(B, Bi)
end

function Bathymetry(g::Meshes.CartesianGrid, ngc = 2)
    (nx, ny) = size(g)
    ngc = 2
    shape = ((nx + 2 * ngc), (ny + 2 * ngc))

    return Bathymetry(zeros(shape), zeros(shape .+ 1) )
end

CUDA.cu(b::Bathymetry) = Bathymetry(CUDA.cu(b.B), CUDA.cu(b.Bi))

function Adapt.adapt_structure(to, b::Bathymetry)
    B = Adapt.adapt_structure(to, b.B)
    Bi = Adapt.adapt_structure(to, b.Bi)
    
    return Bathymetry(B, Bi)
end